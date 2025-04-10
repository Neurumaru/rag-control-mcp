import os
import httpx
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from typing import List, Dict
import uuid

# --- Configuration ---
# Using configurations from PoC 1 where applicable
DATA_DIR = "../basic_rag/data" # Point to PoC 1 data directory
MCP_SERVER_URL = "http://localhost:5001" # PoC 2 MCP server URL
COLLECTION_NAME = "basic_rag_poc" # Same collection name as PoC 1 for consistency
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma3:12b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Client-side storage for original text chunks
id_to_text: Dict[str, str] = {}

# --- Setup LLM and Embedding Model (Reused from PoC 1) ---
print(f"Setting up LLM: {LLM_MODEL_NAME}")
Settings.llm = Ollama(model=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

print(f"Setting up Embedding Model: {EMBED_MODEL_NAME}")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model # Set globally if other LlamaIndex components need it

# --- Helper Function to Interact with MCP Server ---
async def post_to_mcp(endpoint: str, data: dict):
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{MCP_SERVER_URL}{endpoint}", json=data)
            response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.RequestError as exc:
            print(f"HTTP Request failed: {exc}")
            raise
        except httpx.HTTPStatusError as exc:
            print(f"HTTP Error response {exc.response.status_code} while requesting {exc.request.url!r}. Response: {exc.response.text}")
            raise

async def get_from_mcp(endpoint: str):
    """Helper function to send GET requests to the MCP server."""
    async with httpx.AsyncClient(timeout=10.0) as client: # Shorter timeout for health check
        try:
            response = await client.get(f"{MCP_SERVER_URL}{endpoint}")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            print(f"HTTP GET Request failed: {exc}")
            raise
        except httpx.HTTPStatusError as exc:
            print(f"HTTP GET Error response {exc.response.status_code} while requesting {exc.request.url!r}. Response: {exc.response.text}")
            raise

# --- Indexing Function (using MCP) ---
async def index_documents_via_mcp(data_dir: str, collection_name: str):
    global id_to_text
    print(f"Loading documents from: {data_dir}")
    documents = SimpleDirectoryReader(data_dir).load_data()
    if not documents:
        print("Error: No documents found.")
        return False

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"Generating {len(nodes)} embeddings...")
    embeddings = embed_model.get_text_embedding_batch([node.get_content() for node in nodes], show_progress=True)

    ids = [str(uuid.uuid4()) for _ in nodes] # Generate unique IDs
    metadatas = [node.metadata for node in nodes]
    texts = [node.get_content() for node in nodes]

    # Store text locally mapped by ID
    id_to_text = {ids[i]: texts[i] for i in range(len(ids))}
    print(f"Stored {len(id_to_text)} text chunks locally.")

    print(f"Adding {len(ids)} embeddings to MCP collection '{collection_name}'...")
    add_request_data = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": texts # Also send documents if MCP/ChromaDB supports it
    }
    try:
        response = await post_to_mcp(f"/collections/{collection_name}/add", add_request_data)
        print(f"MCP Add Response: {response}")
        return True
    except Exception as e:
        print(f"Failed to add documents via MCP: {e}")
        return False

# --- Query Function (using MCP) ---
async def query_rag_via_mcp(query: str, collection_name: str):
    print(f"\nQuerying via MCP: '{query}'")

    # 1. Embed the query
    query_embedding = embed_model.get_text_embedding(query)

    # 2. Query MCP for similar document IDs
    query_request_data = {
        "query_embeddings": [query_embedding],
        "n_results": 3, # Get top 3 results
        "include": ["distances", "metadatas"] # IDs are always returned
    }
    try:
        print("Sending query to MCP server...")
        mcp_response = await post_to_mcp(f"/collections/{collection_name}/query", query_request_data)
        print(f"MCP Query Response: {mcp_response}")
    except Exception as e:
        print(f"Failed to query MCP: {e}")
        return

    # 3. Retrieve text chunks using IDs from the local map
    retrieved_ids = mcp_response.get("ids", [[]])[0] # Chroma returns lists within lists
    retrieved_texts = [id_to_text.get(doc_id, "[Content not found]") for doc_id in retrieved_ids]
    retrieved_distances = mcp_response.get("distances", [[]])[0]

    if not retrieved_texts:
        print("No relevant documents found via MCP.")
        return

    print("\n--- Retrieved Context ---")
    for i, text in enumerate(retrieved_texts):
        dist_str = f"{retrieved_distances[i]:.4f}" if i < len(retrieved_distances) else "N/A"
        print(f"- ID: {retrieved_ids[i]}, Distance: {dist_str}\n  Text: {text[:150]}...")
    print("------------------------")

    # 4. Prepare context and call LLM
    context_str = "\n\n".join(retrieved_texts)
    prompt = f"Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"

    print("\nGenerating response with LLM...")
    response = Settings.llm.complete(prompt)

    print("\nLLM Response:")
    print(response.text)

# --- Main Execution ---
import asyncio

async def main():
    # Check health of MCP server first using GET
    try:
        health = await get_from_mcp("/health") # Use GET helper
        if health.get("status") != "ok":
            print("MCP Server health check failed. Aborting.")
            return
        print("MCP Server is healthy.")
    except Exception as e:
        print(f"Could not reach MCP Server at {MCP_SERVER_URL}. Please ensure it's running. Error: {e}")
        return

    # Decide whether to re-index. For PoC, we might re-index every time
    # or add a check (e.g., ask user, check collection status via MCP if implemented)
    # For simplicity, let's re-index each time for this script.
    print("\nAttempting to index documents...")
    indexed = await index_documents_via_mcp(DATA_DIR, COLLECTION_NAME)

    if indexed:
        print("\nIndexing complete. Proceeding to query.")
        # Example queries (similar to PoC 1)
        await query_rag_via_mcp("mcp-rag-control 프로젝트가 무엇인가요?", COLLECTION_NAME)
        await query_rag_via_mcp("이 PoC의 목표는 무엇인가요?", COLLECTION_NAME)
        await query_rag_via_mcp("ChromaDB 서버 모드는 어떻게 사용되나요?", COLLECTION_NAME) # New query relevant to PoC 2
    else:
        print("\nFailed to index documents. Cannot proceed with queries.")

if __name__ == "__main__":
    asyncio.run(main()) 