import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# --- Configuration ---
DATA_DIR = "data"
VECTOR_STORE_DIR = "vectorstore"
COLLECTION_NAME = "basic_rag_poc"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemma3:12b" # 사용자가 지정한 LLM
OLLAMA_BASE_URL = "http://localhost:11434" # 기본 Ollama URL

# --- Setup LLM and Embedding Model ---
print(f"Setting up LLM: {LLM_MODEL_NAME}")
Settings.llm = Ollama(model=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

print(f"Setting up Embedding Model: {EMBED_MODEL_NAME}")
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# --- Load or Create Vector Index ---
def load_or_create_index(data_dir, vector_store_dir, collection_name):
    vector_store_path = os.path.join(vector_store_dir, collection_name)
    print(f"Checking for existing vector store at: {vector_store_path}")

    if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
        print("Loading existing vector store...")
        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        print("Vector store loaded successfully.")
    else:
        print("Creating new vector store...")
        # Ensure directory exists
        os.makedirs(vector_store_path, exist_ok=True)

        # Load documents
        print(f"Loading documents from: {data_dir}")
        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            print("Error: No documents found in the data directory.")
            return None
        print(f"Loaded {len(documents)} document(s).")

        # Create client and collection
        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index (this will embed and store)
        print("Creating index and embedding documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print("New vector store created and documents embedded.")

    return index

# --- Query Function ---
def query_rag(index, query):
    print(f"\nQuerying index with: '{query}'")
    if index is None:
        print("Error: Index is not available.")
        return

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print("\nResponse:")
    print(response)
    print("\n--- Sources ---")
    for node in response.source_nodes:
        print(f"Node ID: {node.node_id}, Score: {node.score:.4f}")
        # print(f"Content: {node.text[:100]}...") # Optionally print source text snippet
    print("---------------")

# --- Main Execution ---
if __name__ == "__main__":
    # Set relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_DIR)
    vector_store_path_root = os.path.join(script_dir, VECTOR_STORE_DIR)

    index = load_or_create_index(data_path, vector_store_path_root, COLLECTION_NAME)

    if index:
        # Example queries
        query_rag(index, "mcp-rag-control 프로젝트가 무엇인가요?")
        query_rag(index, "이 PoC의 목표는 무엇인가요?")
        query_rag(index, "FAISS는 어떤 기술인가요?")

        # Interactive query loop
        # while True:
        #     user_query = input("\nEnter your query (or type 'quit' to exit): ")
        #     if user_query.lower() == 'quit':
        #         break
        #     query_rag(index, user_query) 