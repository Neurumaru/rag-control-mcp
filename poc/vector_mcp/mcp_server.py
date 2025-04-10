import chromadb
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import contextlib

# --- Configuration ---
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
MCP_SERVER_HOST = "0.0.0.0"
MCP_SERVER_PORT = 5001 # PoC 2 MCP 서버 포트

# --- Pydantic Models for API ---
class AddRequest(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: Optional[List[Dict[str, Any]]] = None
    documents: Optional[List[str]] = None # ChromaDB can store documents too

class QueryRequest(BaseModel):
    query_embeddings: List[List[float]]
    n_results: int = 5
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    include: List[str] = Field(default=["metadatas", "distances"]) # Default includes

class HealthResponse(BaseModel):
    status: str

# --- Global variable for the client (simplifies startup/dependency for now) ---
# In a more complex app, consider FastAPI's state or lifespan events
chroma_client_instance: Optional[chromadb.Client] = None

def get_chroma_client() -> chromadb.Client:
    """FastAPI Dependency to get the ChromaDB client instance."""
    global chroma_client_instance
    if chroma_client_instance is None:
        # This part will run only when the actual server starts,
        # tests will override this dependency.
        print(f"Initializing ChromaDB HttpClient to {CHROMA_HOST}:{CHROMA_PORT}")
        try:
            chroma_client_instance = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            chroma_client_instance.heartbeat() # Initial connection check
            print("ChromaDB HttpClient initialized successfully.")
        except Exception as e:
            print(f"FATAL: Failed to initialize ChromaDB HttpClient: {e}")
            # Raise an exception or handle appropriately to prevent app start with bad state
            raise RuntimeError(f"Could not connect to ChromaDB: {e}") from e
    return chroma_client_instance

# --- FastAPI App ---
# Use contextlib.asynccontextmanager for lifespan events if needed later
app = FastAPI(title="Vector Search MCP (ChromaDB)")

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check(client: chromadb.Client = Depends(get_chroma_client)):
    """Checks if the MCP server is running and can connect to ChromaDB."""
    try:
        client.heartbeat() # Check connection to ChromaDB
        return {"status": "ok"}
    except Exception as e:
        # Log the error in a real application
        raise HTTPException(status_code=503, detail=f"ChromaDB connection failed: {e}")

@app.post("/collections/{collection_name}/add")
async def add_embeddings(
    collection_name: str,
    request: AddRequest,
    client: chromadb.Client = Depends(get_chroma_client)
):
    """Adds embeddings, documents, and metadatas to a ChromaDB collection."""
    try:
        collection = client.get_or_create_collection(name=collection_name)
        collection.add(
            embeddings=request.embeddings,
            documents=request.documents,
            metadatas=request.metadatas,
            ids=request.ids
        )
        return {"message": f"Added {len(request.ids)} items to collection '{collection_name}'"}
    except Exception as e:
        print(f"Error adding to collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collections/{collection_name}/query")
async def query_collection(
    collection_name: str,
    request: QueryRequest,
    client: chromadb.Client = Depends(get_chroma_client)
):
    """Queries a ChromaDB collection for similar embeddings."""
    try:
        collection = client.get_or_create_collection(name=collection_name)
        results = collection.query(
            query_embeddings=request.query_embeddings,
            n_results=request.n_results,
            where=request.where,
            where_document=request.where_document,
            include=request.include
        )
        return results
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Run Server (for local testing) ---
if __name__ == "__main__":
    # Initialize the client when running directly (optional, Depends handles it)
    # try:
    #     get_chroma_client()
    # except RuntimeError as e:
    #     print(f"Could not start server: {e}")
    #     exit(1)

    print(f"Starting MCP server on {MCP_SERVER_HOST}:{MCP_SERVER_PORT}")
    # Note: When running with uvicorn directly, the dependency injection
    # handles the client creation on the first request if not already initialized.
    uvicorn.run(app, host=MCP_SERVER_HOST, port=MCP_SERVER_PORT) 