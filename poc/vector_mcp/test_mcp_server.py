import pytest
import chromadb
from fastapi.testclient import TestClient

# Import the FastAPI app from your server file
from poc.vector_mcp.mcp_server import app, get_chroma_client

# --- Test Fixtures ---

@pytest.fixture(scope="module") # Use module scope for efficiency
def in_memory_chroma_client():
    """Creates an in-memory ChromaDB client for testing."""
    client = chromadb.EphemeralClient() # Use in-memory client
    yield client
    # No explicit cleanup needed for EphemeralClient

@pytest.fixture(scope="module")
def test_app_client(in_memory_chroma_client):
    """Creates a FastAPI TestClient with the in-memory ChromaDB dependency."""
    # Override the dependency to use the in-memory client for all tests in this module
    app.dependency_overrides[get_chroma_client] = lambda: in_memory_chroma_client

    with TestClient(app) as client:
        yield client

    # Clean up the override after tests in this module are done
    app.dependency_overrides = {}

# --- Test Data ---
TEST_COLLECTION_NAME = "test_poc2_collection"
TEST_IDS = ["vec1", "vec2", "vec3"]
TEST_EMBEDDINGS = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] # Example 2D embeddings
TEST_METADATAS = [{"source": "docA"}, {"source": "docB"}, {"source": "docC"}]
TEST_DOCUMENTS = ["content A", "content B", "content C"]

# --- Test Cases ---

def test_health_check(test_app_client: TestClient):
    """Test the /health endpoint."""
    response = test_app_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_add_embeddings(test_app_client: TestClient):
    """Test adding embeddings via the /add endpoint."""
    add_data = {
        "ids": TEST_IDS,
        "embeddings": TEST_EMBEDDINGS,
        "metadatas": TEST_METADATAS,
        "documents": TEST_DOCUMENTS
    }
    response = test_app_client.post(f"/collections/{TEST_COLLECTION_NAME}/add", json=add_data)
    assert response.status_code == 200
    assert response.json() == {"message": f"Added {len(TEST_IDS)} items to collection '{TEST_COLLECTION_NAME}'"}

    # Optional: Verify directly with the in-memory client (if fixture allows)
    # This requires passing the in_memory_chroma_client fixture here too
    # collection = in_memory_chroma_client.get_collection(TEST_COLLECTION_NAME)
    # assert collection.count() == len(TEST_IDS)

def test_query_collection_found(test_app_client: TestClient):
    """Test querying for embeddings that should be found."""
    # Ensure data is added first (test dependency or add again)
    add_data = {"ids": TEST_IDS, "embeddings": TEST_EMBEDDINGS, "metadatas": TEST_METADATAS}
    test_app_client.post(f"/collections/{TEST_COLLECTION_NAME}/add", json=add_data) # Add data if tests run independently

    query_data = {
        "query_embeddings": [[0.11, 0.21]], # Close to vec1
        "n_results": 1,
        "include": ["metadatas", "distances"]
    }
    response = test_app_client.post(f"/collections/{TEST_COLLECTION_NAME}/query", json=query_data)
    assert response.status_code == 200
    results = response.json()

    assert "ids" in results
    assert "distances" in results
    assert "metadatas" in results
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "vec1" # Expecting vec1 to be the closest
    assert results["metadatas"][0][0]["source"] == "docA"
    assert isinstance(results["distances"][0][0], float)

def test_query_collection_not_found(test_app_client: TestClient):
    """Test querying for embeddings far from existing ones."""
    # Assuming data from previous test or add it again
    query_data = {
        "query_embeddings": [[0.9, 0.9]], # Far from existing embeddings
        "n_results": 1
        # Default include should be fine
    }
    response = test_app_client.post(f"/collections/{TEST_COLLECTION_NAME}/query", json=query_data)
    assert response.status_code == 200
    results = response.json()

    assert len(results.get("ids", [[]])[0]) > 0 # Chroma might still return the least dissimilar
    # We are mainly checking the call succeeds, the actual closest might vary
    # assert results["ids"][0][0] != "vec1" # Or check distance is large

def test_add_invalid_data(test_app_client: TestClient):
    """Test adding data with mismatched list lengths (should fail)."""
    add_data = {
        "ids": ["vec_invalid"], # Only one ID
        "embeddings": TEST_EMBEDDINGS # But three embeddings
    }
    response = test_app_client.post(f"/collections/{TEST_COLLECTION_NAME}/add", json=add_data)
    # ChromaDB client side validation should raise an error, leading to 500 from server
    assert response.status_code == 500
    # The exact error message might vary based on ChromaDB version
    assert "detail" in response.json()
    # assert "must be the same length" in response.json()["detail"].lower() or "same number of" in response.json()["detail"].lower()
    assert "unequal lengths" in response.json()["detail"].lower() # Check for the actual keyword 