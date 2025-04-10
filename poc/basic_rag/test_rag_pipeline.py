import os
import pytest
from pytest_mock import MockerFixture
from unittest.mock import MagicMock, patch, PropertyMock

# Import the module to be tested (assuming it's in the same directory or path is set)
from rag_pipeline import load_or_create_index, query_rag, Settings, DATA_DIR, VECTOR_STORE_DIR, COLLECTION_NAME

# Define paths relative to the test file location
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, DATA_DIR)
TEST_VECTOR_STORE_DIR = os.path.join(TEST_DIR, VECTOR_STORE_DIR)
TEST_VECTOR_STORE_PATH = os.path.join(TEST_VECTOR_STORE_DIR, COLLECTION_NAME)

# --- Fixtures ---

@pytest.fixture(autouse=True)
def setup_and_teardown(mocker: MockerFixture):
    """Sets up mocks for external dependencies and cleans up vector store dir."""
    # Mock external dependencies that are not core to the logic being tested
    mocker.patch("rag_pipeline.Ollama", return_value=MagicMock()) # Mock LLM initialization
    mocker.patch("rag_pipeline.HuggingFaceEmbedding", return_value=MagicMock()) # Mock Embedding model initialization
    mocker.patch("rag_pipeline.Settings", llm=MagicMock(), embed_model=MagicMock()) # Mock Settings assignment

    # Ensure data directory exists for tests that might need it
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    # Create a dummy data file if it doesn't exist for loading tests
    dummy_file_path = os.path.join(TEST_DATA_DIR, "dummy_doc.txt")
    if not os.path.exists(dummy_file_path):
        with open(dummy_file_path, "w") as f:
            f.write("This is dummy content.")

    yield # Run the test

    # Clean up: Remove the vector store directory after tests
    # Use shutil.rmtree for robust removal, needs import shutil
    import shutil
    if os.path.exists(TEST_VECTOR_STORE_DIR):
        # print(f"\nCleaning up test vector store: {TEST_VECTOR_STORE_DIR}")
        try:
            shutil.rmtree(TEST_VECTOR_STORE_DIR)
        except OSError as e:
            print(f"Error removing directory {TEST_VECTOR_STORE_DIR}: {e}")
    # Clean up dummy data file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)

@pytest.fixture
def mock_chromadb(mocker: MockerFixture):
    """Mocks chromadb client and collection."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mocker.patch("rag_pipeline.chromadb.PersistentClient", return_value=mock_client)
    return mock_client, mock_collection

@pytest.fixture
def mock_llamaindex_storage(mocker: MockerFixture):
    """Mocks LlamaIndex StorageContext and VectorStoreIndex."""
    mock_vector_store = MagicMock()
    mock_storage_context = MagicMock()
    mock_storage_context.from_defaults.return_value = mock_storage_context
    mock_index = MagicMock()

    mocker.patch("rag_pipeline.ChromaVectorStore", return_value=mock_vector_store)
    mocker.patch("rag_pipeline.StorageContext", new=mock_storage_context)
    mocker.patch("rag_pipeline.VectorStoreIndex", new=mock_index)
    mock_index.from_vector_store.return_value = mock_index
    mock_index.from_documents.return_value = mock_index

    return mock_vector_store, mock_storage_context, mock_index

# --- Test Cases ---

def test_load_or_create_index_creates_new(mock_chromadb, mock_llamaindex_storage, mocker: MockerFixture):
    """Test index creation when vector store doesn't exist."""
    mocker.patch("os.path.exists", return_value=False)
    # Mock the load_data method directly on the class
    mock_load_data = mocker.patch("rag_pipeline.SimpleDirectoryReader.load_data", return_value=[MagicMock(text="doc1")])
    mocker.patch("os.makedirs") # Mock makedirs to avoid actual creation

    mock_vector_store, mock_storage_context, mock_index = mock_llamaindex_storage

    index = load_or_create_index(TEST_DATA_DIR, TEST_VECTOR_STORE_DIR, COLLECTION_NAME)

    assert index is not None
    # Check if load_data was called
    mock_load_data.assert_called_once()
    mock_llamaindex_storage[2].from_documents.assert_called_once()
    mock_chromadb[0].get_or_create_collection.assert_called_with(COLLECTION_NAME)
    mock_llamaindex_storage[1].from_defaults.assert_called_once()

def test_load_or_create_index_loads_existing(mock_chromadb, mock_llamaindex_storage, mocker: MockerFixture):
    """Test index loading when vector store exists."""
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.listdir", return_value=["dummy_file"])
    # Mock the load_data method directly to check it's NOT called
    mock_load_data = mocker.patch("rag_pipeline.SimpleDirectoryReader.load_data")

    mock_vector_store, mock_storage_context, mock_index = mock_llamaindex_storage

    index = load_or_create_index(TEST_DATA_DIR, TEST_VECTOR_STORE_DIR, COLLECTION_NAME)

    assert index is not None
    mock_load_data.assert_not_called() # Ensure load_data was not called
    mock_llamaindex_storage[2].from_documents.assert_not_called()
    mock_llamaindex_storage[2].from_vector_store.assert_called_once()
    mock_chromadb[0].get_or_create_collection.assert_called_with(COLLECTION_NAME)
    mock_llamaindex_storage[1].from_defaults.assert_called_once()

def test_query_rag_success(mocker: MockerFixture, capsys):
    """Test successful query execution and output."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_response = MagicMock()

    # Mock the response object structure LlamaIndex uses
    mock_node = MagicMock()
    type(mock_node).node_id = PropertyMock(return_value="node-123")
    type(mock_node).score = PropertyMock(return_value=0.85)
    mock_response.response = "This is the mocked LLM response."
    mock_response.source_nodes = [mock_node]
    # Mock the __str__ method to return the desired response string for print()
    mock_response.__str__.return_value = "This is the mocked LLM response."

    mock_query_engine.query.return_value = mock_response
    mock_index.as_query_engine.return_value = mock_query_engine

    query = "What is the project about?"
    query_rag(mock_index, query)

    mock_index.as_query_engine.assert_called_once()
    mock_query_engine.query.assert_called_once_with(query)

    captured = capsys.readouterr()
    assert f"Querying index with: '{query}'" in captured.out
    assert "Response:" in captured.out
    assert "This is the mocked LLM response." in captured.out
    assert "--- Sources ---" in captured.out
    assert "Node ID: node-123, Score: 0.8500" in captured.out

def test_query_rag_index_none(capsys):
    """Test query execution when index is None."""
    query = "Test query"
    query_rag(None, query)

    captured = capsys.readouterr()
    assert f"Querying index with: '{query}'" in captured.out
    assert "Error: Index is not available." in captured.out 