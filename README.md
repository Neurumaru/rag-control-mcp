# MCP-RAG-Control

## 🚀 Project Overview

MCP-RAG-Control is a next-generation RAG (Retrieval-Augmented Generation) control system using **data flow-based architecture**. It integrates **LangGraph** and **MCP (Model Context Protocol)** to build scalable and modular RAG pipelines.

### ✨ Key Features
- 🔄 **Data Flow-based ModuleType**: 37 specialized module types
- 🌐 **MCP Standard Integration**: Connect all external systems through standardized interfaces
- 🎯 **LangGraph Compatible**: Complex workflow orchestration
- 🧪 **Full Test Coverage**: 35 tests passing with 45% coverage
- 🏗️ **Modular Architecture**: Independent and reusable components

## 📊 Development Status (January 2025)

### ✅ Completed Components (Tier 1)
- **Agent A (Project Infrastructure)**: 60% Complete
  - ✅ pyproject.toml and project structure
  - ⏳ CI/CD, Docker setup (planned)
- **Agent B (Data Models)**: 100% Complete
  - ✅ 37 data flow-based ModuleTypes
  - ✅ Pydantic V2 models and validation
  - ✅ LangGraph compatible schemas
- **Agent C (Core Utilities)**: 100% Complete
  - ✅ LangGraph Config & Factory
  - ✅ Structured logging system
  - ✅ Configuration management and validation

### 🔄 Currently in Development (Tier 2)
- **Agent D**: MCP Adapter System (Vector DB integration)
- **Agent E**: Registry Storage System (Module/Pipeline management)
- **Agent F**: Test Framework (Integration testing)

## 🏗️ System Architecture

### Core Components

#### 📡 **MCP Adapter Layer**
- **Vector Databases**: FAISS, Pinecone, Weaviate, Chroma
- **Standard MCP Interface**: Connect all external systems through unified interface
- **Auto Health Checks**: Connection monitoring and recovery

#### 🎛️ **LangGraph Controller** (Planned)
- **Workflow Orchestration**: Execute complex RAG pipelines
- **State Management**: Checkpoint-based stable execution
- **Error Handling**: Automatic retry and recovery logic

#### 🗄️ **Registry Storage**
- **Module Registry**: Manage modules based on 37 ModuleTypes
- **Pipeline Registry**: User-defined RAG pipelines
- **Dependency Management**: Automatic module dependency validation

#### 🌐 **FastAPI Backend** (Planned)
- **RESTful API**: Module/Pipeline CRUD operations
- **Execution Engine**: Pipeline execution and monitoring
- **Auto Documentation**: OpenAPI/Swagger support

#### 🖥️ **Streamlit Web Interface** (Planned)
- **Dashboard**: Real-time system status monitoring
- **Pipeline Builder**: Drag & drop pipeline configuration
- **RAG Testing**: Interactive Q&A testing

### 🔄 Data Flow Pattern

```
User Query → Text Processing → Embedding → Vector Search → Document Search → Context Building → LLM Generation → Response
   TEXT    →     TEXT       → EMBEDDINGS →   VECTORS   →  DOCUMENTS  →   CONTEXT    →  RESPONSE
```

#### Key ModuleType Examples:
- **TEXT_PREPROCESSOR**: Text cleaning and chunking
- **EMBEDDING_ENCODER**: Convert text to vectors
- **VECTOR_STORE**: Vector database integration
- **SIMILARITY_SEARCH**: Semantic similarity search
- **CONTEXT_BUILDER**: RAG context construction
- **LLM_GENERATOR**: Language model-based generation

## 🚀 Quick Start

### Installation and Setup

```bash
# Clone the project
git clone https://github.com/your-repo/mcp-rag-control.git
cd mcp-rag-control

# Install dependencies using UV
uv sync

# Install in development mode
uv pip install -e .

# Run tests
uv run pytest
```

### Currently Available Features

#### 1. Module System
```python
from mcp_rag_control.models import Module, ModuleType, ModuleConfig

# Create a vector store module
module = Module(
    name="my_vector_store",
    module_type=ModuleType.VECTOR_STORE,
    mcp_server_url="https://my-vector-db.com/mcp",
    config=ModuleConfig(dimension=512, metric="cosine")
)
```

#### 2. MCP Adapter Usage
```python
from mcp_rag_control.adapters import VectorAdapter

# Search through vector adapter
result = await adapter.execute_operation("search", {
    "query_vector": [0.1, 0.2, ...],
    "top_k": 10,
    "threshold": 0.7
})
```

#### 3. LangGraph Integration
```python
from mcp_rag_control.utils import LangGraphConfig, create_langgraph_logger

# LangGraph configuration
config = LangGraphConfig(
    checkpointer_type="memory",
    recursion_limit=25,
    enable_stream=True
)

# LangGraph-specific logger
logger = create_langgraph_logger("my-thread-id")
```

## Key Technical Terms

### RAG (Retrieval-Augmented Generation)
- A hybrid paradigm combining traditional information retrieval with generative language models
- Enhances LLM responses by retrieving relevant information from external knowledge bases
- Addresses issues like outdated information, hallucinations, and lack of domain-specific knowledge
- Operates in three stages: Retrieval, Augmentation, and Generation
- **Detailed Example Scenario:**
    1. **User Question:** A user asks for "Latest financial product recommendations".
    2. **Query Processing:** The system converts the question text into an embedding vector (e.g., 512-dimensional real vector).
    3. **Information Retrieval:**
        * Queries a connected financial product database (e.g., SQL database with vector search capabilities).
        * **SQL Example (similarity and recency sorting):**
            ```sql
            SELECT product_id, name, release_date, description
            FROM financial_products
            WHERE release_date > '2024-01-01' -- Example: products after a specific date
            ORDER BY vector_distance_cosine(embedding, query_embedding) DESC -- Sorted by similarity to query vector
            LIMIT 5;
            ```
        * **Vector Database via MCP**: Queries connected vector database (e.g., FAISS, Pinecone) through MCP interface for semantic similarity search
        * Retrieves a set of records containing the most relevant and recent product information (product name, release date, description, etc.).
    4. **Information Augmentation:** Organizes the retrieved product information into a structured format (e.g., JSON, Markdown) to create context for the LLM prompt.
        ```markdown
        [Context]
        1. Product Name: Smart Deposit Alpha, Release Date: 2024-03-15, Features: AI-based automatic interest rate adjustment
        2. Product Name: Global Bond Fund Plus, Release Date: 2024-02-28, Features: Diversified investment in developed/emerging market bonds
        ...
        ```
    5. **Answer Generation:** The context and original question are sent to an LLM (e.g., GPT-4). The LLM generates an accurate and detailed answer based on the provided up-to-date information.

### MCP (Model Context Protocol)
- A standardized protocol connecting LLM applications with various external data sources
- Manages and transmits contextual information used by generation models in RAG systems
- Enables dynamic and bidirectional context exchange
- Provides interoperability between diverse data sources
- **Detailed Example Scenario:**
    1. **Complex Question:** A user asks about a specific financial product (e.g., ID 123): "What are the recent news articles related to this product's historical yield trends?"
    2. **Parallel Search Request:** The controller analyzes this question and determines that two types of information are needed.
        * Yield data: Query a time-series database (e.g., InfluxDB)
        * Related news: Query a vector database (e.g., FAISS)
    3. **MCP-based Communication:**
        * The controller uses the MCP standard request format to asynchronously query each data source (InfluxDB MCP, FAISS MCP).
        * **Standard Request Format (Example):**
            ```json
            {
              "source_id": "influxdb_mcp_1",
              "operation": "query_timeseries",
              "params": {"product_id": 123, "metric": "yield", "time_range": "1y"},
              "request_id": "req-abc-1"
            }
            ```
            ```json
            {
              "source_id": "faiss_mcp_2",
              "operation": "vector_similarity_search",
              "params": {"query_embedding": [0.1, 0.5, ...], "product_id": 123, "top_k": 3},
              "request_id": "req-abc-2"
            }
            ```
    4. **Standard Response Reception:** Each MCP returns results in a standard response format to the controller upon completion of processing.
        * **Standard Response Format (Example):**
            ```json
            {
              "source_id": "influxdb_mcp_1",
              "status": "success",
              "data": {"timestamps": [...], "values": [...]},
              "request_id": "req-abc-1"
            }
            ```
            ```json
            {
              "source_id": "faiss_mcp_2",
              "status": "success",
              "data": [{"news_id": 789, "title": "...", "similarity": 0.85}, ...],
              "request_id": "req-abc-2"
            }
            ```
    5. **Context Integration and Generation:** The controller examines the two types of data received through MCP (yield time series, news article list), organizes them into an integrated context, and passes it to the LLM. The LLM then generates a comprehensive answer based on this information.

## Key Technical Components

### Vector Database Integration via MCP
- **Unified Interface**: All vector databases (FAISS, Pinecone, Weaviate, Chroma, etc.) integrate through standardized MCP protocol
- **Scalable Architecture**: Any vector database can be connected as an MCP server without code changes to the core system
- **Standard Operations**: Search, add, delete, update, and validate operations through consistent MCP interface
- **Performance Optimization**: Efficient similarity search and clustering for high-density vectors
- **Enterprise Ready**: Supports large-scale datasets with distributed vector storage

### LangGraph
- A framework for managing complex workflows in agent RAG systems
- Acts as a central coordinator determining the control flow of RAG systems
- Supports feedback loops and agent behaviors
- Connects retrieval components, memory systems, and language generation modules

## 📋 Development Roadmap

### Next Steps (Tier 2-5)

#### Tier 2 (Core Components)
- **Agent D**: Complete MCP Adapter System
- **Agent E**: Implement Registry Storage System
- **Agent F**: Comprehensive Test Framework

#### Tier 3 (Integration System)
- **Agent G**: LangGraph-based Central Controller

#### Tier 4 (Interfaces)
- **Agent H**: FastAPI Backend Implementation
- **Agent I**: Streamlit Web Interface

#### Tier 5 (Completion)
- **Agent J**: Examples and Demo Implementation
- **Agent K**: Deployment and Operations System

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### 📝 License

This project is licensed under the MIT License.

### 🔗 Related Links

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph/)
- [MCP Standard](https://modelcontextprotocol.io/)
- [Project Documentation](/docs/index.md)
- [Development Guide](/TODOs.md)