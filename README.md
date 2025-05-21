# MCP-RAG-Control Project Description

## Project Overview
MCP-RAG-Control is a map-based control architecture for agent-based RAG (Retrieval-Augmented Generation) systems. This system is designed to efficiently manage and control complex information retrieval and generation processes.

## Architecture Components

The MCP-RAG-Control architecture consists of the following main components:

### API Interface
- Provides RESTful API for interactions with users or external systems
- Offers endpoints for module registration, pipeline configuration, and query execution

### Controller
- Acts as the central control unit coordinating all requests and flows
- Based on LangGraph for managing and executing complex workflows
- Controls communication between modules and data flow

### Module Registry
- Manages various modules such as data sources, vector stores, embedding models
- Stores metadata and configuration information for MCP-compatible modules

### Pipeline Registry
- Stores configuration information for user-defined RAG pipelines
- Defines module connection methods, data flow, and execution sequence

### MCP Adapters
- Standardized communication interfaces with various external systems
- Supports customized MCP implementations for different data sources

## Component Interaction Flow

### Module Registration

1. **User Request:** The user registers a module via the API (by calling the `/modules` endpoint).
2. **Controller Processing:** The controller receives and processes the module registration request.
3. **Module Registration:** The module registration request is stored in the module registry, and the list of registered modules is returned.

### Pipeline Registration

1. **User Request:** The user registers a pipeline via the API (by calling the `/pipelines` endpoint).
2. **Controller Processing:** The controller receives and processes the pipeline registration request.
3. **Pipeline Registration:** The pipeline registration request is stored in the pipeline registry, and the list of registered pipelines is returned.

### Pipeline Execution

1. **User Request:** The user submits a question via the API (by calling the `/execute` endpoint).
2. **Query Processing:** The controller analyzes the user's question and identifies the appropriate pipeline.
3. **Pipeline Execution:** The controller finds and sequentially executes the necessary modules for pipeline execution.
4. **Response Return:** The final generated response is delivered to the user through the API interface.

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

### FAISS (Facebook AI Similarity Search)
- An open-source library for efficient similarity search and clustering of high-density vectors
- Used as a vector store in RAG to store vector embeddings of documents or text fragments
- Designed to handle large-scale datasets

### LangGraph
- A framework for managing complex workflows in agent RAG systems
- Acts as a central coordinator determining the control flow of RAG systems
- Supports feedback loops and agent behaviors
- Connects retrieval components, memory systems, and language generation modules