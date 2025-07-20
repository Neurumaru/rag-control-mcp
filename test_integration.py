#!/usr/bin/env python3
"""Integration test for Agent B."""

import sys
import subprocess
import time
import httpx
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_integration():
    """Test basic integration of Agent B."""
    print("🧪 Starting Agent B Integration Test")
    
    # Start server in background
    print("🚀 Starting test server...")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.mcp_rag_control.api.app:app",
        "--host", "127.0.0.1",
        "--port", "8003"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(3)
    
    base_url = "http://127.0.0.1:8003"
    
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            # Test 1: Health check
            print("\n1️⃣ Testing health check...")
            response = client.get("/health")
            assert response.status_code == 200, f"Health check failed: {response.text}"
            health_data = response.json()
            print(f"   ✅ Health check passed: {health_data['status']}")
            
            # Test 2: Root endpoint
            print("\n2️⃣ Testing root endpoint...")
            response = client.get("/")
            assert response.status_code == 200, f"Root endpoint failed: {response.text}"
            root_data = response.json()
            print(f"   ✅ Root endpoint passed: {root_data['message']}")
            
            # Test 3: Module registration
            print("\n3️⃣ Testing module registration...")
            module_request = {
                "name": "test-vector-store",
                "module_type": "vector_store",
                "description": "Test vector store for integration testing",
                "mcp_server_url": "http://localhost:8080/mcp",
                "config": {
                    "host": "localhost",
                    "port": 5432,
                    "database_name": "test_vectors",
                    "dimensions": 768
                },
                "tags": ["test", "integration"]
            }
            
            response = client.post("/api/v1/modules", json=module_request)
            assert response.status_code == 200, f"Module registration failed: {response.text}"
            module_data = response.json()
            module_id = module_data["module"]["id"]
            print(f"   ✅ Module registered successfully: {module_id}")
            
            # Test 4: Module listing
            print("\n4️⃣ Testing module listing...")
            response = client.get("/api/v1/modules")
            assert response.status_code == 200, f"Module listing failed: {response.text}"
            modules_data = response.json()
            assert modules_data["total"] >= 1, "No modules found"
            print(f"   ✅ Module listing passed: {modules_data['total']} modules found")
            
            # Test 5: Pipeline registration
            print("\n5️⃣ Testing pipeline registration...")
            import uuid
            step1_id = str(uuid.uuid4())
            step2_id = str(uuid.uuid4())
            
            pipeline_request = {
                "name": "test-rag-pipeline",
                "description": "Test RAG pipeline for integration testing",
                "version": "1.0.0",
                "steps": [
                    {
                        "id": step1_id,
                        "name": "Embed Query",
                        "step_type": "processing",
                        "module_id": module_id,
                        "parameters": {"operation": "embed"},
                        "next_steps": [step2_id],
                        "conditions": {},
                        "timeout_seconds": 30.0
                    },
                    {
                        "id": step2_id, 
                        "name": "Retrieve Documents",
                        "step_type": "retrieval",
                        "module_id": module_id,
                        "parameters": {"top_k": 5},
                        "next_steps": [],
                        "conditions": {},
                        "timeout_seconds": 30.0
                    }
                ],
                "tags": ["test", "rag"],
                "variables": {},
                "config": {}
            }
            
            response = client.post("/api/v1/pipelines", json=pipeline_request)
            assert response.status_code == 200, f"Pipeline registration failed: {response.text}"
            pipeline_data = response.json()
            pipeline_id = pipeline_data["pipeline"]["id"]
            print(f"   ✅ Pipeline registered successfully: {pipeline_id}")
            
            # Test 6: Pipeline activation
            print("\n6️⃣ Testing pipeline activation...")
            response = client.post(f"/api/v1/pipelines/{pipeline_id}/activate")
            assert response.status_code == 200, f"Pipeline activation failed: {response.text}"
            print(f"   ✅ Pipeline activated successfully")
            
            # Test 7: Query execution
            print("\n7️⃣ Testing query execution...")
            query_request = {
                "query": "What are the latest financial recommendations?",
                "pipeline_id": pipeline_id,
                "context": {"user_id": "test_user"},
                "parameters": {"max_tokens": 100}
            }
            
            response = client.post("/api/v1/execute", json=query_request)
            assert response.status_code == 200, f"Query execution failed: {response.text}"
            query_data = response.json()
            execution_id = query_data["execution_id"]
            print(f"   ✅ Query executed successfully: {execution_id}")
            print(f"   📝 Response: {query_data['response'][:100]}...")
            
            # Test 8: Execution history
            print("\n8️⃣ Testing execution history...")
            response = client.get("/api/v1/executions")
            assert response.status_code == 200, f"Execution history failed: {response.text}"
            executions_data = response.json()
            assert executions_data["total"] >= 1, "No executions found"
            print(f"   ✅ Execution history passed: {executions_data['total']} executions found")
            
            print("\n🎉 All integration tests passed!")
            print(f"   📊 Modules: {modules_data['total']}")
            print(f"   📊 Pipelines: 1 (active)")
            print(f"   📊 Executions: {executions_data['total']}")
            
            return True
            
    finally:
        # Stop server
        print("\n🛑 Stopping test server...")
        server_process.terminate()
        server_process.wait()


if __name__ == "__main__":
    try:
        success = test_integration()
        print("\n✅ Integration test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)