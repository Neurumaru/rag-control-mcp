# MCP-RAG-Control 어댑터 설계

## 1. 기본 어댑터 인터페이스 (adapters/base_adapter.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class MCPRequest(BaseModel):
    """MCP 요청 모델"""
    source_id: str
    operation: str
    params: Dict[str, Any]
    request_id: str


class MCPResponse(BaseModel):
    """MCP 응답 모델"""
    source_id: str
    status: str
    data: Any
    request_id: str
    error: Optional[str] = None


class BaseAdapter(ABC):
    """MCP 어댑터 기본 인터페이스"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        """
        어댑터 초기화
        
        Args:
            module_id: 모듈 ID
            config: 모듈 구성 설정
        """
        self.module_id = module_id
        self.config = config
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        데이터 소스에 연결
        
        Returns:
            연결 성공 여부
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        데이터 소스 연결 해제
        
        Returns:
            연결 해제 성공 여부
        """
        pass
    
    @abstractmethod
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        MCP 요청 처리
        
        Args:
            request: MCP 요청 객체
            
        Returns:
            MCP 응답 객체
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        어댑터 상태 확인
        
        Returns:
            상태 정보 딕셔너리
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        어댑터 지원 기능 목록 반환
        
        Returns:
            지원 기능 목록
        """
        pass


class AdapterRegistry:
    """어댑터 등록 관리"""
    
    _adapters: Dict[str, type] = {}
    
    @classmethod
    def register(cls, adapter_type: str) -> callable:
        """
        어댑터 유형 등록 데코레이터
        
        Args:
            adapter_type: 어댑터 유형
            
        Returns:
            데코레이터 함수
        """
        def wrapper(adapter_class: type) -> type:
            cls._adapters[adapter_type] = adapter_class
            return adapter_class
        return wrapper
    
    @classmethod
    def get_adapter_class(cls, adapter_type: str) -> Optional[type]:
        """
        어댑터 클래스 조회
        
        Args:
            adapter_type: 어댑터 유형
            
        Returns:
            어댑터 클래스
        """
        return cls._adapters.get(adapter_type)
    
    @classmethod
    def create_adapter(cls, adapter_type: str, module_id: str, config: Dict[str, Any]) -> Optional[BaseAdapter]:
        """
        어댑터 인스턴스 생성
        
        Args:
            adapter_type: 어댑터 유형
            module_id: 모듈 ID
            config: 모듈 구성 설정
            
        Returns:
            어댑터 인스턴스
        """
        adapter_class = cls.get_adapter_class(adapter_type)
        if adapter_class:
            return adapter_class(module_id, config)
        return None
```

## 2. 벡터 데이터베이스 어댑터 (adapters/vector_adapter.py)

```python
import uuid
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from faiss import IndexFlatL2

from mcp_rag_control.adapters.base_adapter import (AdapterRegistry, BaseAdapter,
                                                  MCPRequest, MCPResponse)


@AdapterRegistry.register("faiss")
class FAISSAdapter(BaseAdapter):
    """FAISS 벡터 데이터베이스 어댑터"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        """
        FAISS 어댑터 초기화
        
        Args:
            module_id: 모듈 ID
            config: 모듈 구성 설정
        """
        super().__init__(module_id, config)
        self.dimension = config.get("dimension", 768)
        self.index: Optional[IndexFlatL2] = None
        self.document_map: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
    
    async def connect(self) -> bool:
        """
        FAISS 인덱스 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            return True
        except Exception as e:
            print(f"FAISS 연결 오류: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        FAISS 인덱스 리소스 해제
        
        Returns:
            해제 성공 여부
        """
        try:
            self.index = None
            self.document_map = {}
            return True
        except Exception as e:
            print(f"FAISS 연결 해제 오류: {str(e)}")
            return False
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        MCP 요청 처리
        
        Args:
            request: MCP 요청 객체
            
        Returns:
            MCP 응답 객체
        """
        operation = request.operation
        params = request.params
        
        try:
            if operation == "add_document":
                result = await self._add_document(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "add_documents":
                result = await self._add_documents(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "query":
                result = await self._query(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "delete_document":
                result = await self._delete_document(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            else:
                return MCPResponse(
                    source_id=request.source_id,
                    status="error",
                    data=None,
                    request_id=request.request_id,
                    error=f"지원하지 않는 작업: {operation}"
                )
        except Exception as e:
            return MCPResponse(
                source_id=request.source_id,
                status="error",
                data=None,
                request_id=request.request_id,
                error=str(e)
            )
    
    async def _add_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 문서 추가
        
        Args:
            params: 요청 매개변수
            
        Returns:
            추가된 문서 ID
        """
        embedding = np.array(params["embedding"], dtype=np.float32).reshape(1, -1)
        document = params.get("document", {})
        
        doc_id = self.next_id
        self.next_id += 1
        
        self.index.add(embedding)
        self.document_map[doc_id] = document
        
        return {"document_id": doc_id}
    
    async def _add_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        다수 문서 추가
        
        Args:
            params: 요청 매개변수
            
        Returns:
            추가된 문서 ID 목록
        """
        embeddings = np.array(params["embeddings"], dtype=np.float32)
        documents = params.get("documents", [])
        
        doc_ids = []
        
        for i, doc in enumerate(documents):
            doc_id = self.next_id
            self.next_id += 1
            doc_ids.append(doc_id)
            self.document_map[doc_id] = doc
        
        self.index.add(embeddings)
        
        return {"document_ids": doc_ids}
    
    async def _query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        벡터 쿼리 실행
        
        Args:
            params: 요청 매개변수
            
        Returns:
            검색 결과
        """
        query_embedding = np.array(params["query_embedding"], dtype=np.float32).reshape(1, -1)
        k = params.get("top_k", 5)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.document_map):
                doc = self.document_map[idx]
                results.append({
                    "document_id": idx,
                    "document": doc,
                    "score": float(distances[0][i])
                })
        
        return {"results": results}
    
    async def _delete_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 삭제 (FAISS는 직접 삭제를 지원하지 않으므로 마킹만 함)
        
        Args:
            params: 요청 매개변수
            
        Returns:
            삭제 결과
        """
        doc_id = params["document_id"]
        
        if doc_id in self.document_map:
            del self.document_map[doc_id]
            return {"success": True}
        else:
            return {"success": False, "message": f"문서 ID {doc_id}를 찾을 수 없음"}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        어댑터 상태 확인
        
        Returns:
            상태 정보
        """
        is_connected = self.index is not None
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "document_count": self.index.ntotal if is_connected else 0,
            "dimension": self.dimension
        }
    
    def get_capabilities(self) -> List[str]:
        """
        어댑터 지원 기능 목록 반환
        
        Returns:
            지원 기능 목록
        """
        return [
            "add_document",
            "add_documents",
            "query",
            "delete_document"
        ]


@AdapterRegistry.register("chroma")
class ChromaAdapter(BaseAdapter):
    """ChromaDB 벡터 데이터베이스 어댑터"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        """
        ChromaDB 어댑터 초기화
        
        Args:
            module_id: 모듈 ID
            config: 모듈 구성 설정
        """
        super().__init__(module_id, config)
        self.collection_name = config.get("collection_name", "default")
        self.chroma_client = None
        self.collection = None
    
    async def connect(self) -> bool:
        """
        ChromaDB 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            import chromadb
            
            # ChromaDB 클라이언트 초기화
            self.chroma_client = chromadb.Client()
            
            # 컬렉션 생성 또는 가져오기
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            return True
        except Exception as e:
            print(f"ChromaDB 연결 오류: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        ChromaDB 연결 해제
        
        Returns:
            연결 해제 성공 여부
        """
        try:
            self.collection = None
            self.chroma_client = None
            return True
        except Exception as e:
            print(f"ChromaDB 연결 해제 오류: {str(e)}")
            return False
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        MCP 요청 처리
        
        Args:
            request: MCP 요청 객체
            
        Returns:
            MCP 응답 객체
        """
        operation = request.operation
        params = request.params
        
        try:
            if operation == "add_document":
                result = await self._add_document(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "add_documents":
                result = await self._add_documents(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "query":
                result = await self._query(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "delete_document":
                result = await self._delete_document(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            else:
                return MCPResponse(
                    source_id=request.source_id,
                    status="error",
                    data=None,
                    request_id=request.request_id,
                    error=f"지원하지 않는 작업: {operation}"
                )
        except Exception as e:
            return MCPResponse(
                source_id=request.source_id,
                status="error",
                data=None,
                request_id=request.request_id,
                error=str(e)
            )
    
    async def _add_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 문서 추가
        
        Args:
            params: 요청 매개변수
            
        Returns:
            추가된 문서 ID
        """
        doc_id = params.get("document_id", str(uuid.uuid4()))
        document = params.get("document", {})
        embedding = params.get("embedding", [])
        metadata = params.get("metadata", {})
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document.get("text", "")]
        )
        
        return {"document_id": doc_id}
    
    async def _add_documents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        다수 문서 추가
        
        Args:
            params: 요청 매개변수
            
        Returns:
            추가된 문서 ID 목록
        """
        documents = params.get("documents", [])
        embeddings = params.get("embeddings", [])
        metadatas = params.get("metadatas", [{}] * len(documents))
        
        # ID 생성 또는 사용
        doc_ids = params.get("document_ids", [str(uuid.uuid4()) for _ in range(len(documents))])
        
        # 텍스트 추출
        texts = [doc.get("text", "") for doc in documents]
        
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
        return {"document_ids": doc_ids}
    
    async def _query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        벡터 쿼리 실행
        
        Args:
            params: 요청 매개변수
            
        Returns:
            검색 결과
        """
        query_embedding = params.get("query_embedding", [])
        k = params.get("top_k", 5)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # 결과 포맷팅
        formatted_results = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            formatted_results.append({
                "document_id": doc_id,
                "document": {
                    "text": results.get("documents", [[]])[0][i]
                },
                "metadata": results.get("metadatas", [[]])[0][i],
                "score": results.get("distances", [[]])[0][i]
            })
        
        return {"results": formatted_results}
    
    async def _delete_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 삭제
        
        Args:
            params: 요청 매개변수
            
        Returns:
            삭제 결과
        """
        doc_id = params.get("document_id")
        
        try:
            self.collection.delete(ids=[doc_id])
            return {"success": True}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        어댑터 상태 확인
        
        Returns:
            상태 정보
        """
        is_connected = self.collection is not None
        
        count = 0
        if is_connected:
            try:
                count = self.collection.count()
            except:
                pass
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "collection": self.collection_name,
            "document_count": count
        }
    
    def get_capabilities(self) -> List[str]:
        """
        어댑터 지원 기능 목록 반환
        
        Returns:
            지원 기능 목록
        """
        return [
            "add_document",
            "add_documents",
            "query",
            "delete_document"
        ]
```

## 3. 데이터베이스 어댑터 (adapters/database_adapter.py)

```python
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mcp_rag_control.adapters.base_adapter import (AdapterRegistry, BaseAdapter,
                                                  MCPRequest, MCPResponse)


@AdapterRegistry.register("sql")
class SQLAdapter(BaseAdapter):
    """SQL 데이터베이스 어댑터"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        """
        SQL 어댑터 초기화
        
        Args:
            module_id: 모듈 ID
            config: 모듈 구성 설정
        """
        super().__init__(module_id, config)
        self.connection_string = config.get("connection_string")
        self.engine: Optional[AsyncEngine] = None
    
    async def connect(self) -> bool:
        """
        데이터베이스 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            self.engine = create_async_engine(
                self.connection_string, 
                echo=False,
                future=True
            )
            return True
        except Exception as e:
            print(f"SQL 연결 오류: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        데이터베이스 연결 해제
        
        Returns:
            연결 해제 성공 여부
        """
        try:
            if self.engine:
                await self.engine.dispose()
                self.engine = None
            return True
        except Exception as e:
            print(f"SQL 연결 해제 오류: {str(e)}")
            return False
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        MCP 요청 처리
        
        Args:
            request: MCP 요청 객체
            
        Returns:
            MCP 응답 객체
        """
        operation = request.operation
        params = request.params
        
        try:
            if operation == "execute_query":
                result = await self._execute_query(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            elif operation == "execute_statement":
                result = await self._execute_statement(params)
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result,
                    request_id=request.request_id
                )
            else:
                return MCPResponse(
                    source_id=request.source_id,
                    status="error",
                    data=None,
                    request_id=request.request_id,
                    error=f"지원하지 않는 작업: {operation}"
                )
        except Exception as e:
            return MCPResponse(
                source_id=request.source_id,
                status="error",
                data=None,
                request_id=request.request_id,
                error=str(e)
            )
    
    async def _execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        읽기 쿼리 실행
        
        Args:
            params: 요청 매개변수
            
        Returns:
            쿼리 결과
        """
        query = params.get("query")
        query_params = params.get("params", {})
        
        if not self.engine:
            raise ValueError("데이터베이스 연결이 설정되지 않았습니다.")
        
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), query_params)
            
            # 결과 변환
            rows = result.fetchall()
            column_names = result.keys()
            
            formatted_results = [
                {column: value for column, value in zip(column_names, row)}
                for row in rows
            ]
            
            return {"results": formatted_results, "row_count": len(formatted_results)}
    
    async def _execute_statement(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        쓰기 SQL 명령문 실행
        
        Args:
            params: 요청 매개변수
            
        Returns:
            실행 결과
        """
        statement = params.get("statement")
        statement_params = params.get("params", {})
        
        if not self.engine:
            raise ValueError("데이터베이스 연결이 설정되지 않았습니다.")
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text(statement), statement_params)
            
            return {
                "row_count": result.rowcount,
                "last_inserted_id": result.lastrowid
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        어댑터 상태 확인
        
        Returns:
            상태 정보
        """
        is_connected = self.engine is not None
        
        if is_connected:
            try:
                async with self.engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                    status = "healthy"
            except Exception:
                status = "unhealthy"
        else:
            status = "disconnected"
        
        return {
            "status": status,
            "connection_info": self.connection_string.split('@')[-1].split('/')[0] if self.connection_string else None
        }
    
    def get_capabilities(self) -> List[str]:
        """
        어댑터 지원 기능 목록 반환
        
        Returns:
            지원 기능 목록
        """
        return [
            "execute_query",
            "execute_statement"
        ]
```

## 4. MCP 클라이언트 (adapters/mcp_client.py)

```python
import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

class MCPClientConfig(BaseModel):
    """MCP 클라이언트 설정"""
    server_type: str = Field(..., description="서버 유형 (stdio 또는 http)")
    server_path: str = Field(..., description="서버 경로 또는 URL")
    server_args: List[str] = Field(default_factory=list, description="서버 명령 인자")
    timeout: int = Field(default=30, description="요청 타임아웃 (초)")


class MCPClient:
    """MCP 클라이언트"""
    
    def __init__(self, config: MCPClientConfig):
        """
        MCP 클라이언트 초기화
        
        Args:
            config: 클라이언트 설정
        """
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict[str, Any]] = []
        self.connected = False
    
    async def connect(self) -> bool:
        """
        MCP 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            if self.config.server_type == "stdio":
                server_params = StdioServerParameters(
                    command="python" if ".py" in self.config.server_path else "node",
                    args=[self.config.server_path] + self.config.server_args
                )
                
                async with stdio_client(server_params) as (read, write):
                    self.session = ClientSession(read, write)
                    await self.session.__aenter__()
                
                # 초기화
                await self.session.initialize()
                
                # 도구 목록 조회
                tools_result = await self.session.list_tools()
                self.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    }
                    for tool in tools_result.tools
                ]
                
                self.connected = True
                return True
            
            elif self.config.server_type == "http":
                # HTTP 기반 연결은 별도 구현 필요
                # Streamable HTTP 클라이언트는 다른 구현 방식이 필요
                raise NotImplementedError("HTTP 기반 연결은 아직 구현되지 않았습니다.")
            
            else:
                raise ValueError(f"지원하지 않는 서버 유형: {self.config.server_type}")
        
        except Exception as e:
            print(f"MCP 서버 연결 오류: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """
        MCP 서버 연결 해제
        
        Returns:
            연결 해제 성공 여부
        """
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
                self.connected = False
            return True
        except Exception as e:
            print(f"MCP 서버 연결 해제 오류: {str(e)}")
            return False
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP 도구 실행
        
        Args:
            tool_name: 도구 이름
            params: 도구 매개변수
            
        Returns:
            도구 실행 결과
        """
        if not self.connected or not self.session:
            raise ValueError("MCP 서버에 연결되어 있지 않습니다.")
        
        # 도구 찾기
        tool = next((t for t in self.tools if t["name"] == tool_name), None)
        if not tool:
            raise ValueError(f"도구를 찾을 수 없음: {tool_name}")
        
        # 도구 실행
        try:
            result = await self.session.execute_tool(tool_name, params)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """
        사용 가능한 도구 목록 반환
        
        Returns:
            도구 이름 목록
        """
        return [tool["name"] for tool in self.tools]


class MCPClientManager:
    """MCP 클라이언트 관리자"""
    
    _clients: Dict[str, MCPClient] = {}
    
    @classmethod
    async def get_client(cls, client_id: str, config: Optional[MCPClientConfig] = None) -> Optional[MCPClient]:
        """
        클라이언트 인스턴스 가져오기 또는 생성
        
        Args:
            client_id: 클라이언트 ID
            config: 클라이언트 설정 (생성시 필요)
            
        Returns:
            MCP 클라이언트 인스턴스
        """
        # 기존 클라이언트 반환
        if client_id in cls._clients:
            return cls._clients[client_id]
        
        # 설정이 없으면 생성 불가
        if not config:
            return None
        
        # 새 클라이언트 생성
        client = MCPClient(config)
        connected = await client.connect()
        
        if connected:
            cls._clients[client_id] = client
            return client
        
        return None
    
    @classmethod
    async def close_client(cls, client_id: str) -> bool:
        """
        클라이언트 연결 종료 및 제거
        
        Args:
            client_id: 클라이언트 ID
            
        Returns:
            성공 여부
        """
        if client_id in cls._clients:
            client = cls._clients[client_id]
            success = await client.disconnect()
            
            if success:
                del cls._clients[client_id]
            
            return success
        
        return False
    
    @classmethod
    async def close_all(cls) -> None:
        """모든 클라이언트 연결 종료"""
        for client_id in list(cls._clients.keys()):
            await cls.close_client(client_id)
```

## 5. MCP 어댑터 (adapters/mcp_adapter.py)

```python
import uuid
from typing import Any, Dict, List, Optional

from mcp_rag_control.adapters.base_adapter import (AdapterRegistry, BaseAdapter,
                                                  MCPRequest, MCPResponse)
from mcp_rag_control.adapters.mcp_client import MCPClientConfig, MCPClientManager


@AdapterRegistry.register("mcp")
class MCPAdapter(BaseAdapter):
    """MCP 호환 어댑터"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        """
        MCP 어댑터 초기화
        
        Args:
            module_id: 모듈 ID
            config: 모듈 구성 설정
        """
        super().__init__(module_id, config)
        
        # MCP 클라이언트 설정 추출
        self.client_id = config.get("client_id", module_id)
        self.server_type = config.get("server_type", "stdio")
        self.server_path = config.get("server_path", "")
        self.server_args = config.get("server_args", [])
        self.timeout = config.get("timeout", 30)
        
        # 도구 매핑 설정
        self.tool_mapping = config.get("tool_mapping", {})
        
        # 클라이언트 인스턴스
        self.client = None
    
    async def connect(self) -> bool:
        """
        MCP 서버에 연결
        
        Returns:
            연결 성공 여부
        """
        try:
            # 클라이언트 설정 생성
            client_config = MCPClientConfig(
                server_type=self.server_type,
                server_path=self.server_path,
                server_args=self.server_args,
                timeout=self.timeout
            )
            
            # 클라이언트 가져오기 또는 생성
            self.client = await MCPClientManager.get_client(self.client_id, client_config)
            
            return self.client is not None
        except Exception as e:
            print(f"MCP 서버 연결 오류: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """
        MCP 서버 연결 해제
        
        Returns:
            연결 해제 성공 여부
        """
        if self.client:
            return await MCPClientManager.close_client(self.client_id)
        return True
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """
        MCP 요청 처리
        
        Args:
            request: MCP 요청 객체
            
        Returns:
            MCP 응답 객체
        """
        if not self.client:
            return MCPResponse(
                source_id=request.source_id,
                status="error",
                data=None,
                request_id=request.request_id,
                error="MCP 서버에 연결되어 있지 않습니다."
            )
        
        # 작업 매핑
        operation = request.operation
        params = request.params
        
        # 도구 이름 매핑 확인
        tool_name = self.tool_mapping.get(operation, operation)
        
        try:
            # 도구 실행
            result = await self.client.execute_tool(tool_name, params)
            
            if result["status"] == "success":
                return MCPResponse(
                    source_id=request.source_id,
                    status="success",
                    data=result["data"],
                    request_id=request.request_id
                )
            else:
                return MCPResponse(
                    source_id=request.source_id,
                    status="error",
                    data=None,
                    request_id=request.request_id,
                    error=result.get("error", "알 수 없는 오류")
                )
        except Exception as e:
            return MCPResponse(
                source_id=request.source_id,
                status="error",
                data=None,
                request_id=request.request_id,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        어댑터 상태 확인
        
        Returns:
            상태 정보
        """
        if not self.client:
            return {
                "status": "unhealthy", 
                "message": "MCP 서버에 연결되어 있지 않습니다."
            }
        
        # 클라이언트 상태 확인
        capabilities = self.client.get_capabilities()
        
        return {
            "status": "healthy" if self.client.connected else "unhealthy",
            "capabilities": capabilities,
            "server_type": self.server_type,
            "server_path": self.server_path
        }
    
    def get_capabilities(self) -> List[str]:
        """
        어댑터 지원 기능 목록 반환
        
        Returns:
            지원 기능 목록
        """
        if self.client:
            # 클라이언트 capabilities 반환
            return self.client.get_capabilities()
        
        # 클라이언트가 없는 경우 빈 목록 반환
        return []
```

## 6. LangChain MCP 통합 예제 (examples/langchain_mcp_example.py)

```python
import asyncio
import os
from typing import List, Dict, Any

# Note: langchain-mcp 라이브러리는 개발 중이므로 MCP Python SDK를 직접 사용
# from langchain_mcp_adapters.client import MultiServerMCPClient  # 미래 구현 예정
# from langchain_mcp_adapters.tools import load_mcp_tools  # 미래 구현 예정
from langgraph.prebuilt import create_react_agent

from mcp_rag_control.adapters.mcp_adapter import MCPAdapter
from mcp_rag_control.adapters.base_adapter import MCPRequest, MCPResponse


async def run_langchain_mcp_example():
    """LangChain MCP 어댑터 예제 실행"""
    
    # 1. MCP 어댑터 설정
    adapter_config = {
        "client_id": "math_server",
        "server_type": "stdio",
        "server_path": "./examples/math_server.py",
        "server_args": [],
        "timeout": 30,
        "tool_mapping": {
            "add": "add",
            "multiply": "multiply"
        }
    }
    
    # 2. MCP 어댑터 인스턴스 생성
    adapter = MCPAdapter("math_module", adapter_config)
    connected = await adapter.connect()
    
    if not connected:
        print("MCP 서버 연결 실패")
        return
    
    try:
        # 3. MCP 요청 테스트
        add_request = MCPRequest(
            source_id="math_module",
            operation="add",
            params={"a": 3, "b": 5},
            request_id="req-1"
        )
        
        add_response = await adapter.process_request(add_request)
        print(f"덧셈 결과: {add_response.data}")
        
        # 4. 추가 MCP 요청 테스트
        multiply_request = MCPRequest(
            source_id="math_module",
            operation="multiply",
            params={"a": add_response.data, "b": 12},
            request_id="req-2"
        )
        
        multiply_response = await adapter.process_request(multiply_request)
        print(f"곱셈 결과: {multiply_response.data}")
        
        # Note: LangChain MCP 통합은 향후 구현 예정
        # OpenAI API 키 설정
        # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
        
        # LangChain과 MCP 통합 예제 (개념적)
        print("LangChain MCP 통합은 MCP Python SDK 안정화 후 구현 예정")
    
    finally:
        # 연결 해제
        await adapter.disconnect()


# 간단한 MCP 서버 구현 (examples/math_server.py로 저장)
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
async def add(a: float, b: float) -> float:
    '''두 숫자를 더합니다.'''
    return a + b

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    '''두 숫자를 곱합니다.'''
    return a * b

if __name__ == "__main__":
    mcp.run()
"""


if __name__ == "__main__":
    asyncio.run(run_langchain_mcp_example())
```

## 7. 어댑터 사용 방법

### 7.1 MCP 서버 생성

```python
# examples/math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
async def add(a: float, b: float) -> float:
    '''두 숫자를 더합니다.'''
    return a + b

@mcp.tool()
async def multiply(a: float, b: float) -> float:
    '''두 숫자를 곱합니다.'''
    return a * b

if __name__ == "__main__":
    mcp.run()
```

### 7.2 MCP 모듈 등록

```python
# MCP 어댑터 설정
adapter_config = {
    "client_id": "unique_client_id",
    "server_type": "stdio",  # stdio 또는 http
    "server_path": "/path/to/server/script.py",
    "server_args": [],
    "timeout": 30,
    "tool_mapping": {
        "operation_name": "mcp_tool_name"
    }
}

# 모듈 등록 요청
module_request = {
    "module_id": "mcp_module",
    "name": "MCP 모듈",
    "type": "mcp",
    "description": "MCP 서버 연동 모듈",
    "config": adapter_config
}
```

### 7.3 파이프라인에서 사용

```python
# 파이프라인 설정
pipeline_config = {
    "pipeline_id": "mcp_pipeline",
    "name": "MCP 파이프라인",
    "description": "MCP 서버를 활용한 파이프라인",
    "modules": [
        {
            "step": 1,
            "module_id": "mcp_module",
            "config": {
                "operation": "add",
                "params": {"a": "${a}", "b": "${b}"},
                "output_field": "result"
            }
        }
    ]
}
```

### 7.4 의존성 추가

```toml
[project.dependencies]
# 기존 의존성...
mcp-python = ">=0.1.0"
langchain-mcp-adapters = ">=0.0.1"
```

이 설계는 MCP 서버와의 연동을 표준화하고, LangChain과 같은 외부 라이브러리와의 통합을 용이하게 합니다.