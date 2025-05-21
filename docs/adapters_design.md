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