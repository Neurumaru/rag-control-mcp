# MCP-RAG-Control 모델 설계

## 1. 모듈 모델 (models/module.py)

```python
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModuleType(str, Enum):
    """모듈 유형 정의"""
    DATA_SOURCE = "data_source"  # 데이터 소스 (예: 웹사이트, 데이터베이스, API)
    VECTOR_STORE = "vector_store"  # 벡터 저장소 (예: FAISS, ChromaDB)
    EMBEDDING_MODEL = "embedding_model"  # 임베딩 모델 (예: Sentence-Transformers)
    LLM = "llm"  # 언어 모델 (예: OpenAI, HuggingFace)
    RETRIEVER = "retriever"  # 검색기 (예: 벡터 검색, 키워드 검색)
    RERANKER = "reranker"  # 재순위화 모듈 (예: 관련성 점수 재계산)
    CUSTOM = "custom"  # 사용자 정의 모듈


class ModuleBase(BaseModel):
    """모듈 기본 모델"""
    name: str = Field(..., description="모듈 이름")
    type: ModuleType = Field(..., description="모듈 유형")
    description: Optional[str] = Field(None, description="모듈 설명")


class ModuleCreate(ModuleBase):
    """모듈 생성 모델"""
    module_id: Optional[str] = Field(None, description="모듈 ID (미지정시 자동 생성)")
    config: Dict[str, Any] = Field(default_factory=dict, description="모듈 구성 설정")


class ModuleUpdate(BaseModel):
    """모듈 업데이트 모델"""
    name: Optional[str] = Field(None, description="모듈 이름")
    description: Optional[str] = Field(None, description="모듈 설명")
    config: Optional[Dict[str, Any]] = Field(None, description="모듈 구성 설정")


class Module(ModuleBase):
    """모듈 응답 모델"""
    module_id: str = Field(..., description="모듈 ID")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="업데이트 시간")
    config: Dict[str, Any] = Field(default_factory=dict, description="모듈 구성 설정")

    class Config:
        orm_mode = True


class ModuleInfo(BaseModel):
    """모듈 기본 정보 모델 (목록 조회용)"""
    module_id: str = Field(..., description="모듈 ID")
    name: str = Field(..., description="모듈 이름")
    type: ModuleType = Field(..., description="모듈 유형")
    description: Optional[str] = Field(None, description="모듈 설명")

    class Config:
        orm_mode = True
```

## 2. 파이프라인 모델 (models/pipeline.py)

```python
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineModuleConfig(BaseModel):
    """파이프라인 내 모듈 설정"""
    step: int = Field(..., description="실행 단계 (1부터 시작)")
    module_id: str = Field(..., description="모듈 ID")
    config: Dict[str, Any] = Field(default_factory=dict, description="모듈 실행 설정")


class PipelineBase(BaseModel):
    """파이프라인 기본 모델"""
    name: str = Field(..., description="파이프라인 이름")
    description: Optional[str] = Field(None, description="파이프라인 설명")


class PipelineCreate(PipelineBase):
    """파이프라인 생성 모델"""
    pipeline_id: Optional[str] = Field(None, description="파이프라인 ID (미지정시 자동 생성)")
    modules: List[PipelineModuleConfig] = Field(..., description="파이프라인 모듈 구성")


class PipelineUpdate(BaseModel):
    """파이프라인 업데이트 모델"""
    name: Optional[str] = Field(None, description="파이프라인 이름")
    description: Optional[str] = Field(None, description="파이프라인 설명")
    modules: Optional[List[PipelineModuleConfig]] = Field(None, description="파이프라인 모듈 구성")


class PipelineModuleDetail(PipelineModuleConfig):
    """파이프라인 내 모듈 상세 정보"""
    module_name: str = Field(..., description="모듈 이름")
    module_type: str = Field(..., description="모듈 유형")


class Pipeline(PipelineBase):
    """파이프라인 응답 모델"""
    pipeline_id: str = Field(..., description="파이프라인 ID")
    created_at: datetime = Field(..., description="생성 시간")
    updated_at: datetime = Field(..., description="업데이트 시간")
    modules: List[PipelineModuleDetail] = Field(..., description="파이프라인 모듈 구성")

    class Config:
        orm_mode = True


class PipelineInfo(BaseModel):
    """파이프라인 기본 정보 모델 (목록 조회용)"""
    pipeline_id: str = Field(..., description="파이프라인 ID")
    name: str = Field(..., description="파이프라인 이름")
    description: Optional[str] = Field(None, description="파이프라인 설명")

    class Config:
        orm_mode = True
```

## 3. API 요청/응답 모델 (models/request.py)

```python
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """검색된 소스 정보"""
    source_id: str = Field(..., description="소스 ID")
    content: str = Field(..., description="소스 내용")
    relevance_score: float = Field(..., description="관련성 점수")
    metadata: Optional[Dict[str, Any]] = Field(None, description="소스 메타데이터")


class TokenUsage(BaseModel):
    """토큰 사용량 정보"""
    prompt_tokens: int = Field(..., description="프롬프트 토큰 수")
    completion_tokens: int = Field(..., description="생성 토큰 수")
    total_tokens: int = Field(..., description="총 토큰 수")


class ExecuteRequest(BaseModel):
    """파이프라인 실행 요청"""
    pipeline_id: str = Field(..., description="파이프라인 ID")
    query: str = Field(..., description="사용자 질문")
    parameters: Optional[Dict[str, Any]] = Field(None, description="실행 매개변수")


class AsyncExecuteRequest(ExecuteRequest):
    """비동기 파이프라인 실행 요청"""
    callback_url: Optional[str] = Field(None, description="결과 콜백 URL")


class ExecuteResponse(BaseModel):
    """파이프라인 실행 응답"""
    answer: str = Field(..., description="생성된 응답")
    sources: List[Source] = Field(default_factory=list, description="참조 소스")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class AsyncJobResponse(BaseModel):
    """비동기 작업 응답"""
    job_id: str = Field(..., description="작업 ID")
    message: str = Field(..., description="상태 메시지")


class JobStatus(BaseModel):
    """작업 상태 정보"""
    job_id: str = Field(..., description="작업 ID")
    status: str = Field(..., description="작업 상태 (pending|processing|completed|failed)")
    created_at: str = Field(..., description="작업 생성 시간")
    updated_at: str = Field(..., description="작업 업데이트 시간")
    result: Optional[ExecuteResponse] = Field(None, description="실행 결과 (완료된 경우)")
    error: Optional[str] = Field(None, description="오류 메시지 (실패한 경우)")


class ErrorResponse(BaseModel):
    """오류 응답"""
    status: str = Field("error", description="응답 상태")
    error: str = Field(..., description="오류 유형")
    message: str = Field(..., description="오류 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="추가 오류 상세 정보")
```

## 4. 데이터베이스 모델 (SQL 스키마)

### 4.1 모듈 테이블

```sql
CREATE TABLE modules (
    id SERIAL PRIMARY KEY,
    module_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    description TEXT,
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_modules_type ON modules(type);
```

### 4.2 파이프라인 테이블

```sql
CREATE TABLE pipelines (
    id SERIAL PRIMARY KEY,
    pipeline_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

### 4.3 파이프라인 모듈 테이블

```sql
CREATE TABLE pipeline_modules (
    id SERIAL PRIMARY KEY,
    pipeline_id VARCHAR(64) NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    module_id VARCHAR(64) NOT NULL REFERENCES modules(module_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(pipeline_id, step)
);

CREATE INDEX idx_pipeline_modules_pipeline_id ON pipeline_modules(pipeline_id);
CREATE INDEX idx_pipeline_modules_module_id ON pipeline_modules(module_id);
```

### 4.4 비동기 작업 테이블

```sql
CREATE TABLE async_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(64) UNIQUE NOT NULL,
    pipeline_id VARCHAR(64) NOT NULL REFERENCES pipelines(pipeline_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    parameters JSONB,
    callback_url TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    result JSONB,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_async_jobs_status ON async_jobs(status);
CREATE INDEX idx_async_jobs_created_at ON async_jobs(created_at);
```