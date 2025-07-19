# MCP-RAG-Control API 설계

## 1. 모듈 관리 API

### 1.1 모듈 등록
- **엔드포인트**: `POST /api/modules`
- **설명**: 새로운 모듈을 시스템에 등록
- **요청 형식**:
```json
{
  "module_id": "unique_module_id",
  "name": "모듈 이름",
  "type": "데이터소스|벡터저장소|임베딩모델|...",
  "description": "모듈 설명",
  "config": {
    "connection_string": "...",
    "api_key": "...",
    "other_params": "..."
  }
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "module": {
    "module_id": "unique_module_id",
    "name": "모듈 이름",
    "type": "데이터소스|벡터저장소|임베딩모델|...",
    "description": "모듈 설명"
  }
}
```

### 1.2 모듈 목록 조회
- **엔드포인트**: `GET /api/modules`
- **설명**: 시스템에 등록된 모듈 목록 조회
- **요청 파라미터**: `?type=데이터소스|벡터저장소|임베딩모델|...` (선택 사항)
- **응답 형식**:
```json
{
  "status": "success",
  "modules": [
    {
      "module_id": "unique_module_id",
      "name": "모듈 이름",
      "type": "데이터소스|벡터저장소|임베딩모델|...",
      "description": "모듈 설명"
    },
    ...
  ]
}
```

### 1.3 모듈 상세 조회
- **엔드포인트**: `GET /api/modules/{module_id}`
- **설명**: 특정 모듈의 상세 정보 조회
- **응답 형식**:
```json
{
  "status": "success",
  "module": {
    "module_id": "unique_module_id",
    "name": "모듈 이름",
    "type": "데이터소스|벡터저장소|임베딩모델|...",
    "description": "모듈 설명",
    "config": {
      "connection_string": "...",
      "api_key": "...",
      "other_params": "..."
    }
  }
}
```

### 1.4 모듈 업데이트
- **엔드포인트**: `PUT /api/modules/{module_id}`
- **설명**: 기존 모듈 정보 업데이트
- **요청 형식**:
```json
{
  "name": "업데이트된 모듈 이름",
  "description": "업데이트된 모듈 설명",
  "config": {
    "updated_param": "..."
  }
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "module": {
    "module_id": "unique_module_id",
    "name": "업데이트된 모듈 이름",
    "type": "데이터소스|벡터저장소|임베딩모델|...",
    "description": "업데이트된 모듈 설명"
  }
}
```

### 1.5 모듈 삭제
- **엔드포인트**: `DELETE /api/modules/{module_id}`
- **설명**: 시스템에서 모듈 삭제
- **응답 형식**:
```json
{
  "status": "success",
  "message": "모듈이 성공적으로 삭제되었습니다."
}
```

## 2. 파이프라인 관리 API

### 2.1 파이프라인 등록
- **엔드포인트**: `POST /api/pipelines`
- **설명**: 새로운 RAG 파이프라인 등록
- **요청 형식**:
```json
{
  "pipeline_id": "unique_pipeline_id",
  "name": "파이프라인 이름",
  "description": "파이프라인 설명",
  "modules": [
    {
      "step": 1,
      "module_id": "data_source_module_id",
      "config": {
        "output_field": "documents"
      }
    },
    {
      "step": 2,
      "module_id": "embedding_model_id",
      "config": {
        "input_field": "documents",
        "output_field": "embeddings"
      }
    },
    {
      "step": 3,
      "module_id": "vector_store_id",
      "config": {
        "input_field": "embeddings",
        "output_field": "results"
      }
    }
  ]
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "pipeline": {
    "pipeline_id": "unique_pipeline_id",
    "name": "파이프라인 이름",
    "description": "파이프라인 설명"
  }
}
```

### 2.2 파이프라인 목록 조회
- **엔드포인트**: `GET /api/pipelines`
- **설명**: 시스템에 등록된 파이프라인 목록 조회
- **응답 형식**:
```json
{
  "status": "success",
  "pipelines": [
    {
      "pipeline_id": "unique_pipeline_id",
      "name": "파이프라인 이름",
      "description": "파이프라인 설명"
    },
    ...
  ]
}
```

### 2.3 파이프라인 상세 조회
- **엔드포인트**: `GET /api/pipelines/{pipeline_id}`
- **설명**: 특정 파이프라인의 상세 정보 조회
- **응답 형식**:
```json
{
  "status": "success",
  "pipeline": {
    "pipeline_id": "unique_pipeline_id",
    "name": "파이프라인 이름",
    "description": "파이프라인 설명",
    "modules": [
      {
        "step": 1,
        "module_id": "data_source_module_id",
        "module_name": "데이터 소스 모듈 이름",
        "config": {
          "output_field": "documents"
        }
      },
      ...
    ]
  }
}
```

### 2.4 파이프라인 업데이트
- **엔드포인트**: `PUT /api/pipelines/{pipeline_id}`
- **설명**: 기존 파이프라인 정보 업데이트
- **요청 형식**:
```json
{
  "name": "업데이트된 파이프라인 이름",
  "description": "업데이트된 파이프라인 설명",
  "modules": [
    {
      "step": 1,
      "module_id": "updated_module_id",
      "config": {
        "output_field": "documents"
      }
    },
    ...
  ]
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "pipeline": {
    "pipeline_id": "unique_pipeline_id",
    "name": "업데이트된 파이프라인 이름",
    "description": "업데이트된 파이프라인 설명"
  }
}
```

### 2.5 파이프라인 삭제
- **엔드포인트**: `DELETE /api/pipelines/{pipeline_id}`
- **설명**: 시스템에서 파이프라인 삭제
- **응답 형식**:
```json
{
  "status": "success",
  "message": "파이프라인이 성공적으로 삭제되었습니다."
}
```

## 3. 파이프라인 실행 API

### 3.1 파이프라인 실행
- **엔드포인트**: `POST /api/execute`
- **설명**: 파이프라인을 실행하여 사용자 질문에 응답
- **요청 형식**:
```json
{
  "pipeline_id": "unique_pipeline_id",
  "query": "사용자 질문",
  "parameters": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "additional_params": "..."
  }
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "result": {
    "answer": "생성된 응답",
    "sources": [
      {
        "source_id": "source_1",
        "content": "참조된 소스 내용",
        "relevance_score": 0.92
      },
      ...
    ],
    "metadata": {
      "processing_time": 0.543,
      "token_usage": {
        "prompt_tokens": 150,
        "completion_tokens": 200,
        "total_tokens": 350
      }
    }
  }
}
```

### 3.2 비동기 파이프라인 실행
- **엔드포인트**: `POST /api/execute/async`
- **설명**: 파이프라인을 비동기적으로 실행
- **요청 형식**:
```json
{
  "pipeline_id": "unique_pipeline_id",
  "query": "사용자 질문",
  "parameters": {
    "max_tokens": 1000,
    "temperature": 0.7,
    "additional_params": "..."
  },
  "callback_url": "https://example.com/callback"
}
```
- **응답 형식**:
```json
{
  "status": "success",
  "job_id": "async_job_id",
  "message": "비동기 작업이 시작되었습니다."
}
```

### 3.3 비동기 작업 상태 조회
- **엔드포인트**: `GET /api/execute/jobs/{job_id}`
- **설명**: 비동기 작업의 상태 조회
- **응답 형식**:
```json
{
  "status": "success",
  "job": {
    "job_id": "async_job_id",
    "status": "pending|processing|completed|failed",
    "created_at": "2024-05-22T15:30:00Z",
    "updated_at": "2024-05-22T15:32:00Z",
    "result": {} // 완료된 경우에만 결과 포함
  }
}
```