# RAG 시스템 데이터 플로우 기반 ModuleType 재설계

## 개요

기존의 기능별 ModuleType 분류를 RAG 시스템의 **데이터 변환 패턴**과 **입출력 플로우**를 중심으로 재설계했습니다. 이를 통해 더 명확한 데이터 플로우 추적, 모듈 간 호환성 검증, 그리고 확장 가능한 RAG 파이프라인 구성이 가능해졌습니다.

## 핵심 설계 원칙

### 1. 데이터 중심 설계
- **Input → Processing → Output** 패턴을 기반으로 모듈 분류
- 각 모듈의 입출력 데이터 타입과 스키마를 명확히 정의
- 데이터 변환 패턴에 따른 모듈 그룹화

### 2. RAG 파이프라인 최적화
- 전형적인 RAG 워크플로우를 반영한 모듈 타입 정의
- 사용자 질문 → 응답까지의 전체 데이터 플로우 지원
- 멀티모달 및 고급 RAG 패턴 확장성 고려

### 3. 호환성 및 검증
- 모듈 간 데이터 타입 호환성 자동 검증
- 파이프라인 구성 시 데이터 플로우 유효성 확인
- 표준 스키마를 통한 일관성 보장

## 새로운 DataType 체계

```python
class DataType(str, Enum):
    TEXT = \"text\"                # 일반 텍스트
    EMBEDDINGS = \"embeddings\"    # 벡터 임베딩
    VECTORS = \"vectors\"          # 검색된 벡터 결과
    DOCUMENTS = \"documents\"      # 문서 컬렉션
    CONTEXT = \"context\"          # RAG 컨텍스트
    RESPONSE = \"response\"        # LLM 응답
    METADATA = \"metadata\"        # 메타데이터
    MULTIMODAL = \"multimodal\"    # 이미지, 오디오 등
    STRUCTURED = \"structured\"    # JSON, 테이블 등
    STREAM = \"stream\"            # 스트리밍 데이터
```

## 재설계된 ModuleType

### 1. 입력 처리 모듈 (Text → Text/Structured)

#### TEXT_PREPROCESSOR
- **역할**: 텍스트 정제, 분할, 정규화
- **입력**: `TEXT` (원본 텍스트)
- **출력**: `TEXT` (처리된 텍스트 + 메타데이터)
- **용도**: 문서 전처리, 청킹, 노이즈 제거

#### QUERY_ANALYZER  
- **역할**: 질문 분석 및 의도 파악
- **입력**: `TEXT` (사용자 질문)
- **출력**: `STRUCTURED` (분석 결과)
- **용도**: 질문 분류, 엔티티 추출, 검색 전략 결정

#### TEXT_SPLITTER
- **역할**: 문서 분할
- **입력**: `TEXT` (긴 문서)
- **출력**: `TEXT` (분할된 청크들)
- **용도**: 효율적인 벡터화를 위한 문서 분할

### 2. 임베딩 모듈 (Text → Embeddings)

#### EMBEDDING_ENCODER
- **역할**: 텍스트를 벡터로 변환
- **입력**: `TEXT` (텍스트)
- **출력**: `EMBEDDINGS` (벡터 배열)
- **용도**: 문서/질문 임베딩 생성

#### RERANKING_ENCODER
- **역할**: 재순위화용 임베딩 생성
- **입력**: `TEXT` (쿼리 + 문서 페어)
- **출력**: `EMBEDDINGS` (재순위화 벡터)
- **용도**: 검색 결과 재순위화

### 3. 벡터 연산 모듈 (Embeddings → Vectors/Documents)

#### VECTOR_STORE
- **역할**: 벡터 저장 및 관리
- **입력**: `EMBEDDINGS` (벡터 + 메타데이터)
- **출력**: `STRUCTURED` (저장 결과)
- **용도**: 벡터 인덱싱, 저장, 관리

#### SIMILARITY_SEARCH
- **역할**: 유사도 검색
- **입력**: `EMBEDDINGS` (쿼리 벡터)
- **출력**: `VECTORS` (유사한 벡터들)
- **용도**: 의미적 유사성 기반 검색

#### VECTOR_INDEX
- **역할**: 벡터 인덱싱
- **입력**: `EMBEDDINGS` (벡터들)
- **출력**: `STRUCTURED` (인덱스 정보)
- **용도**: 검색 성능 최적화

### 4. 문서 처리 모듈 (Documents → Documents/Context)

#### DOCUMENT_LOADER
- **역할**: 문서 로딩
- **입력**: `STRUCTURED` (파일 참조)
- **출력**: `DOCUMENTS` (문서 컬렉션)
- **용도**: 다양한 포맷의 문서 로딩

#### DOCUMENT_FILTER
- **역할**: 문서 필터링
- **입력**: `DOCUMENTS` (문서들)
- **출력**: `DOCUMENTS` (필터된 문서들)
- **용도**: 관련성, 품질 기반 필터링

#### DOCUMENT_RANKER
- **역할**: 문서 재순위화
- **입력**: `DOCUMENTS` (검색된 문서들)
- **출력**: `DOCUMENTS` (재순위화된 문서들)
- **용도**: 정확도 향상을 위한 재순위화

#### CONTEXT_BUILDER
- **역할**: 컨텍스트 구성
- **입력**: `DOCUMENTS` (관련 문서들)
- **출력**: `CONTEXT` (RAG 컨텍스트)
- **용도**: LLM 입력을 위한 컨텍스트 구성

### 5. 생성 모듈 (Context+Query → Response)

#### LLM_GENERATOR
- **역할**: LLM 응답 생성
- **입력**: `CONTEXT` (컨텍스트 + 질문)
- **출력**: `RESPONSE` (생성된 응답)
- **용도**: 최종 답변 생성

#### PROMPT_TEMPLATE
- **역할**: 프롬프트 템플릿 처리
- **입력**: `CONTEXT` + `TEXT` (템플릿)
- **출력**: `TEXT` (완성된 프롬프트)
- **용도**: 동적 프롬프트 생성

#### RESPONSE_FORMATTER
- **역할**: 응답 포매팅
- **입력**: `RESPONSE` (원본 응답)
- **출력**: `RESPONSE` (포맷된 응답)
- **용도**: 출력 형식 조정

### 6. 메모리 및 상태 관리 (Any → Any with persistence)

#### MEMORY_STORE
- **역할**: 대화 기록 저장
- **특성**: Stateful
- **용도**: 대화 연속성 유지

#### CACHE_MANAGER
- **역할**: 캐싱 관리
- **특성**: Stateful
- **용도**: 성능 최적화

#### SESSION_MANAGER
- **역할**: 세션 관리
- **특성**: Stateful
- **용도**: 사용자 세션 추적

### 7. 데이터 소스 (External → Text/Documents)

#### WEB_SCRAPER
- **역할**: 웹 스크래핑
- **입력**: `STRUCTURED` (URL 정보)
- **출력**: `TEXT` 또는 `DOCUMENTS`
- **용도**: 실시간 웹 콘텐츠 수집

#### DATABASE_CONNECTOR
- **역할**: 데이터베이스 연결
- **입력**: `STRUCTURED` (쿼리 정보)
- **출력**: `STRUCTURED` 또는 `DOCUMENTS`
- **용도**: 구조화된 데이터 접근

#### API_CONNECTOR
- **역할**: API 연결
- **입력**: `STRUCTURED` (API 요청)
- **출력**: `STRUCTURED` 또는 `TEXT`
- **용도**: 외부 서비스 통합

#### FILE_LOADER
- **역할**: 파일 로딩
- **입력**: `STRUCTURED` (파일 경로)
- **출력**: `DOCUMENTS`
- **용도**: 로컬 파일 처리

### 8. 멀티모달 지원 (Multimodal → Text/Embeddings)

#### IMAGE_PROCESSOR
- **역할**: 이미지 처리
- **입력**: `MULTIMODAL` (이미지 데이터)
- **출력**: `TEXT` (이미지 설명)
- **용도**: 이미지 분석 및 설명 생성

#### AUDIO_PROCESSOR
- **역할**: 오디오 처리
- **입력**: `MULTIMODAL` (오디오 데이터)
- **출력**: `TEXT` (음성 인식 결과)
- **용도**: 음성 인식, 오디오 분석

#### OCR_PROCESSOR
- **역할**: OCR 처리
- **입력**: `MULTIMODAL` (이미지 내 텍스트)
- **출력**: `TEXT` (추출된 텍스트)
- **용도**: 이미지에서 텍스트 추출

### 9. 품질 및 평가 (Any → Metadata)

#### QUALITY_CHECKER
- **역할**: 품질 검사
- **출력**: `METADATA` (품질 메트릭)
- **용도**: 응답 품질 평가

#### BIAS_DETECTOR
- **역할**: 편향 탐지
- **출력**: `METADATA` (편향 분석)
- **용도**: 공정성 검증

#### FACTUALITY_CHECKER
- **역할**: 사실성 확인
- **출력**: `METADATA` (사실성 검증)
- **용도**: 사실 정확성 검증

### 10. 플로우 제어 (Any → Any with logic)

#### CONDITIONAL_ROUTER
- **역할**: 조건부 라우팅
- **특성**: 로직 기반 분기
- **용도**: 동적 파이프라인 제어

#### PARALLEL_PROCESSOR
- **역할**: 병렬 처리
- **특성**: 동시 실행
- **용도**: 성능 최적화

#### AGGREGATOR
- **역할**: 결과 집계
- **용도**: 여러 모듈 결과 통합

## 전형적인 RAG 데이터 플로우

### 1. 기본 RAG 파이프라인
```
사용자 질문(TEXT) 
  → TEXT_PREPROCESSOR → 정제된 질문(TEXT)
  → EMBEDDING_ENCODER → 질문 벡터(EMBEDDINGS) 
  → SIMILARITY_SEARCH → 유사 벡터들(VECTORS)
  → DOCUMENT_LOADER → 관련 문서들(DOCUMENTS)
  → CONTEXT_BUILDER → RAG 컨텍스트(CONTEXT)
  → LLM_GENERATOR → 최종 응답(RESPONSE)
```

### 2. 고급 RAG 파이프라인 (재순위화 포함)
```
사용자 질문(TEXT)
  → QUERY_ANALYZER → 질문 분석(STRUCTURED)
  → TEXT_PREPROCESSOR → 정제된 질문(TEXT)
  → EMBEDDING_ENCODER → 질문 벡터(EMBEDDINGS)
  → SIMILARITY_SEARCH → 유사 벡터들(VECTORS)
  → DOCUMENT_RANKER → 재순위화된 문서들(DOCUMENTS)
  → CONTEXT_BUILDER → RAG 컨텍스트(CONTEXT)
  → LLM_GENERATOR → 생성된 응답(RESPONSE)
  → RESPONSE_FORMATTER → 최종 응답(RESPONSE)
```

### 3. 멀티모달 RAG 파이프라인
```
이미지+질문(MULTIMODAL)
  → IMAGE_PROCESSOR → 이미지 설명(TEXT)
  → TEXT_PREPROCESSOR → 통합 텍스트(TEXT)
  → EMBEDDING_ENCODER → 통합 벡터(EMBEDDINGS)
  → SIMILARITY_SEARCH → 유사 벡터들(VECTORS)
  → DOCUMENT_LOADER → 관련 문서들(DOCUMENTS)
  → CONTEXT_BUILDER → 멀티모달 컨텍스트(CONTEXT)
  → LLM_GENERATOR → 멀티모달 응답(RESPONSE)
```

## 주요 개선사항

### 1. 명확한 데이터 플로우
- 각 모듈의 입출력이 명확히 정의됨
- 데이터 변환 과정이 투명하게 추적 가능
- 파이프라인 디버깅과 최적화 용이

### 2. 모듈 간 호환성 보장
- 자동 스키마 검증으로 호환성 확인
- 타입 안전성 보장
- 잘못된 모듈 연결 방지

### 3. 확장성 및 유연성
- 새로운 데이터 타입 쉽게 추가 가능
- 멀티모달 데이터 지원
- 커스텀 변환 로직 지원

### 4. 성능 최적화
- 배치 처리 지원 여부 명시
- 스트리밍 지원 여부 명시
- 메모리 및 지연시간 요구사항 정의

### 5. 품질 관리
- 각 단계별 품질 메트릭 수집
- 오류 처리 및 폴백 메커니즘
- 모니터링 및 로깅 통합

## 사용 예시

### 모듈 생성
```python
from mcp_rag_control.models import ModuleSchemaRegistry, ModuleType

# 자동으로 적절한 capabilities 생성
capabilities = ModuleSchemaRegistry.create_module_capabilities(
    ModuleType.EMBEDDING_ENCODER,
    expected_latency_ms=100.0,
    max_batch_size=32
)

module = Module(
    name=\"Sentence Transformer\",
    module_type=ModuleType.EMBEDDING_ENCODER,
    capabilities=capabilities,
    config=ModuleConfig(
        model_name=\"all-MiniLM-L6-v2\",
        dimensions=384
    )
)
```

### 파이프라인 검증
```python
from mcp_rag_control.models import RAGDataFlowPatterns

# 기본 RAG 플로우 검증
flow = RAGDataFlowPatterns.BASIC_RAG_FLOW
validation = RAGDataFlowPatterns.validate_flow_compatibility(flow)

if validation['is_valid']:
    print(\"파이프라인이 유효합니다!\")
else:
    print(f\"오류: {validation['errors']}\")
```

## 마이그레이션 가이드

### 기존 모듈 타입 → 새 모듈 타입 매핑

| 기존 타입 | 새 타입 | 비고 |
|-----------|---------|------|
| `VECTOR_STORE` | `VECTOR_STORE` | 동일 유지 |
| `DATABASE` | `DATABASE_CONNECTOR` | 명확한 역할 표현 |
| `EMBEDDING` | `EMBEDDING_ENCODER` | 구체적 기능 명시 |
| `LLM` | `LLM_GENERATOR` | 생성 역할 명확화 |
| `RETRIEVER` | `SIMILARITY_SEARCH` | 검색 방식 구체화 |
| `PROCESSOR` | 세분화 필요 | 구체적 처리 타입 선택 |
| `FILTER` | `DOCUMENT_FILTER` | 문서 필터링으로 구체화 |
| `TRANSFORM` | `CUSTOM_TRANSFORM` | 커스텀 변환으로 분류 |

### 기존 코드 업데이트
1. **ModuleType 업데이트**: 새로운 타입으로 변경
2. **Capabilities 추가**: 새로운 `capabilities` 필드 설정
3. **스키마 검증 활용**: `ModuleSchemaRegistry` 사용
4. **데이터 플로우 검증**: `RAGDataFlowPatterns` 활용

## 결론

이번 재설계를 통해 RAG 시스템의 데이터 플로우가 더욱 명확해지고, 모듈 간 상호작용이 예측 가능해졌습니다. 새로운 구조는 다음과 같은 이점을 제공합니다:

1. **투명성**: 데이터가 어떻게 변환되는지 명확히 추적 가능
2. **안정성**: 타입 안전성과 호환성 보장
3. **확장성**: 새로운 요구사항에 쉽게 대응 가능
4. **성능**: 최적화 포인트가 명확히 식별됨
5. **유지보수성**: 모듈 역할과 책임이 명확히 분리됨

이러한 개선을 통해 더욱 견고하고 확장 가능한 RAG 시스템을 구축할 수 있습니다.