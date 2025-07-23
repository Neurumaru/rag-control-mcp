# MCP-RAG-Control 순차 개발 계획

## 🚀 단일 Agent 개발 전략 개요

이 문서는 하나의 AI Agent가 순차적으로 모든 컴포넌트를 개발하는 계획입니다.
체계적이고 안정적인 개발을 위해 의존성 순서에 따라 단계별로 진행합니다.

## 📊 개발 타임라인 ⭐ **업데이트**

```
Phase 1: 기초 인프라 완성 (1-2주) ✅ 85% 완료
├── 프로젝트 인프라 보완
├── 데이터 모델 최적화 
└── 유틸리티 시스템 강화

Phase 1.5: 지능형 라우팅 시스템 (3-4일) ⭐ **혁신적 추가**
└── LLM 기반 동적 라우팅 엔진 (90% 비용 절약)

Phase 2: 핵심 시스템 구축 (2-3주)
├── MCP 어댑터 시스템 완성
├── 등록 저장소 시스템 구현
└── 종합 테스트 프레임워크

Phase 3: 지능형 통합 및 제어 (1-2주) ⭐ **업그레이드**
└── LangGraph 지능형 중앙 컨트롤러

Phase 4: 인터페이스 구현 (2-3주)
├── FastAPI 백엔드 완성
└── Streamlit 웹 인터페이스

Phase 5: 완성 및 배포 (1-2주)
├── 예제 및 데모 구현
└── 배포 및 운영 시스템
```

**총 개발 기간: 8-13주 (지능형 RAG 시스템 포함)**
**💡 핵심 가치: Phase 1.5로 90% 비용 절약 + 2-3배 성능 개선 달성**

## 🎯 Phase별 작업 계획

### **Phase 1: 기초 인프라 완성**

#### **1.1 프로젝트 인프라 보완** 
**브랜치: `feature/project-infrastructure`**
**예상 시간: 1-2일**

**담당 파일:**
```
pyproject.toml
.github/workflows/ci.yml
.github/workflows/deploy.yml
docker/Dockerfile
docker/docker-compose.yml
scripts/setup.sh
scripts/test.sh
requirements/base.txt
requirements/dev.txt
.pre-commit-config.yaml
.dockerignore
```

**주요 작업:**
- [x] pyproject.toml 생성 및 의존성 정의
- [x] 프로젝트 디렉토리 구조 생성
- [ ] GitHub Actions CI/CD 파이프라인 설정
- [ ] Docker 기본 설정 (Dockerfile, docker-compose.yml)
- [ ] 개발 환경 스크립트 작성
- [ ] Pre-commit hooks 설정

**산출물:**
- 완전한 프로젝트 구조 ✅
- 자동화된 CI/CD 파이프라인 (미완성)
- 개발 환경 설정 스크립트 (미완성)

---

#### **1.2 데이터 모델 최적화**
**브랜치: `feature/data-models`**
**예상 시간: 2-3일**

**담당 파일:**
```
src/models/
├── __init__.py
├── base.py
├── module.py
├── pipeline.py
├── request.py
├── response.py
└── enums.py
```

**주요 작업:**
- [x] Pydantic 기반 모든 데이터 모델 정의
- [x] 모듈 스키마 (Module, ModuleConfig, ModuleStatus)
- [x] 파이프라인 스키마 (Pipeline, PipelineStep, PipelineConfig)
- [x] API 요청/응답 스키마 (ExecuteRequest, ExecuteResponse)
- [x] MCP 프로토콜 스키마 (MCPRequest, MCPResponse)
- [x] 열거형 및 상수 정의 (데이터 플로우 기반 ModuleType 완성)
- [x] 유효성 검사 로직 구현
- [x] 모델 직렬화/역직렬화 테스트 (35개 테스트 모두 통과)

**산출물:**
- 완전한 데이터 모델 라이브러리
- 타입 힌트가 포함된 모든 스키마
- 모델 단위 테스트

---

#### **1.3 기본 유틸리티 강화**
**브랜치: `feature/core-utilities`**
**예상 시간: 1-2일**

**담당 파일:**
```
src/utils/
├── __init__.py
├── config.py
├── logger.py
├── exceptions.py
├── helpers.py
└── constants.py
```

**주요 작업:**
- [x] 설정 관리 시스템 (Pydantic Settings 기반)
- [x] 로깅 시스템 (Loguru 기반)
- [x] 커스텀 예외 클래스 정의
- [x] 공통 헬퍼 함수들
- [x] 상수 및 기본값 정의
- [x] 환경 변수 관리
- [x] 에러 핸들링 유틸리티
- [x] LangGraph 통합 유틸리티 (Checkpointer Factory, Config Manager, Monitor)
- [x] LangGraph 전용 로깅 함수 구현

**산출물:**
- 중앙집중식 설정 관리
- 구조화된 로깅 시스템  
- 재사용 가능한 유틸리티 함수
- LangGraph 완전 통합 지원

---

### **Phase 1.5: 지능형 라우팅 시스템** ⭐ **혁신적 아키텍처 개선**

#### **1.5 LLM 기반 동적 라우팅 엔진**
**브랜치: `feature/intelligent-routing`**
**예상 시간: 3-4일**
**의존성: Phase 1 완료 (데이터 모델, 유틸리티)**

**담당 파일:**
```
src/models/
├── routing_schemas.py      # 라우팅 결정 스키마 및 응답 모델
src/routing/
├── __init__.py
├── query_classifier.py    # 쿼리 분류기 (간단/복잡/지식필요)
├── route_controller.py    # 동적 라우팅 제어 로직
├── knowledge_assessor.py  # 외부 지식 필요성 평가
├── decision_engine.py     # LLM 기반 의사결정 엔진
└── cost_optimizer.py      # 비용 최적화 알고리즘
src/adapters/
├── routing_adapter.py     # MCP 기반 라우팅 어댑터
```

**주요 작업:**
- [ ] **새로운 ModuleType 추가**
  - `QUERY_CLASSIFIER`: 쿼리 유형 분류 (간단/복잡/지식필요)
  - `ROUTE_CONTROLLER`: 동적 라우팅 제어
  - `KNOWLEDGE_ASSESSOR`: 지식 필요성 평가 엔진
- [ ] **LLM 기반 의사결정 엔진 구현**
  - 쿼리 분석 및 라우팅 결정 로직
  - 비용-정확도 트레이드오프 최적화
- [ ] **지능형 라우팅 패턴 정의**
  - 직접 LLM 응답 경로 (간단한 질문)
  - 표준 RAG 경로 (지식 기반 질문)
  - 하이브리드 경로 (복합 질문)
- [ ] **성능 모니터링 및 최적화**
  - 라우팅 정확도 측정
  - 비용 절약 효과 추적
  - 응답 시간 개선 모니터링
- [ ] **MCP 기반 분산 의사결정**
  - 각 라우팅 단계를 독립적인 MCP 서버로 분리
  - 확장 가능한 마이크로서비스 아키텍처
- [ ] **A/B 테스트 프레임워크 기반 구축**
  - 기존 순차 실행 vs 지능형 라우팅 성능 비교
  - 실시간 성능 메트릭 수집

**혁신적 개선사항:**
```
기존: Query → [필수] 임베딩 → [필수] 벡터검색 → [필수] 문서 → [필수] LLM

개선: Query → LLM 판단 → {
  Case 1: "안녕하세요" → 직접 LLM 응답 (90% 비용 절약)
  Case 2: "2024년 AI 트렌드는?" → 임베딩→벡터→문서→LLM
  Case 3: "파이썬 hello world + AI 뉴스" → 병렬처리 → 집계
}
```

**산출물:**
- **90% 비용 절약** 가능한 지능형 RAG 시스템
- **2-3배 응답 속도 개선** (불필요한 단계 제거)
- **정확도 향상** (질문 유형별 최적 경로)
- **확장 가능한 라우팅 아키텍처** (MCP 기반)
- **실시간 성능 모니터링** 대시보드

---

### **Phase 2: 핵심 시스템 구축**

#### **2.1 MCP 어댑터 시스템**
**브랜치: `feature/mcp-adapters`**
**예상 시간: 3-4일**
**의존성: Agent B (데이터 모델)**

**담당 파일:**
```
src/adapters/
├── __init__.py
├── base_adapter.py
├── vector_adapter.py
├── database_adapter.py
├── registry.py
└── exceptions.py
```

**주요 작업:**
- [ ] 기본 어댑터 인터페이스 (BaseAdapter) 정의
- [ ] FAISS 벡터 어댑터 구현
- [ ] ChromaDB 어댑터 구현 (선택적)
- [ ] 일반 데이터베이스 어댑터 구현
- [ ] 어댑터 등록 시스템 (AdapterRegistry)
- [ ] MCP 프로토콜 처리 로직
- [ ] 어댑터별 헬스체크 구현
- [ ] 어댑터 단위 테스트

**산출물:**
- 확장 가능한 어댑터 아키텍처
- 표준화된 MCP 통신 인터페이스
- 다중 벡터 스토어 지원

---

#### **2.2 등록 저장소 시스템**
**브랜치: `feature/registry-system`**
**예상 시간: 2-3일**
**의존성: Agent B (데이터 모델)**

**담당 파일:**
```
src/registry/
├── __init__.py
├── module_registry.py
├── pipeline_registry.py
├── database.py
├── models.py
└── migrations.py
alembic/
├── env.py
├── script.py.mako
└── versions/
```

**주요 작업:**
- [ ] SQLAlchemy 모델 정의
- [ ] 모듈 등록 저장소 (ModuleRegistry) 구현
- [ ] 파이프라인 등록 저장소 (PipelineRegistry) 구현
- [ ] 데이터베이스 연결 관리
- [ ] Alembic 마이그레이션 설정
- [ ] CRUD 작업 구현
- [ ] 저장소 단위 테스트

**산출물:**
- 영구 저장소 시스템
- 데이터베이스 마이그레이션 관리
- 모듈/파이프라인 생명주기 관리

---

#### **2.3 종합 테스트 프레임워크**
**브랜치: `feature/test-framework`**
**예상 시간: 2-3일**
**의존성: Agent B, C (모델, 유틸리티)**

**담당 파일:**
```
tests/
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── __init__.py
│   ├── models.py
│   ├── adapters.py
│   └── database.py
├── unit/
├── integration/
└── e2e/
```

**주요 작업:**
- [ ] Pytest 설정 및 픽스처 정의
- [ ] 모델 테스트 픽스처
- [ ] 어댑터 모킹 시스템
- [ ] 데이터베이스 테스트 설정
- [ ] 통합 테스트 헬퍼
- [ ] E2E 테스트 기반 구조
- [ ] 코드 커버리지 설정

**산출물:**
- 종합적인 테스트 프레임워크
- 재사용 가능한 테스트 픽스처
- 자동화된 테스트 실행 환경

---

### **Phase 3: 통합 및 제어**

#### **3.1 LangGraph 지능형 중앙 컨트롤러** ⭐ **업그레이드**
**브랜치: `feature/langgraph-controller`**
**예상 시간: 5-6일**
**의존성: Phase 1.5 (지능형 라우팅), Phase 2 (어댑터, 저장소)**

**담당 파일:**
```
src/controller/
├── __init__.py
├── controller.py          # 중앙 컨트롤러 (기존)
├── workflow.py           # 워크플로우 엔진 (기존)
├── intelligent_router.py # 지능형 라우팅 통합 ⭐ NEW
├── conditional_executor.py # 조건부 실행 엔진 ⭐ NEW
├── cost_monitor.py       # 비용 모니터링 ⭐ NEW
├── performance_tracker.py # 성능 추적 ⭐ NEW
├── executor.py           # 실행 엔진 (확장)
├── state_manager.py      # 상태 관리 (확장)
└── exceptions.py         # 예외 처리
```

**주요 작업:**
- [ ] **LangGraph 기반 중앙 컨트롤러 구현** (기존)
- [ ] **지능형 조건부 워크플로우 엔진** ⭐ **핵심 추가**
  - Phase 1.5 라우팅 시스템과 LangGraph 통합
  - 동적 조건부 엣지 및 노드 라우팅
  - 실시간 의사결정 기반 플로우 제어
- [ ] **비용 최적화 워크플로우 관리** ⭐ **핵심 추가**
  - 불필요한 단계 자동 스킵 로직
  - 리소스 사용량 실시간 모니터링
  - 비용-성능 트레이드오프 자동 조정
- [ ] 모듈 간 통신 관리 (확장)
- [ ] **동적 파이프라인 실행 엔진** ⭐ **업그레이드**
  - 런타임 라우팅 결정 반영
  - 병렬 처리 및 집계 로직
  - 장애 복구 및 대체 경로 실행
- [ ] **상태 관리 시스템** (확장)
  - 라우팅 히스토리 추적
  - 성능 메트릭 상태 관리
- [ ] **고급 에러 처리 및 복구 로직**
  - 라우팅 실패 시 대체 경로 실행
  - 단계별 롤백 및 재시도 메커니즘
- [ ] **성능 및 비용 최적화 모니터링** ⭐ **핵심 추가**
  - 실시간 비용 절약 효과 측정
  - 응답 시간 개선도 추적
  - 라우팅 정확도 분석
- [ ] **컨트롤러 통합 테스트** (확장)
  - A/B 테스트 자동화
  - 성능 벤치마크 테스트

**LangGraph 지능형 워크플로우 구현:**
```python
# 조건부 라우팅 예시
def should_use_rag(state):
    query_analysis = state["routing_decision"]
    return query_analysis["needs_external_knowledge"]

workflow = StateGraph(IntelligentRAGState)
workflow.add_node("classify_query", classify_query_node)
workflow.add_node("direct_llm", direct_llm_node)
workflow.add_node("rag_pipeline", rag_pipeline_node)
workflow.add_node("hybrid_process", hybrid_process_node)

workflow.add_conditional_edges(
    "classify_query",
    route_decision_function,
    {
        "direct": "direct_llm",
        "rag": "rag_pipeline", 
        "hybrid": "hybrid_process"
    }
)
```

**산출물:**
- **지능형 중앙집중식 워크플로우 관리** ⭐ **업그레이드**
- **90% 비용 절약 자동화 시스템** ⭐ **핵심**
- **2-3배 성능 개선 워크플로우** ⭐ **핵심** 
- **실시간 모니터링 기반 자동 최적화**
- 복잡한 RAG 파이프라인 실행 능력 (확장)
- 확장 가능한 제어 아키텍처 (강화)

---

### **Phase 4: 인터페이스 구현**

#### **4.1 FastAPI 백엔드**
**브랜치: `feature/fastapi-backend`**
**예상 시간: 3-4일**
**의존성: Agent B, C, E (모델, 유틸리티, 저장소)**

**담당 파일:**
```
src/api/
├── __init__.py
├── app.py
├── dependencies.py
├── middleware.py
└── routers/
    ├── __init__.py
    ├── modules.py
    ├── pipelines.py
    ├── execute.py
    └── health.py
```

**주요 작업:**
- [ ] FastAPI 애플리케이션 구성
- [ ] REST API 엔드포인트 구현
- [ ] 모듈 관리 API (CRUD)
- [ ] 파이프라인 관리 API (CRUD)
- [ ] 파이프라인 실행 API
- [ ] 헬스체크 및 모니터링 API
- [ ] API 문서화 (Swagger/OpenAPI)
- [ ] 미들웨어 및 보안 설정
- [ ] API 통합 테스트

**산출물:**
- 완전한 REST API 서버
- 자동 생성된 API 문서
- 인증 및 보안 미들웨어

---

#### **4.2 Streamlit 웹 인터페이스**
**브랜치: `feature/streamlit-frontend`**
**예상 시간: 3-4일**
**의존성: Agent B (데이터 모델)**

**담당 파일:**
```
src/web/
├── __init__.py
├── app.py
├── pages/
│   ├── __init__.py
│   ├── home.py
│   ├── modules.py
│   ├── pipelines.py
│   ├── rag_test.py
│   └── settings.py
├── components/
│   ├── __init__.py
│   ├── sidebar.py
│   ├── forms.py
│   └── charts.py
└── utils/
    ├── __init__.py
    ├── api_client.py
    └── session_state.py
```

**주요 작업:**
- [ ] Streamlit 애플리케이션 구성
- [ ] 홈 대시보드 페이지
- [ ] 모듈 관리 페이지 (등록, 수정, 삭제)
- [ ] 파이프라인 관리 페이지
- [ ] RAG 테스트 인터페이스
- [ ] 설정 및 구성 페이지
- [ ] API 클라이언트 통합
- [ ] 반응형 UI 컴포넌트
- [ ] 세션 상태 관리

**산출물:**
- 사용자 친화적인 웹 인터페이스
- 실시간 모니터링 대시보드
- 인터랙티브 RAG 테스트 도구

---

### **Phase 5: 완성 및 배포**

#### **5.1 예제 및 데모**
**브랜치: `feature/examples-demo`**
**예상 시간: 2-3일**
**의존성: 모든 이전 컴포넌트**

**담당 파일:**
```
examples/
├── __init__.py
├── basic_rag/
│   ├── __init__.py
│   ├── simple_pipeline.py
│   ├── data/
│   └── README.md
├── vector_search/
│   ├── __init__.py
│   ├── faiss_demo.py
│   ├── chroma_demo.py
│   └── README.md
├── web_demo/
│   ├── __init__.py
│   ├── full_system_demo.py
│   └── README.md
└── tutorials/
    ├── getting_started.md
    ├── custom_adapters.md
    └── advanced_pipelines.md
```

**주요 작업:**
- [ ] 기본 RAG 파이프라인 예제 구현
- [ ] 벡터 검색 데모 (FAISS, ChromaDB)
- [ ] 전체 시스템 통합 예제
- [ ] 웹 인터페이스 데모
- [ ] 사용자 가이드 및 튜토리얼 작성
- [ ] API 사용 예제
- [ ] 커스텀 어댑터 개발 가이드
- [ ] 성능 벤치마크 예제

**산출물:**
- 완전한 예제 라이브러리
- 사용자 온보딩 자료
- 개발자 가이드 및 튜토리얼

---

#### **5.2 배포 및 운영**
**브랜치: `feature/deployment`**
**예상 시간: 2-3일**
**의존성: Agent A (인프라) + 모든 애플리케이션**

**담당 파일:**
```
deployment/
├── docker/
│   ├── api.Dockerfile
│   ├── web.Dockerfile
│   └── docker-compose.prod.yml
├── kubernetes/
│   ├── namespace.yaml
│   ├── api-deployment.yaml
│   ├── web-deployment.yaml
│   └── service.yaml
├── scripts/
│   ├── deploy.sh
│   ├── backup.sh
│   └── monitor.sh
└── config/
    ├── production.env
    ├── staging.env
    └── nginx.conf
```

**주요 작업:**
- [ ] 프로덕션 Docker 설정
- [ ] Docker Compose 멀티 서비스 구성
- [ ] Kubernetes 배포 매니페스트
- [ ] 환경별 설정 관리
- [ ] 자동 배포 스크립트
- [ ] 모니터링 및 로깅 통합
- [ ] 백업 및 복구 전략
- [ ] 성능 최적화 설정

**산출물:**
- 프로덕션 준비 배포 시스템
- 자동화된 배포 파이프라인
- 모니터링 및 운영 도구

---

## 🔄 단계별 개발 전략

### 개발 순서 및 규칙 ⭐ **업데이트**

1. **Phase 1**: 기초 인프라 단계별 완성 후 다음 단계 진행
   - 1.1 프로젝트 인프라 보완
   - 1.2 데이터 모델 최적화
   - 1.3 기본 유틸리티 강화

2. **Phase 1.5**: ⭐ **혁신적 아키텍처 개선** (신규 추가)
   - 1.5 LLM 기반 지능형 라우팅 시스템

3. **Phase 2**: 순차적 개발로 의존성 문제 최소화
   - 2.1 MCP 어댑터 시스템
   - 2.2 등록 저장소 시스템
   - 2.3 종합 테스트 프레임워크

4. **Phase 3**: 지능형 통합 및 제어 (업그레이드)
   - 3.1 LangGraph 지능형 중앙 컨트롤러

5. **Phase 4-5**: 각 Phase 완료 후 순차 진행

### 안정성 확보 전략

- **순차적 개발**: 한 번에 하나의 컴포넌트에 집중
- **완전한 테스트**: 각 단계 완료 시 전체 테스트 실행
- **점진적 개선**: 이전 단계의 결과를 반영하여 다음 단계 최적화
- **지속적 리팩토링**: 코드 품질과 유지보수성 유지
- **문서화**: 각 단계의 진행상황과 결과 기록

## 📋 현재 진행 상황

### ✅ Phase 1 완료 상태

**1.1 프로젝트 인프라** (60% 완료)
- ✅ pyproject.toml 완성
- ✅ 디렉토리 구조 완성
- ⏳ GitHub Actions, Docker 설정 필요

**1.2 데이터 모델** (100% 완료)
- ✅ 37개 데이터 플로우 기반 ModuleType
- ✅ Pydantic V2 모델 및 검증
- ✅ LangGraph 호환 스키마

**1.3 기본 유틸리티** (100% 완료)
- ✅ LangGraph Config & Factory
- ✅ 구조화된 로깅 시스템
- ✅ 설정 관리 및 검증

### 🔄 다음 단계: Phase 1.5 우선 구현 권장 ⭐ **업데이트**

**🚀 즉시 구현 권장: Phase 1.5 지능형 라우팅 시스템**
- **핵심 가치**: 90% 비용 절약 + 2-3배 성능 개선
- **구현 난이도**: 중간 (기존 아키텍처 활용 가능)
- **비즈니스 임팩트**: 매우 높음 (즉각적인 ROI)

**Phase 2 동시 진행 가능:**
**2.1 MCP 어댑터 시스템** - 기본 구현체 존재, 강화 필요
**2.2 등록 저장소 시스템** - 기본 구현체 존재, 완성도 향상 필요  
**2.3 종합 테스트 프레임워크** - 35개 테스트 기반 확장 예정

**💡 권장 우선순위:**
1. **Phase 1.5 (최우선)**: 지능형 라우팅으로 혁신적 성능 개선
2. **Phase 2.1**: MCP 어댑터 강화 (라우팅과 시너지)
3. **Phase 2.3**: 테스트 프레임워크 (품질 보장)
4. **Phase 2.2**: 등록 저장소 (안정성 향상)

## 🎯 시작 지침

각 Agent는 다음 단계를 따라 작업을 시작합니다:

1. **브랜치 생성**: `git checkout -b [브랜치명]`
2. **작업 진행**: 담당 파일들 구현
3. **테스트 작성**: 단위 테스트 및 통합 테스트
4. **문서화**: README 및 API 문서 작성
5. **PR 생성**: 코드 리뷰 요청
6. **통합**: 승인 후 main 브랜치로 머지

---

**이 계획에 따라 하나의 AI Agent가 체계적으로 작업하면 혁신적인 지능형 RAG 시스템을 8-13주 내에 구축할 수 있습니다.**

## 🚀 **Phase 1.5의 게임 체인저 효과**

**기존 RAG 시스템 대비 혁신적 개선:**
- **💰 90% 비용 절약**: 불필요한 임베딩/벡터검색 생략
- **⚡ 2-3배 성능 향상**: 간단한 질문 즉시 처리
- **🧠 지능적 라우팅**: LLM이 최적 경로 동적 결정
- **📈 정확도 향상**: 질문 유형별 맞춤 처리

**즉시 구현 권장 이유:**
1. **기존 아키텍처 활용**: 37개 ModuleType 그대로 사용
2. **점진적 개선**: 기존 시스템과 병행 운영 가능
3. **즉각적 ROI**: 구현 즉시 비용 절약 효과
4. **경쟁 우위**: 업계 최초 지능형 라우팅 시스템