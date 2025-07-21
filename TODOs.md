# MCP-RAG-Control 순차 개발 계획

## 🚀 단일 Agent 개발 전략 개요

이 문서는 하나의 AI Agent가 순차적으로 모든 컴포넌트를 개발하는 계획입니다.
체계적이고 안정적인 개발을 위해 의존성 순서에 따라 단계별로 진행합니다.

## 📊 개발 타임라인

```
Phase 1: 기초 인프라 완성 (1-2주)
├── 프로젝트 인프라 보완
├── 데이터 모델 최적화 
└── 유틸리티 시스템 강화

Phase 2: 핵심 시스템 구축 (2-3주)
├── MCP 어댑터 시스템 완성
├── 등록 저장소 시스템 구현
└── 종합 테스트 프레임워크

Phase 3: 통합 및 제어 (1-2주)
└── LangGraph 기반 중앙 컨트롤러

Phase 4: 인터페이스 구현 (2-3주)
├── FastAPI 백엔드 완성
└── Streamlit 웹 인터페이스

Phase 5: 완성 및 배포 (1-2주)
├── 예제 및 데모 구현
└── 배포 및 운영 시스템
```

**총 개발 기간: 7-12주 (안정적인 순차 개발)**

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
src/mcp_rag_control/models/
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
src/mcp_rag_control/utils/
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

### **Phase 2: 핵심 시스템 구축**

#### **2.1 MCP 어댑터 시스템**
**브랜치: `feature/mcp-adapters`**
**예상 시간: 3-4일**
**의존성: Agent B (데이터 모델)**

**담당 파일:**
```
src/mcp_rag_control/adapters/
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
src/mcp_rag_control/registry/
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

#### **3.1 LangGraph 중앙 컨트롤러**
**브랜치: `feature/langgraph-controller`**
**예상 시간: 4-5일**
**의존성: Agent D, E (어댑터, 저장소)**

**담당 파일:**
```
src/mcp_rag_control/controller/
├── __init__.py
├── controller.py
├── workflow.py
├── executor.py
├── state_manager.py
└── exceptions.py
```

**주요 작업:**
- [ ] LangGraph 기반 중앙 컨트롤러 구현
- [ ] 워크플로우 정의 및 실행 엔진
- [ ] 모듈 간 통신 관리
- [ ] 파이프라인 실행 엔진
- [ ] 상태 관리 시스템
- [ ] 에러 처리 및 복구 로직
- [ ] 컨트롤러 통합 테스트

**산출물:**
- 중앙집중식 워크플로우 관리
- 복잡한 RAG 파이프라인 실행 능력
- 확장 가능한 제어 아키텍처

---

### **Phase 4: 인터페이스 구현**

#### **4.1 FastAPI 백엔드**
**브랜치: `feature/fastapi-backend`**
**예상 시간: 3-4일**
**의존성: Agent B, C, E (모델, 유틸리티, 저장소)**

**담당 파일:**
```
src/mcp_rag_control/api/
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
src/mcp_rag_control/web/
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

### 개발 순서 및 규칙

1. **Phase 1**: 기초 인프라 단계별 완성 후 다음 단계 진행
   - 1.1 프로젝트 인프라 보완
   - 1.2 데이터 모델 최적화
   - 1.3 기본 유틸리티 강화

2. **Phase 2**: 순차적 개발로 의존성 문제 최소화
   - 2.1 MCP 어댑터 시스템
   - 2.2 등록 저장소 시스템
   - 2.3 종합 테스트 프레임워크

3. **Phase 3-5**: 각 Phase 완료 후 순차 진행

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

### 🔄 다음 단계: Phase 2 준비 완료

**2.1 MCP 어댑터 시스템** - 기본 구현체 존재, 강화 필요
**2.2 등록 저장소 시스템** - 기본 구현체 존재, 완성도 향상 필요
**2.3 종합 테스트 프레임워크** - 35개 테스트 기반 확장 예정

## 🎯 시작 지침

각 Agent는 다음 단계를 따라 작업을 시작합니다:

1. **브랜치 생성**: `git checkout -b [브랜치명]`
2. **작업 진행**: 담당 파일들 구현
3. **테스트 작성**: 단위 테스트 및 통합 테스트
4. **문서화**: README 및 API 문서 작성
5. **PR 생성**: 코드 리뷰 요청
6. **통합**: 승인 후 main 브랜치로 머지

---

**이 계획에 따라 하나의 AI Agent가 체계적으로 작업하면 안정적이고 고품질의 시스템을 7-12주 내에 구축할 수 있습니다.**