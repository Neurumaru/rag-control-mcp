# MCP-RAG-Control 프로젝트 구조

```
/mcp-rag-control/
├── pyproject.toml               # 프로젝트 메타데이터 및 의존성
├── README.md                    # 영문 README
├── README-KR.md                 # 한글 README
├── src/
│   ├── mcp_rag_control/
│   │   ├── __init__.py
│   │   ├── api/                 # API 인터페이스
│   │   │   ├── __init__.py
│   │   │   ├── app.py           # FastAPI 앱 정의
│   │   │   ├── router.py        # API 라우터
│   │   │   └── endpoints/       # API 엔드포인트
│   │   │       ├── __init__.py
│   │   │       ├── modules.py   # 모듈 관리 엔드포인트
│   │   │       ├── pipelines.py # 파이프라인 관리 엔드포인트
│   │   │       └── execute.py   # 파이프라인 실행 엔드포인트
│   │   │
│   │   ├── controller/          # 컨트롤러
│   │   │   ├── __init__.py
│   │   │   └── controller.py    # LangGraph 기반 중앙 제어 장치
│   │   │
│   │   ├── registry/            # 등록 저장소
│   │   │   ├── __init__.py
│   │   │   ├── module_registry.py  # 모듈 등록 저장소
│   │   │   └── pipeline_registry.py # 파이프라인 등록 저장소
│   │   │
│   │   ├── adapters/            # MCP 어댑터
│   │   │   ├── __init__.py
│   │   │   ├── base_adapter.py  # 기본 어댑터 인터페이스
│   │   │   ├── vector_adapter.py # 벡터 DB 어댑터 (FAISS 등)
│   │   │   └── database_adapter.py # 일반 DB 어댑터
│   │   │
│   │   ├── models/              # 데이터 모델
│   │   │   ├── __init__.py
│   │   │   ├── module.py        # 모듈 스키마
│   │   │   ├── pipeline.py      # 파이프라인 스키마
│   │   │   └── request.py       # API 요청/응답 스키마
│   │   │
│   │   └── utils/               # 유틸리티 함수
│   │       ├── __init__.py
│   │       ├── config.py        # 설정 관리
│   │       └── logger.py        # 로깅
│   │
├── tests/                       # 테스트
│   ├── __init__.py
│   ├── api/                     # API 테스트
│   ├── controller/              # 컨트롤러 테스트
│   ├── registry/                # 등록 저장소 테스트
│   └── adapters/                # 어댑터 테스트
│
└── examples/                    # 예제 코드
    ├── basic_rag/               # 기본 RAG 파이프라인 예제
    └── vector_mcp/              # 벡터 MCP 예제
```