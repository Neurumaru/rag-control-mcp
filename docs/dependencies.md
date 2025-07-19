# MCP-RAG-Control 프로젝트 의존성

## 핵심 의존성

### 웹 프레임워크
- **FastAPI**: 고성능 웹 프레임워크 (API 구현)
- **Uvicorn**: ASGI 서버 (FastAPI 실행)
- **Pydantic**: 데이터 검증 및 설정 관리

### 데이터베이스
- **SQLAlchemy**: SQL 툴킷 및 ORM (모듈 및 파이프라인 저장소)
- **Alembic**: 데이터베이스 마이그레이션
- **PostgreSQL** (선택적): 프로덕션 데이터베이스

### 벡터 저장소
- **FAISS**: Facebook AI Similarity Search (벡터 검색)
- **ChromaDB**: 벡터 데이터베이스 (선택적)

### RAG 및 LLM 통합
- **LangChain**: LLM 애플리케이션 프레임워크
- **LangGraph**: 에이전트 워크플로우 관리
- **Sentence-Transformers**: 텍스트 임베딩 생성
- **MCP-Python**: Model Context Protocol 클라이언트/서버 SDK
- **LangChain-MCP-Adapters**: LangChain과 MCP 통합 라이브러리

### 유틸리티
- **Python-dotenv**: 환경 변수 관리
- **Loguru**: 로깅
- **Pydantic-settings**: 설정 관리

### 비동기 처리
- **asyncio**: 비동기 I/O
- **aiohttp**: 비동기 HTTP 클라이언트

## 개발 의존성

### 테스팅
- **Pytest**: 테스트 프레임워크
- **Pytest-asyncio**: 비동기 테스트 지원
- **Pytest-cov**: 코드 커버리지
- **Httpx**: FastAPI 테스트 클라이언트

### 코드 품질
- **Black**: 코드 포맷팅
- **Isort**: 임포트 정렬
- **Flake8**: 린팅
- **Mypy**: 타입 체킹

### 문서화
- **Sphinx**: 문서 생성
- **MkDocs**: 문서 사이트 생성

## pyproject.toml 예시

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mcp-rag-control"
version = "0.1.0"
description = "에이전트 기반 RAG 시스템을 위한 맵 기반 제어 아키텍처"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
license = {text = "MIT"}
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "fastapi>=0.95.0",
    "uvicorn>=0.22.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.10.0",
    "faiss-cpu>=1.7.4",
    "langchain>=0.1.0",
    "langgraph>=0.0.15",
    "mcp-python>=0.1.0",
    "sentence-transformers>=2.2.2",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.0",
    "aiohttp>=3.8.5",
    "streamlit>=1.28.0",
    "plotly>=5.15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.1",
    "pytest-asyncio>=0.20.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.24.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.2.0",
]
docs = [
    "sphinx>=7.0.0",
    "mkdocs>=1.4.3",
]
postgres = [
    "psycopg2-binary>=2.9.6",
]
chroma = [
    "chromadb>=0.4.6",
]
web = [
    "streamlit>=1.28.0",
    "plotly>=5.15.0",
    "pandas>=2.0.0",
]

[project.urls]
"Homepage" = "https://github.com/yourusername/mcp-rag-control"
"Bug Tracker" = "https://github.com/yourusername/mcp-rag-control/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/mcp_rag_control"]

[tool.hatch.build.targets.sdist]
include = [
    "src/mcp_rag_control",
    "README.md",
    "README-KR.md",
    "LICENSE",
]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
asyncio_mode = "auto"
```

## 설치 지침

1. 기본 설치:
```bash
pip install -e .
```

2. 개발 환경 설치:
```bash
pip install -e ".[dev]"
```

3. 문서화 도구 설치:
```bash
pip install -e ".[docs]"
```

4. PostgreSQL 지원 설치:
```bash
pip install -e ".[postgres]"
```

5. ChromaDB 지원 설치:
```bash
pip install -e ".[chroma]"
```

6. 웹 인터페이스 지원 설치:
```bash
pip install -e ".[web]"
```

7. 모든 의존성 설치:
```bash
pip install -e ".[dev,docs,postgres,chroma,web]"
```