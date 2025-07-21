# MCP-RAG-Control 프로젝트 상세 설명

## 프로젝트 개요
mcp-rag-control은 에이전트 기반 RAG(Retrieval-Augmented Generation) 시스템을 위한 맵 기반 제어 아키텍처입니다. 이 시스템은 복잡한 정보 검색과 생성 프로세스를 효율적으로 관리하고 제어하기 위해 설계되었습니다.

## 아키텍처 구성

MCP-RAG-Control 아키텍처는 다음과 같은 주요 컴포넌트로 구성됩니다:

### API 인터페이스
- 사용자 또는 외부 시스템과의 상호작용을 위한 RESTful API 제공
- 모듈 등록, 파이프라인 구성, 쿼리 실행 엔드포인트 제공

### 컨트롤러
- 시스템의 중앙 제어 장치로 모든 요청과 흐름을 조정
- LangGraph 기반으로 복잡한 워크플로우 관리 및 실행
- 모듈 간 통신 및 데이터 흐름 제어

### 모듈 등록 저장소
- 데이터 소스, 벡터 저장소, 임베딩 모델 등 다양한 모듈 관리
- MCP 호환 모듈에 대한 메타데이터 및 설정 정보 저장

### 파이프라인 등록 저장소
- 사용자 정의 RAG 파이프라인 구성 정보 저장
- 모듈 연결 방식, 데이터 흐름, 실행 순서 정의

### MCP 어댑터
- 다양한 외부 시스템과의 표준화된 통신 인터페이스
- 데이터 소스별 맞춤형 MCP 구현 지원

## 컴포넌트 상호작용 흐름

### 모듈 등록

1.  **사용자 요청:** 사용자가 API를 통해 모듈을 등록하면 (`/modules` 엔드포인트 호출).
2.  **컨트롤러 처리:** 컨트롤러는 요청을 받아 모듈 등록 요청을 처리.
3.  **모듈 등록:** 모듈 등록 요청은 모듈 등록 저장소에 저장되고, 등록된 모듈 목록을 반환.

### 파이프라인 등록

1.  **사용자 요청:** 사용자가 API를 통해 파이프라인을 등록하면 (`/pipelines` 엔드포인트 호출).
2.  **컨트롤러 처리:** 컨트롤러는 요청을 받아 파이프라인 등록 요청을 처리.
3.  **파이프라인 등록:** 파이프라인 등록 요청은 파이프라인 등록 저장소에 저장되고, 등록된 파이프라인 목록을 반환.

### 파이프라인 실행

1.  **사용자 요청:** 사용자가 API를 통해 질문을 제출하면 (`/execute` 엔드포인트 호출).
2.  **쿼리 처리:** 컨트롤러는 사용자 질문을 분석하고 해당 파이프라인을 식별.
3.  **파이프라인 실행:** 컨트롤러는 파이프라인 실행을 위해 필요한 모듈을 찾아 순차적으로 실행.
4.  **응답 반환:** 생성된 최종 응답은 API 인터페이스를 통해 사용자에게 전달됨.

## 주요 기술 용어 설명

### RAG (Retrieval-Augmented Generation)
- 기존 정보 검색과 생성형 언어 모델을 결합한 하이브리드 패러다임
- 외부 지식 베이스에서 관련 정보를 검색하여 LLM의 응답을 보강함
- 구식 정보, 환각 현상(hallucination), 도메인 특화 지식 부족 등의 문제를 해결
- 검색(Retrieval), 증강(Augmentation), 생성(Generation) 3단계로 작동
- **상세 예시 시나리오:**
    1.  **사용자 질문:** 사용자가 "최신 금융 상품 추천"이라고 질문합니다.
    2.  **쿼리 처리:** 시스템은 질문 텍스트를 임베딩 벡터(예: 512차원 실수 벡터)로 변환합니다.
    3.  **정보 검색 (Retrieval):**
        *   연결된 금융 상품 데이터베이스(예: 벡터 검색 기능이 있는 SQL 데이터베이스)에 쿼리합니다.
        *   **SQL 예시 (유사도 및 최신순 정렬):**
            ```sql
            SELECT product_id, name, release_date, description
            FROM financial_products
            WHERE release_date > '2024-01-01' -- 예시: 특정 날짜 이후 상품
            ORDER BY vector_distance_cosine(embedding, query_embedding) DESC -- 쿼리 벡터와 유사도 높은 순
            LIMIT 5;
            ```
        *   가장 관련성 높고 최신인 상품 정보(상품명, 출시일, 설명 등) 레코드 셋을 가져옵니다.
    4.  **정보 증강 (Augmentation):** 검색된 상품 정보들을 구조화된 형식(예: JSON, Markdown)으로 정리하여 LLM 프롬프트의 컨텍스트로 구성합니다.
        ```markdown
        [컨텍스트]
        1. 상품명: 스마트 예금 알파, 출시일: 2024-03-15, 특징: AI 기반 자동 금리 조정
        2. 상품명: 글로벌 채권 펀드 플러스, 출시일: 2024-02-28, 특징: 선진국/신흥국 채권 분산 투자
        ...
        ```
    5.  **답변 생성 (Generation):** 구성된 컨텍스트와 원본 질문을 LLM(예: GPT-4)에 전달합니다. LLM은 제공된 최신 정보를 바탕으로 정확하고 상세한 답변을 생성합니다.

### MCP (Model Context Protocol)
- LLM 애플리케이션과 다양한 외부 데이터 소스를 연결하는 표준화된 프로토콜
- RAG 시스템에서 생성 모델이 사용하는 맥락 정보를 관리하고 전달
- 동적이고 양방향 맥락 교환을 가능하게 함
- 다양한 데이터 소스 간의 상호 운용성 제공
- **상세 예시 시나리오:**
    1.  **복합 질문:** 사용자가 특정 금융 상품(예: ID 123)에 대해 "이 상품의 과거 수익률 추이와 관련된 최근 뉴스 기사는 무엇인가요?" 라고 질문합니다.
    2.  **병렬 검색 요청:** 컨트롤러는 이 질문을 분석하여 두 가지 정보가 필요하다고 판단합니다.
        *   수익률 데이터: 시계열 데이터베이스(예: InfluxDB)에 쿼리
        *   관련 뉴스: 벡터 데이터베이스(예: FAISS)에 쿼리
    3.  **MCP 기반 통신:**
        *   컨트롤러는 MCP 표준 요청 형식을 사용하여 각 데이터 소스(InfluxDB MCP, FAISS MCP)에 비동기적으로 쿼리를 보냅니다.
        *   **표준 요청 형식 (예시):**
            ```json
            {
              "source_id": "influxdb_mcp_1",
              "operation": "query_timeseries",
              "params": {"product_id": 123, "metric": "yield", "time_range": "1y"},
              "request_id": "req-abc-1"
            }
            ```
            ```json
            {
              "source_id": "faiss_mcp_2",
              "operation": "vector_similarity_search",
              "params": {"query_embedding": [0.1, 0.5, ...], "product_id": 123, "top_k": 3},
              "request_id": "req-abc-2"
            }
            ```
    4.  **표준 응답 수신:** 각 MCP는 처리가 완료되면 표준 응답 형식으로 결과를 컨트롤러에 반환합니다.
        *   **표준 응답 형식 (예시):**
            ```json
            {
              "source_id": "influxdb_mcp_1",
              "status": "success",
              "data": {"timestamps": [...], "values": [...]},
              "request_id": "req-abc-1"
            }
            ```
            ```json
            {
              "source_id": "faiss_mcp_2",
              "status": "success",
              "data": [{"news_id": 789, "title": "...", "similarity": 0.85}, ...],
              "request_id": "req-abc-2"
            }
            ```
    5.  **컨텍스트 통합 및 생성:** 컨트롤러는 MCP를 통해 수신된 두 종류의 데이터(수익률 시계열, 뉴스 기사 목록)를 확인하고, 이를 하나의 통합된 컨텍스트로 구성하여 LLM에 전달합니다. LLM은 이를 바탕으로 종합적인 답변을 생성합니다.

## 🔧 주요 기술 컴포넌트

### Vector Database MCP 통합
- **통합 인터페이스**: 모든 벡터 데이터베이스(FAISS, Pinecone, Weaviate, Chroma 등)를 표준 MCP 프로토콜로 통합
- **확장 가능한 아키텍처**: 코어 시스템 변경 없이 MCP 서버로 연결 가능
- **표준 작업**: 일관된 MCP 인터페이스를 통한 검색, 추가, 삭제, 업데이트, 검증 작업
- **성능 최적화**: 고밀도 벡터를 위한 효율적인 유사도 검색 및 클러스터링
- **엔터프라이즈 대응**: 분산 벡터 저장소로 대규모 데이터셋 지원

### LangGraph
- 에이전트 RAG 시스템의 복잡한 워크플로우를 관리하는 프레임워크
- 중앙 조정자 역할을 하며 RAG 시스템의 제어 흐름 결정
- 피드백 루프와 에이전트 동작 지원
- 검색 컴포넌트, 메모리 시스템, 언어 생성 모듈 간의 연결 역할

## 📋 개발 로드맵

### 다음 단계 (Tier 2-5)

#### Tier 2 (핵심 컴포넌트)
- **Agent D**: MCP 어댑터 시스템 완성
- **Agent E**: 등록 저장소 시스템 구현
- **Agent F**: 종합적인 테스트 프레임워크

#### Tier 3 (통합 시스템)
- **Agent G**: LangGraph 기반 중앙 컨트롤러

#### Tier 4 (인터페이스)
- **Agent H**: FastAPI 백엔드 구현
- **Agent I**: Streamlit 웹 인터페이스

#### Tier 5 (완성)
- **Agent J**: 예제 및 데모 구현
- **Agent K**: 배포 및 운영 시스템

### 🤝 기여 방법

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 오픈

### 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

### 🔗 관련 링크

- [LangGraph 문서](https://python.langchain.com/docs/langgraph/)
- [MCP 표준](https://modelcontextprotocol.io/)
- [프로젝트 문서](/docs/index.md)
- [개발 가이드](/TODOs.md)
