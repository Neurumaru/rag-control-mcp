# mcp-rag-control 프로젝트 상세 설명

## 프로젝트 개요
mcp-rag-control은 에이전트 기반 RAG(Retrieval-Augmented Generation) 시스템을 위한 맵 기반 제어 아키텍처입니다. 이 시스템은 복잡한 정보 검색과 생성 프로세스를 효율적으로 관리하고 제어하기 위해 설계되었습니다.

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
    5.  **답변 생성 (Generation):** 구성된 컨텍스트와 원본 질문을 LLM(예: GPT-4)에 전달합니다. LLM은 제공된 최신 정보를 바탕으로 정확하고 상세한 답변을 생성합니다. (예: "최근 출시된 상품으로는 AI 기반 금리 조정 기능이 있는 '스마트 예금 알파'와 글로벌 채권에 분산 투자하는 '글로벌 채권 펀드 플러스'가 있습니다...")

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

## 아키텍처 구성

아키텍처는 5개의 계층으로 구성되어 있습니다:

### 1. 사용자 계층 (User Layer)
- 사용자 인터페이스 모듈 포함

### 2. MCP 인터페이스 계층 (MCP Interface Layer)
- API 엔드포인트 제공: `/modules`, `/pipelines`, `/config`, `/status`, `/metrics`, `/execute`
- 시스템과 외부 사용자 간의 인터페이스 역할

### 3. 컨트롤러 계층 (Controller Layer)
- 오케스트레이션 엔진 (Orchestration Engine)
- LangGraph 컨트롤러: 복잡한 에이전트 RAG 워크플로우 조정
- LlamaIndex 컨트롤러: 데이터 수집 및 인덱싱 관리
- 모니터링 모듈: 시스템 상태 및 성능 모니터링

### 4. 파이프라인 계층 (Pipeline Layer)
- RAG 파이프라인: 검색 모듈과 생성 모듈 포함
- 에이전트 파이프라인: MCP 에이전트 모듈 포함
- 사용자 정의 파이프라인: 커스텀 모듈 포함

### 5. 외부 MCP 계층 (External MCP Layer)
- 검색 MCP: FAISS 서버(벡터 검색), ElasticSearch(전문 검색)
- 생성 MCP: OpenAI GPT, HuggingFace 모델
- 지식 MCP: Neo4j(그래프 데이터베이스), Apache Jena(시멘틱 웹 프레임워크)

## 아키텍처 다이어그램

```mermaid
%%{init: {'theme': 'dark'}}%%
graph TB
    %% Define layers using subgraphs
    subgraph Layer1[User Layer]
        UI[User Interface Module]
    end

    subgraph Layer2[MCP Interface Layer]
        subgraph API_Endpoints[API Endpoints]
            Modules["/modules"]
            Pipelines["/pipelines"]
            Config["/config"]
            Status["/status"]
            Metrics["/metrics"]
            Execute["/execute"]
        end
    end

    subgraph Layer3[Controller Layer]
        OE[Orchestration Engine]
        LG[LangGraph Controller]
        LI[LlamaIndex Controller]
        MM[Monitoring Module]
    end

    subgraph Layer4[Pipeline Layer]
        subgraph RAG_Pipeline[RAG Pipeline]
            RM[Retrieval Module]
            GM[Generation Module]
        end
        
        subgraph Agent_Pipeline[Agent Pipeline]
            AM[MCP Agent Module]
        end
        
        subgraph Custom_Pipeline[Custom Pipeline]
            CM[Custom Module 1]
            CM2[Custom Module 2]
        end
    end

    subgraph Layer5[External MCP Layer]
        subgraph Retrieval_MCPs[Retrieval MCPs]
            FAISS[FAISS Server]
            ES[ElasticSearch]
        end
        
        subgraph Generation_MCPs[Generation MCPs]
            GPT[OpenAI GPT]
            HF[HuggingFace]
        end
        
        subgraph Knowledge_MCPs[Knowledge MCPs]
            NEO[Neo4j]
            JENA[Apache Jena]
        end
    end

    %% User Layer Interactions
    UI --> Modules
    UI --> Pipelines
    UI --> Status
    UI --> Execute

    %% API Endpoint Interactions
    Modules --> OE
    Pipelines --> OE
    Execute --> OE
    Status --> MM
    Metrics --> MM
    Config --> OE

    %% Controller Layer Interactions
    OE --> LG
    OE --> LI
    LG --> RAG_Pipeline
    LG --> Agent_Pipeline
    LI --> Custom_Pipeline

    %% Pipeline Control
    RM --> GM
    CM --> CM2
    
    %% External MCP Interactions
    RM --> NEO
    GM --> GPT
    AM --> FAISS
    AM --> GPT
    AM --> NEO

    %% Styling
    classDef primary fill:#2D4059,stroke:#EA5455,stroke-width:2px,color:#fff
    classDef secondary fill:#222831,stroke:#00ADB5,stroke-width:2px,color:#fff
    classDef storage fill:#393E46,stroke:#EEEEEE,stroke-width:2px,color:#fff
    classDef mcp fill:#2D4059,stroke:#00ADB5,stroke-width:2px,color:#fff
    classDef endpoint fill:#EA5455,stroke:#2D4059,stroke-width:2px,color:#fff
    
    class OE primary
    class RM,GM,UI,MM,LG,LI,AM,CM,CM2 secondary
    class FAISS,ES,GPT,HF,NEO,JENA mcp
    class Modules,Pipelines,Config,Status,Metrics,Execute endpoint

    %% Layer Styling
    style Layer1 fill:#1A1A1A,stroke:#EA5455,color:#fff
    style Layer2 fill:#1A1A1A,stroke:#00ADB5,color:#fff
    style Layer3 fill:#1A1A1A,stroke:#EEEEEE,color:#fff
    style Layer4 fill:#1A1A1A,stroke:#EA5455,color:#fff
    style Layer5 fill:#1A1A1A,stroke:#00ADB5,color:#fff
    
    %% Subgraph Styling
    style API_Endpoints fill:#2D4059,stroke:#EA5455,color:#fff
    style RAG_Pipeline fill:#222831,stroke:#00ADB5,color:#fff
    style Agent_Pipeline fill:#222831,stroke:#00ADB5,color:#fff
    style Custom_Pipeline fill:#222831,stroke:#00ADB5,color:#fff
    style Retrieval_MCPs fill:#2D4059,stroke:#00ADB5,color:#fff
    style Generation_MCPs fill:#2D4059,stroke:#00ADB5,color:#fff
    style Knowledge_MCPs fill:#2D4059,stroke:#00ADB5,color:#fff
```

**다이어그램 범례:**
-   **사각형 (흰색 테두리):** 주요 컨트롤러 및 모듈 (`secondary` 클래스)
-   **사각형 (빨간색 테두리):** API 엔드포인트 (`endpoint` 클래스)
-   **사각형 (파란색 테두리):** 외부 MCP 컴포넌트 (`mcp` 클래스)
-   **사각형 (굵은 빨간색 테두리):** 오케스트레이션 엔진 (`primary` 클래스)
-   **회색 배경:** 시스템 아키텍처 계층 구분
-   **진한 파란색/검은색 배경:** 계층 내 논리적 컴포넌트 그룹 구분

## 주요 기술 컴포넌트

### FAISS (Facebook AI Similarity Search)
- 고밀도 벡터의 효율적인 유사성 검색 및 클러스터링을 위한 오픈소스 라이브러리
- RAG에서 문서나 텍스트 조각의 벡터 임베딩을 저장하는 벡터 저장소로 사용
- 대규모 데이터셋을 처리할 수 있도록 설계됨

### ElasticSearch
- Apache Lucene 기반의 강력한 검색 및 분석 엔진
- 키워드 매칭과 의미적 유사성을 기반으로 문서 검색
- 전통적인 전문 검색과 벡터 기반 검색을 결합한 하이브리드 검색 지원

### LangGraph
- 에이전트 RAG 시스템의 복잡한 워크플로우를 관리하는 프레임워크
- 중앙 조정자 역할을 하며 RAG 시스템의 제어 흐름 결정
- 피드백 루프와 에이전트 동작 지원
- 검색 컴포넌트, 메모리 시스템, 언어 생성 모듈 간의 연결 역할

### LlamaIndex
- 개인 또는 기업 데이터 소스를 대규모 언어 모델에 연결하는 오픈소스 데이터 프레임워크
- 다양한 데이터를 인덱싱 가능한 형식으로 변환
- 모듈형 설계로 다양한 구성 요소(벡터 데이터베이스, 임베딩 모델 등) 교체 용이
- LLM에 관련 맥락을 제공하기 위한 고수준 API 제공

### Neo4j
- 복잡한 관계 네트워크를 저장하고 쿼리하는 인기 있는 그래프 데이터베이스
- 지식 그래프와 같은 구조화된 상호 연결 데이터 관리
- 독립적인 문서뿐만 아니라 엔티티 간의 관계도 이해하고 활용 가능
- 지속적인 메모리 소스 역할

### Apache Jena
- 시맨틱 웹 및 연결 데이터 애플리케이션을 구축하기 위한 Java 프레임워크
- RDF 형식의 데이터 저장 및 SPARQL 쿼리 실행 지원
- 미리 정의된 온톨로지를 사용한 추론 지원
- RAG 시스템에 구조화된 쿼리 가능한 지식 베이스 제공

## 맵 기반 제어의 이점
이 아키텍처는 RAG 시스템에서 검색, 추론, 생성 프로세스 전체를 매핑하고 관리하는 프레임워크를 제공합니다. 특히 다양한 데이터 소스에 걸쳐 복잡한 쿼리를 처리하는 여러 자율 에이전트가 협업하는 에이전트 RAG에서 중요합니다.

이러한 아키텍처는 단순한 RAG 시스템에서 인간과 유사한 추론 능력을 갖춘 보다 지능적이고 에이전트 기반 시스템으로의 진화를 나타냅니다.

## 컴포넌트 상호작용 흐름 (예시)

1.  **사용자 요청:** 사용자가 UI를 통해 질문을 제출하면 (`/execute` 엔드포인트 호출).
2.  **오케스트레이션:** 오케스트레이션 엔진(OE)이 요청을 받아 LangGraph 컨트롤러(LG) 또는 LlamaIndex 컨트롤러(LI)에 전달.
3.  **파이프라인 실행:**
    *   **LangGraph (에이전트 RAG):** LG는 정의된 워크플로우에 따라 에이전트 파이프라인(AM)을 실행. AM은 MCP를 통해 외부 검색 MCP(FAISS 등)와 지식 MCP(Neo4j 등)에서 정보를 검색하고, 생성 MCP(GPT 등)를 호출하여 응답 초안 생성. 이 과정에서 여러 차례의 검색-생성 사이클 또는 에이전트 간 협업이 발생할 수 있음.
    *   **LlamaIndex (데이터 처리):** LI는 사용자 정의 파이프라인을 실행하여 데이터 인덱싱 또는 특정 데이터 소스 쿼리 등의 작업을 수행.
    *   **기본 RAG:** LG는 RAG 파이프라인을 실행. 검색 모듈(RM)이 외부 검색 MCP(Neo4j 등)에서 관련 정보를 찾고, 생성 모듈(GM)이 검색된 정보와 함께 외부 생성 MCP(GPT 등)를 호출하여 최종 응답 생성.
4.  **응답 반환:** 생성된 최종 응답은 OE를 거쳐 API 인터페이스를 통해 사용자에게 전달됨.
5.  **모니터링:** 모니터링 모듈(MM)은 시스템 상태(`/status`) 및 성능 지표(`/metrics`)를 지속적으로 수집하고 제공.

## 오류 처리 및 보안

### 오류 처리 전략
-   **모듈 수준:** 각 모듈(검색, 생성, 에이전트 등) 내에서 발생 가능한 예외(예: 외부 API 연결 실패, 데이터 형식 오류)를 처리하고 로깅합니다.
-   **파이프라인 수준:** 파이프라인 실행 중 특정 단계에서 오류 발생 시, 미리 정의된 재시도 로직 또는 대체 경로를 수행합니다. 실패 시 오류 정보를 컨트롤러에 반환합니다.
-   **컨트롤러 수준:** 심각한 오류 발생 시, 사용자에게 적절한 오류 메시지를 반환하고 시스템 관리자에게 알림을 보냅니다. 오케스트레이션 엔진은 실패한 워크플로우의 상태를 기록하고 복구를 시도할 수 있습니다.

### 보안 고려 사항
-   **API 접근 제어:** MCP 인터페이스 계층의 API 엔드포인트는 인증 및 인가 메커니즘(예: API 키, OAuth)을 통해 보호됩니다.
-   **외부 시스템 연동:** 외부 MCP(OpenAI, FAISS 서버 등)와의 통신은 암호화(HTTPS 등)되며, API 키와 같은 민감 정보는 안전하게 관리됩니다 (예: 환경 변수, 시크릿 관리 도구 사용).
-   **데이터 프라이버시:** 사용자 데이터 및 외부 지식 베이스 접근 시 개인 정보 보호 및 데이터 접근 정책을 준수합니다. 민감 데이터는 필요시 마스킹 처리합니다.
-   **입력 값 검증:** 사용자 입력 및 외부 시스템으로부터 받는 데이터는 악의적인 코드 삽입(Injection) 등을 방지하기 위해 철저히 검증됩니다.