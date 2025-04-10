# mcp-rag-control 프로젝트 상세 설명

## 프로젝트 개요
mcp-rag-control은 에이전트 기반 RAG(Retrieval-Augmented Generation) 시스템을 위한 맵 기반 제어 아키텍처입니다. 이 시스템은 복잡한 정보 검색과 생성 프로세스를 효율적으로 관리하고 제어하기 위해 설계되었습니다.

## 주요 기술 용어 설명

### RAG (Retrieval-Augmented Generation)
- 기존 정보 검색과 생성형 언어 모델을 결합한 하이브리드 패러다임
- 외부 지식 베이스에서 관련 정보를 검색하여 LLM의 응답을 보강함
- 구식 정보, 환각 현상(hallucination), 도메인 특화 지식 부족 등의 문제를 해결
- 검색(Retrieval), 증강(Augmentation), 생성(Generation) 3단계로 작동

### MCP (Model Context Protocol)
- LLM 애플리케이션과 다양한 외부 데이터 소스를 연결하는 표준화된 프로토콜
- RAG 시스템에서 생성 모델이 사용하는 맥락 정보를 관리하고 전달
- 동적이고 양방향 맥락 교환을 가능하게 함
- 다양한 데이터 소스 간의 상호 운용성 제공

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