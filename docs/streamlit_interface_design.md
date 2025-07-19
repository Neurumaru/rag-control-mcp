# MCP-RAG-Control Streamlit 웹 인터페이스 설계

## 1. 개요

Streamlit을 사용하여 MCP-RAG-Control 시스템의 사용자 친화적인 웹 인터페이스를 구현합니다. 이 인터페이스를 통해 사용자는 GUI를 통해 모듈 관리, 파이프라인 구성, RAG 시스템 테스트를 수행할 수 있습니다.

## 2. 인터페이스 구조

### 2.1 메인 네비게이션
- **홈 대시보드**: 시스템 상태 및 통계
- **모듈 관리**: 모듈 등록, 수정, 삭제, 조회
- **파이프라인 관리**: 파이프라인 생성, 수정, 삭제, 조회
- **RAG 테스트**: 파이프라인 실행 및 결과 확인
- **시스템 설정**: 전역 설정 관리

### 2.2 페이지별 상세 설계

#### 2.2.1 홈 대시보드 (pages/home.py)
```python
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def render_dashboard():
    st.title("🏠 MCP-RAG-Control 대시보드")
    
    # 시스템 상태 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("등록된 모듈", "12", "2")
    
    with col2:
        st.metric("활성 파이프라인", "5", "1")
    
    with col3:
        st.metric("일일 쿼리", "247", "15")
    
    with col4:
        st.metric("평균 응답시간", "1.2s", "-0.3s")
    
    # 최근 활동 차트
    st.subheader("📊 최근 활동")
    
    # 샘플 데이터 (실제로는 API에서 가져옴)
    dates = [datetime.now() - timedelta(days=i) for i in range(7, 0, -1)]
    queries = [45, 52, 38, 61, 74, 68, 55]
    
    df = pd.DataFrame({"날짜": dates, "쿼리 수": queries})
    fig = px.line(df, x="날짜", y="쿼리 수", title="일별 쿼리 수")
    st.plotly_chart(fig, use_container_width=True)
    
    # 최근 파이프라인 실행
    st.subheader("🔄 최근 파이프라인 실행")
    recent_executions = pd.DataFrame({
        "파이프라인": ["기본 RAG", "고급 검색", "문서 요약", "Q&A"],
        "상태": ["성공", "성공", "실패", "성공"],
        "실행시간": ["1.2s", "2.1s", "0.8s", "1.5s"],
        "시각": ["방금 전", "2분 전", "5분 전", "10분 전"]
    })
    st.dataframe(recent_executions, use_container_width=True)
```

#### 2.2.2 모듈 관리 (pages/modules.py)
```python
import streamlit as st
import requests
import json
from typing import Dict, Any

def render_module_management():
    st.title("🔧 모듈 관리")
    
    # 탭으로 기능 구분
    tab1, tab2, tab3 = st.tabs(["모듈 목록", "모듈 등록", "모듈 상세"])
    
    with tab1:
        render_module_list()
    
    with tab2:
        render_module_registration()
    
    with tab3:
        render_module_details()

def render_module_list():
    st.subheader("📋 등록된 모듈 목록")
    
    # 필터 옵션
    col1, col2 = st.columns([3, 1])
    with col1:
        module_type_filter = st.selectbox(
            "모듈 유형 필터",
            ["전체", "data_source", "vector_store", "embedding_model", "llm", "retriever", "reranker", "custom"]
        )
    
    with col2:
        if st.button("🔄 새로고침"):
            st.rerun()
    
    # 모듈 목록 가져오기
    try:
        modules = fetch_modules(module_type_filter)
        
        if modules:
            for module in modules:
                with st.expander(f"{module['name']} ({module['type']})"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**ID**: {module['module_id']}")
                        st.write(f"**설명**: {module.get('description', 'N/A')}")
                    
                    with col2:
                        st.write(f"**유형**: {module['type']}")
                        st.write(f"**생성일**: {module.get('created_at', 'N/A')}")
                    
                    with col3:
                        if st.button("수정", key=f"edit_{module['module_id']}"):
                            st.session_state.selected_module = module['module_id']
                            st.switch_page("pages/modules.py")
                        
                        if st.button("삭제", key=f"delete_{module['module_id']}"):
                            if delete_module(module['module_id']):
                                st.success("모듈이 삭제되었습니다.")
                                st.rerun()
        else:
            st.info("등록된 모듈이 없습니다.")
    
    except Exception as e:
        st.error(f"모듈 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")

def render_module_registration():
    st.subheader("➕ 새 모듈 등록")
    
    with st.form("module_registration"):
        # 기본 정보
        module_id = st.text_input("모듈 ID", placeholder="unique_module_id")
        module_name = st.text_input("모듈 이름", placeholder="모듈 이름을 입력하세요")
        module_type = st.selectbox(
            "모듈 유형",
            ["data_source", "vector_store", "embedding_model", "llm", "retriever", "reranker", "custom"]
        )
        description = st.text_area("설명", placeholder="모듈에 대한 설명을 입력하세요")
        
        # 설정 정보
        st.subheader("설정 정보")
        
        # 모듈 유형에 따른 동적 설정 폼
        config = render_module_config_form(module_type)
        
        submitted = st.form_submit_button("모듈 등록")
        
        if submitted:
            if module_id and module_name:
                module_data = {
                    "module_id": module_id,
                    "name": module_name,
                    "type": module_type,
                    "description": description,
                    "config": config
                }
                
                if register_module(module_data):
                    st.success(f"모듈 '{module_name}'이 성공적으로 등록되었습니다!")
                    st.rerun()
                else:
                    st.error("모듈 등록에 실패했습니다.")
            else:
                st.error("모듈 ID와 이름은 필수 입력 항목입니다.")

def render_module_config_form(module_type: str) -> Dict[str, Any]:
    """모듈 유형에 따른 설정 폼 렌더링"""
    config = {}
    
    if module_type == "vector_store":
        if st.selectbox("벡터 스토어 유형", ["faiss", "chroma"]) == "faiss":
            config["adapter_type"] = "faiss"
            config["dimension"] = st.number_input("벡터 차원", min_value=1, value=768)
        else:
            config["adapter_type"] = "chroma"
            config["collection_name"] = st.text_input("컬렉션 이름", value="default")
    
    elif module_type == "embedding_model":
        config["model_name"] = st.text_input("모델 이름", value="sentence-transformers/all-MiniLM-L6-v2")
        config["device"] = st.selectbox("디바이스", ["cpu", "cuda"])
    
    elif module_type == "llm":
        llm_provider = st.selectbox("LLM 제공자", ["openai", "huggingface", "anthropic"])
        if llm_provider == "openai":
            config["provider"] = "openai"
            config["model"] = st.selectbox("모델", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])
            config["api_key"] = st.text_input("API 키", type="password")
        elif llm_provider == "huggingface":
            config["provider"] = "huggingface"
            config["model"] = st.text_input("모델 이름", value="microsoft/DialoGPT-medium")
            config["api_key"] = st.text_input("API 키 (선택사항)", type="password")
    
    elif module_type == "data_source":
        source_type = st.selectbox("데이터 소스 유형", ["database", "api", "file"])
        if source_type == "database":
            config["source_type"] = "database"
            config["connection_string"] = st.text_input("연결 문자열")
        elif source_type == "api":
            config["source_type"] = "api"
            config["base_url"] = st.text_input("기본 URL")
            config["api_key"] = st.text_input("API 키", type="password")
    
    # 공통 설정
    config["timeout"] = st.number_input("타임아웃 (초)", min_value=1, value=30)
    
    return config

def render_module_details():
    st.subheader("🔍 모듈 상세 정보")
    
    if "selected_module" in st.session_state:
        module_id = st.session_state.selected_module
        module = fetch_module_details(module_id)
        
        if module:
            st.json(module)
        else:
            st.error("모듈 정보를 찾을 수 없습니다.")
    else:
        st.info("모듈을 선택해주세요.")

# API 호출 함수들
def fetch_modules(module_type_filter: str = "전체"):
    """모듈 목록 조회"""
    try:
        url = "http://localhost:8000/api/modules"
        if module_type_filter != "전체":
            url += f"?type={module_type_filter}"
        
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("modules", [])
        return []
    except:
        return []

def register_module(module_data: Dict[str, Any]) -> bool:
    """모듈 등록"""
    try:
        response = requests.post(
            "http://localhost:8000/api/modules",
            json=module_data
        )
        return response.status_code == 200
    except:
        return False

def delete_module(module_id: str) -> bool:
    """모듈 삭제"""
    try:
        response = requests.delete(f"http://localhost:8000/api/modules/{module_id}")
        return response.status_code == 200
    except:
        return False

def fetch_module_details(module_id: str):
    """모듈 상세 정보 조회"""
    try:
        response = requests.get(f"http://localhost:8000/api/modules/{module_id}")
        if response.status_code == 200:
            return response.json().get("module")
        return None
    except:
        return None
```

#### 2.2.3 파이프라인 관리 (pages/pipelines.py)
```python
import streamlit as st
import requests
from typing import Dict, Any, List

def render_pipeline_management():
    st.title("🔗 파이프라인 관리")
    
    tab1, tab2, tab3 = st.tabs(["파이프라인 목록", "파이프라인 생성", "파이프라인 편집"])
    
    with tab1:
        render_pipeline_list()
    
    with tab2:
        render_pipeline_creation()
    
    with tab3:
        render_pipeline_editing()

def render_pipeline_list():
    st.subheader("📋 파이프라인 목록")
    
    pipelines = fetch_pipelines()
    
    if pipelines:
        for pipeline in pipelines:
            with st.expander(f"📊 {pipeline['name']}"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**ID**: {pipeline['pipeline_id']}")
                    st.write(f"**설명**: {pipeline.get('description', 'N/A')}")
                
                with col2:
                    module_count = len(pipeline.get('modules', []))
                    st.write(f"**모듈 수**: {module_count}")
                    st.write(f"**생성일**: {pipeline.get('created_at', 'N/A')}")
                
                with col3:
                    if st.button("실행", key=f"run_{pipeline['pipeline_id']}"):
                        st.session_state.selected_pipeline = pipeline['pipeline_id']
                        st.switch_page("pages/rag_test.py")
                    
                    if st.button("편집", key=f"edit_{pipeline['pipeline_id']}"):
                        st.session_state.editing_pipeline = pipeline['pipeline_id']
                        st.rerun()
                    
                    if st.button("삭제", key=f"delete_{pipeline['pipeline_id']}"):
                        if delete_pipeline(pipeline['pipeline_id']):
                            st.success("파이프라인이 삭제되었습니다.")
                            st.rerun()
    else:
        st.info("등록된 파이프라인이 없습니다.")

def render_pipeline_creation():
    st.subheader("➕ 새 파이프라인 생성")
    
    # 기본 정보
    pipeline_id = st.text_input("파이프라인 ID", placeholder="unique_pipeline_id")
    pipeline_name = st.text_input("파이프라인 이름", placeholder="파이프라인 이름을 입력하세요")
    description = st.text_area("설명", placeholder="파이프라인에 대한 설명을 입력하세요")
    
    st.subheader("🔧 모듈 구성")
    
    # 사용 가능한 모듈 목록 가져오기
    available_modules = fetch_modules()
    
    if not available_modules:
        st.warning("등록된 모듈이 없습니다. 먼저 모듈을 등록해주세요.")
        return
    
    # 파이프라인 모듈 구성
    if "pipeline_modules" not in st.session_state:
        st.session_state.pipeline_modules = []
    
    # 모듈 추가 폼
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_module = st.selectbox(
                "모듈 선택",
                options=[(m['module_id'], f"{m['name']} ({m['type']})") for m in available_modules],
                format_func=lambda x: x[1] if x else "",
                key="module_selector"
            )
        
        with col2:
            step_number = st.number_input(
                "단계 번호",
                min_value=1,
                value=len(st.session_state.pipeline_modules) + 1,
                key="step_selector"
            )
        
        with col3:
            if st.button("➕ 모듈 추가"):
                if selected_module:
                    module_config = {
                        "step": step_number,
                        "module_id": selected_module[0],
                        "module_name": selected_module[1],
                        "config": {}
                    }
                    st.session_state.pipeline_modules.append(module_config)
                    st.rerun()
    
    # 현재 구성된 모듈들 표시
    if st.session_state.pipeline_modules:
        st.subheader("현재 구성")
        
        # 단계별로 정렬
        sorted_modules = sorted(st.session_state.pipeline_modules, key=lambda x: x['step'])
        
        for i, module in enumerate(sorted_modules):
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.write(f"**단계 {module['step']}**")
                
                with col2:
                    st.write(f"{module['module_name']}")
                    
                    # 모듈별 설정
                    with st.expander("설정"):
                        module['config'] = render_module_step_config(module, i)
                
                with col3:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.pipeline_modules.pop(i)
                        st.rerun()
        
        # 파이프라인 생성 버튼
        if st.button("🚀 파이프라인 생성", type="primary"):
            if pipeline_id and pipeline_name:
                pipeline_data = {
                    "pipeline_id": pipeline_id,
                    "name": pipeline_name,
                    "description": description,
                    "modules": [
                        {
                            "step": m['step'],
                            "module_id": m['module_id'],
                            "config": m['config']
                        }
                        for m in sorted_modules
                    ]
                }
                
                if create_pipeline(pipeline_data):
                    st.success(f"파이프라인 '{pipeline_name}'이 성공적으로 생성되었습니다!")
                    st.session_state.pipeline_modules = []
                    st.rerun()
                else:
                    st.error("파이프라인 생성에 실패했습니다.")
            else:
                st.error("파이프라인 ID와 이름은 필수 입력 항목입니다.")

def render_module_step_config(module: Dict[str, Any], index: int) -> Dict[str, Any]:
    """파이프라인 단계별 모듈 설정"""
    config = module.get('config', {})
    
    # 입력 필드
    input_field = st.text_input(
        "입력 필드",
        value=config.get('input_field', ''),
        key=f"input_field_{index}",
        help="이전 단계의 출력 필드명"
    )
    
    # 출력 필드
    output_field = st.text_input(
        "출력 필드",
        value=config.get('output_field', f'step_{module["step"]}_output'),
        key=f"output_field_{index}",
        help="이 단계의 출력 필드명"
    )
    
    # 작업 유형
    operation = st.text_input(
        "작업",
        value=config.get('operation', 'query'),
        key=f"operation_{index}",
        help="모듈에서 수행할 작업"
    )
    
    return {
        "input_field": input_field,
        "output_field": output_field,
        "operation": operation,
        **config
    }

def render_pipeline_editing():
    st.subheader("✏️ 파이프라인 편집")
    
    if "editing_pipeline" in st.session_state:
        pipeline_id = st.session_state.editing_pipeline
        pipeline = fetch_pipeline_details(pipeline_id)
        
        if pipeline:
            st.json(pipeline)
            # 편집 기능 구현
        else:
            st.error("파이프라인 정보를 찾을 수 없습니다.")
    else:
        st.info("편집할 파이프라인을 선택해주세요.")

# API 호출 함수들
def fetch_pipelines():
    """파이프라인 목록 조회"""
    try:
        response = requests.get("http://localhost:8000/api/pipelines")
        if response.status_code == 200:
            return response.json().get("pipelines", [])
        return []
    except:
        return []

def create_pipeline(pipeline_data: Dict[str, Any]) -> bool:
    """파이프라인 생성"""
    try:
        response = requests.post(
            "http://localhost:8000/api/pipelines",
            json=pipeline_data
        )
        return response.status_code == 200
    except:
        return False

def delete_pipeline(pipeline_id: str) -> bool:
    """파이프라인 삭제"""
    try:
        response = requests.delete(f"http://localhost:8000/api/pipelines/{pipeline_id}")
        return response.status_code == 200
    except:
        return False

def fetch_pipeline_details(pipeline_id: str):
    """파이프라인 상세 정보 조회"""
    try:
        response = requests.get(f"http://localhost:8000/api/pipelines/{pipeline_id}")
        if response.status_code == 200:
            return response.json().get("pipeline")
        return None
    except:
        return None
```

#### 2.2.4 RAG 테스트 (pages/rag_test.py)
```python
import streamlit as st
import requests
import time
from typing import Dict, Any

def render_rag_test():
    st.title("🤖 RAG 시스템 테스트")
    
    # 파이프라인 선택
    pipelines = fetch_pipelines()
    
    if not pipelines:
        st.warning("등록된 파이프라인이 없습니다. 먼저 파이프라인을 생성해주세요.")
        return
    
    # 사이드바에서 파이프라인 선택
    with st.sidebar:
        st.header("설정")
        
        selected_pipeline = st.selectbox(
            "파이프라인 선택",
            options=[(p['pipeline_id'], p['name']) for p in pipelines],
            format_func=lambda x: x[1] if x else "",
            index=0 if pipelines else None
        )
        
        if selected_pipeline:
            pipeline_id = selected_pipeline[0]
            
            # 실행 파라미터
            st.subheader("실행 파라미터")
            max_tokens = st.slider("최대 토큰", 100, 4000, 1000)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            
            execution_params = {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
    
    # 메인 영역
    if selected_pipeline:
        st.subheader(f"📊 파이프라인: {selected_pipeline[1]}")
        
        # 질문 입력
        user_query = st.text_area(
            "질문을 입력하세요:",
            placeholder="RAG 시스템에 질문하고 싶은 내용을 입력하세요...",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            sync_execute = st.button("🚀 즉시 실행", type="primary")
        
        with col2:
            async_execute = st.button("⏰ 비동기 실행")
        
        # 즉시 실행
        if sync_execute and user_query:
            execute_pipeline_sync(pipeline_id, user_query, execution_params)
        
        # 비동기 실행
        if async_execute and user_query:
            execute_pipeline_async(pipeline_id, user_query, execution_params)
        
        # 결과 표시 영역
        if "execution_result" in st.session_state:
            display_execution_result(st.session_state.execution_result)
        
        # 실행 히스토리
        display_execution_history()

def execute_pipeline_sync(pipeline_id: str, query: str, params: Dict[str, Any]):
    """동기 파이프라인 실행"""
    with st.spinner("파이프라인을 실행 중입니다..."):
        start_time = time.time()
        
        request_data = {
            "pipeline_id": pipeline_id,
            "query": query,
            "parameters": params
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/execute",
                json=request_data
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result["execution_time"] = execution_time
                st.session_state.execution_result = result
                
                # 히스토리에 추가
                add_to_history(pipeline_id, query, result, "sync")
                
                st.success(f"실행 완료! (소요시간: {execution_time:.2f}초)")
            else:
                st.error(f"실행 실패: {response.text}")
        
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

def execute_pipeline_async(pipeline_id: str, query: str, params: Dict[str, Any]):
    """비동기 파이프라인 실행"""
    request_data = {
        "pipeline_id": pipeline_id,
        "query": query,
        "parameters": params
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/execute/async",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("job_id")
            
            st.info(f"비동기 작업이 시작되었습니다. 작업 ID: {job_id}")
            
            # 작업 상태 모니터링
            monitor_async_job(job_id, pipeline_id, query)
        else:
            st.error(f"비동기 실행 시작 실패: {response.text}")
    
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

def monitor_async_job(job_id: str, pipeline_id: str, query: str):
    """비동기 작업 상태 모니터링"""
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    max_attempts = 60  # 최대 60초 대기
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"http://localhost:8000/api/execute/jobs/{job_id}")
            
            if response.status_code == 200:
                job_status = response.json().get("job", {})
                status = job_status.get("status", "unknown")
                
                if status == "completed":
                    result = job_status.get("result", {})
                    st.session_state.execution_result = {"status": "success", "result": result}
                    add_to_history(pipeline_id, query, result, "async")
                    
                    status_placeholder.success("✅ 실행 완료!")
                    progress_bar.progress(100)
                    break
                
                elif status == "failed":
                    error = job_status.get("error", "Unknown error")
                    status_placeholder.error(f"❌ 실행 실패: {error}")
                    break
                
                elif status == "processing":
                    progress = min((attempt / max_attempts) * 100, 90)
                    progress_bar.progress(int(progress))
                    status_placeholder.info("🔄 처리 중...")
                
                else:  # pending
                    status_placeholder.info("⏳ 대기 중...")
            
            time.sleep(1)
            attempt += 1
        
        except Exception as e:
            st.error(f"상태 확인 중 오류: {str(e)}")
            break
    
    if attempt >= max_attempts:
        status_placeholder.warning("⚠️ 시간 초과: 작업이 완료되지 않았습니다.")

def display_execution_result(result: Dict[str, Any]):
    """실행 결과 표시"""
    st.subheader("📋 실행 결과")
    
    if result.get("status") == "success":
        result_data = result.get("result", {})
        
        # 답변 표시
        answer = result_data.get("answer", "")
        if answer:
            st.markdown("### 💬 생성된 답변")
            st.markdown(f"<div style='padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>{answer}</div>", 
                       unsafe_allow_html=True)
        
        # 소스 정보
        sources = result_data.get("sources", [])
        if sources:
            st.markdown("### 📚 참조 소스")
            for i, source in enumerate(sources, 1):
                with st.expander(f"소스 {i} (관련도: {source.get('relevance_score', 0):.2f})"):
                    st.write(source.get("content", ""))
                    if source.get("metadata"):
                        st.json(source["metadata"])
        
        # 메타데이터
        metadata = result_data.get("metadata", {})
        if metadata:
            st.markdown("### 📊 실행 정보")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                processing_time = metadata.get("processing_time", 0)
                st.metric("처리 시간", f"{processing_time:.2f}초")
            
            with col2:
                if "token_usage" in metadata:
                    total_tokens = metadata["token_usage"].get("total_tokens", 0)
                    st.metric("사용 토큰", total_tokens)
            
            with col3:
                pipeline_id = metadata.get("pipeline_id", "N/A")
                st.write(f"**파이프라인**: {pipeline_id}")
    
    else:
        st.error(f"실행 실패: {result.get('error', 'Unknown error')}")

def add_to_history(pipeline_id: str, query: str, result: Dict[str, Any], execution_type: str):
    """실행 히스토리에 추가"""
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    history_item = {
        "timestamp": time.time(),
        "pipeline_id": pipeline_id,
        "query": query,
        "result": result,
        "execution_type": execution_type
    }
    
    st.session_state.execution_history.insert(0, history_item)
    
    # 최대 10개 항목만 유지
    if len(st.session_state.execution_history) > 10:
        st.session_state.execution_history = st.session_state.execution_history[:10]

def display_execution_history():
    """실행 히스토리 표시"""
    if "execution_history" not in st.session_state:
        return
    
    history = st.session_state.execution_history
    
    if history:
        st.subheader("📜 실행 히스토리")
        
        for i, item in enumerate(history):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["timestamp"]))
            execution_type = "🚀" if item["execution_type"] == "sync" else "⏰"
            
            with st.expander(f"{execution_type} {timestamp} - {item['query'][:50]}..."):
                st.write(f"**파이프라인**: {item['pipeline_id']}")
                st.write(f"**질문**: {item['query']}")
                
                if "answer" in item["result"]:
                    st.write(f"**답변**: {item['result']['answer'][:200]}...")

# API 호출 함수들은 이전과 동일
def fetch_pipelines():
    """파이프라인 목록 조회"""
    try:
        response = requests.get("http://localhost:8000/api/pipelines")
        if response.status_code == 200:
            return response.json().get("pipelines", [])
        return []
    except:
        return []
```

#### 2.2.5 메인 앱 (web_interface/app.py)
```python
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="MCP-RAG-Control",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 네비게이션
def render_sidebar():
    with st.sidebar:
        st.title("🤖 MCP-RAG-Control")
        st.markdown("---")
        
        # 네비게이션 메뉴
        pages = {
            "🏠 홈 대시보드": "home",
            "🔧 모듈 관리": "modules", 
            "🔗 파이프라인 관리": "pipelines",
            "🤖 RAG 테스트": "rag_test",
            "⚙️ 시스템 설정": "settings"
        }
        
        selected_page = st.selectbox(
            "페이지 선택",
            options=list(pages.keys()),
            index=0
        )
        
        st.markdown("---")
        
        # 시스템 상태
        st.subheader("시스템 상태")
        
        # API 서버 상태 확인
        api_status = check_api_status()
        status_color = "🟢" if api_status else "🔴"
        status_text = "온라인" if api_status else "오프라인"
        st.write(f"{status_color} API 서버: {status_text}")
        
        if not api_status:
            st.warning("API 서버가 실행되지 않았습니다. 먼저 백엔드 서버를 시작해주세요.")
            st.code("uvicorn mcp_rag_control.api.app:app --reload")
        
        return pages[selected_page]

def check_api_status() -> bool:
    """API 서버 상태 확인"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    # 사이드바 렌더링
    current_page = render_sidebar()
    
    # 선택된 페이지 렌더링
    if current_page == "home":
        from pages.home import render_dashboard
        render_dashboard()
    
    elif current_page == "modules":
        from pages.modules import render_module_management
        render_module_management()
    
    elif current_page == "pipelines":
        from pages.pipelines import render_pipeline_management
        render_pipeline_management()
    
    elif current_page == "rag_test":
        from pages.rag_test import render_rag_test
        render_rag_test()
    
    elif current_page == "settings":
        render_settings()

def render_settings():
    """시스템 설정 페이지"""
    st.title("⚙️ 시스템 설정")
    
    with st.expander("API 설정"):
        api_host = st.text_input("API 호스트", value="localhost")
        api_port = st.number_input("API 포트", value=8000)
        st.write(f"현재 API 엔드포인트: http://{api_host}:{api_port}")
    
    with st.expander("UI 설정"):
        theme = st.selectbox("테마", ["자동", "라이트", "다크"])
        language = st.selectbox("언어", ["한국어", "English"])
    
    if st.button("설정 저장"):
        st.success("설정이 저장되었습니다.")

if __name__ == "__main__":
    main()
```

## 3. 실행 방법

### 3.1 개발 환경에서 실행
```bash
# 백엔드 API 서버 시작
uvicorn mcp_rag_control.api.app:app --reload --port 8000

# 새 터미널에서 Streamlit 앱 실행
cd web_interface
streamlit run app.py --server.port 8501
```

### 3.2 프로덕션 환경에서 실행
```bash
# Docker Compose 사용
docker-compose up -d
```

## 4. 주요 기능

- **직관적인 모듈 관리**: GUI를 통한 모듈 등록, 수정, 삭제
- **시각적 파이프라인 구성**: 드래그 앤 드롭 방식의 파이프라인 빌더
- **실시간 RAG 테스트**: 즉시 실행 및 비동기 실행 지원
- **상세한 결과 분석**: 소스 추적, 토큰 사용량, 처리 시간 등
- **실행 히스토리**: 과거 실행 결과 및 성능 분석
- **시스템 모니터링**: 대시보드를 통한 시스템 상태 확인

이 설계는 사용자가 코드 없이도 MCP-RAG-Control 시스템을 완전히 활용할 수 있도록 합니다.