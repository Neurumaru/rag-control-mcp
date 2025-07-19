# MCP-RAG-Control 컨트롤러 설계

## 1. 컨트롤러 (controller/controller.py)

```python
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from mcp_rag_control.adapters.base_adapter import MCPRequest, MCPResponse
from mcp_rag_control.models.module import Module
from mcp_rag_control.models.pipeline import Pipeline, PipelineModuleDetail
from mcp_rag_control.registry.module_registry import ModuleRegistry
from mcp_rag_control.registry.pipeline_registry import PipelineRegistry


class ControllerState(BaseModel):
    """컨트롤러 상태 모델"""
    pipeline_id: str = Field(..., description="파이프라인 ID")
    query: str = Field(..., description="사용자 질문")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="실행 매개변수")
    context: Dict[str, Any] = Field(default_factory=dict, description="실행 컨텍스트")
    current_step: int = Field(0, description="현재 실행 단계")
    error: Optional[str] = Field(None, description="오류 메시지")
    completed: bool = Field(False, description="완료 여부")


class Controller:
    """LangGraph 기반 컨트롤러"""
    
    def __init__(self, module_registry: ModuleRegistry, pipeline_registry: PipelineRegistry):
        """
        컨트롤러 초기화
        
        Args:
            module_registry: 모듈 등록 저장소
            pipeline_registry: 파이프라인 등록 저장소
        """
        self.module_registry = module_registry
        self.pipeline_registry = pipeline_registry
        
        # 모듈 어댑터 인스턴스 캐시
        self._adapters: Dict[str, Any] = {}
        
        # LangGraph 워크플로우 그래프 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        LangGraph 워크플로우 그래프 구성
        
        Returns:
            StateGraph 인스턴스
        """
        # 상태 그래프 생성
        graph = StateGraph(ControllerState)
        
        # 노드 추가
        graph.add_node("initialize", self._initialize)
        graph.add_node("execute_step", self._execute_step)
        graph.add_node("finalize", self._finalize)
        
        # 엣지 설정
        graph.set_entry_point("initialize")
        
        # initialize -> execute_step 또는 finalize
        graph.add_conditional_edges(
            "initialize",
            self._check_initialization,
            {
                "execute": "execute_step",
                "error": "finalize"
            }
        )
        
        # execute_step -> execute_step 또는 finalize
        graph.add_conditional_edges(
            "execute_step",
            self._check_execution,
            {
                "continue": "execute_step",
                "complete": "finalize",
                "error": "finalize"
            }
        )
        
        # 종료 노드 설정
        graph.add_edge("finalize", None)
        
        return graph.compile()
    
    async def execute_pipeline(self, pipeline_id: str, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            pipeline_id: 파이프라인 ID
            query: 사용자 질문
            parameters: 실행 매개변수
            
        Returns:
            실행 결과
        """
        # 초기 상태 설정
        initial_state = ControllerState(
            pipeline_id=pipeline_id,
            query=query,
            parameters=parameters,
            context={},
            current_step=0,
            error=None,
            completed=False
        )
        
        # 그래프 실행
        try:
            for event, state in self.graph.stream(initial_state):
                pass
            
            final_state = state
            
            # 실행 결과 반환
            if final_state.error:
                return {
                    "status": "error",
                    "error": final_state.error
                }
            else:
                return {
                    "status": "success",
                    "result": final_state.context.get("result", {})
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_initialization(self, state: ControllerState) -> str:
        """
        초기화 상태 확인
        
        Args:
            state: 컨트롤러 상태
            
        Returns:
            다음 단계 (execute 또는 error)
        """
        if state.error:
            return "error"
        else:
            return "execute"
    
    def _check_execution(self, state: ControllerState) -> str:
        """
        실행 상태 확인
        
        Args:
            state: 컨트롤러 상태
            
        Returns:
            다음 단계 (continue, complete 또는 error)
        """
        if state.error:
            return "error"
        elif state.completed:
            return "complete"
        else:
            return "continue"
    
    async def _initialize(self, state: ControllerState) -> ControllerState:
        """
        파이프라인 초기화
        
        Args:
            state: 컨트롤러 상태
            
        Returns:
            업데이트된 상태
        """
        try:
            # 파이프라인 정보 조회
            pipeline = await self.pipeline_registry.get_pipeline(state.pipeline_id)
            if not pipeline:
                return state.model_copy(update={"error": f"파이프라인을 찾을 수 없음: {state.pipeline_id}"})
            
            # 컨텍스트 초기화
            state.context["pipeline"] = pipeline.dict()
            state.context["modules"] = []
            state.context["query"] = state.query
            state.context["start_time"] = asyncio.get_event_loop().time()
            
            # 모듈 정보 수집
            modules = pipeline.modules
            modules_sorted = sorted(modules, key=lambda m: m.step)
            
            for module_detail in modules_sorted:
                module = await self.module_registry.get_module(module_detail.module_id)
                if not module:
                    return state.model_copy(update={
                        "error": f"모듈을 찾을 수 없음: {module_detail.module_id}"
                    })
                
                state.context["modules"].append({
                    "step": module_detail.step,
                    "module_id": module_detail.module_id,
                    "module_type": module.type,
                    "config": module_detail.config,
                    "module_config": module.config
                })
            
            return state
        except Exception as e:
            return state.model_copy(update={"error": f"초기화 오류: {str(e)}"})
    
    async def _execute_step(self, state: ControllerState) -> ControllerState:
        """
        파이프라인 단계 실행
        
        Args:
            state: 컨트롤러 상태
            
        Returns:
            업데이트된 상태
        """
        try:
            # 모듈 목록 확인
            modules = state.context.get("modules", [])
            
            # 현재 단계 증가
            current_step = state.current_step + 1
            
            # 현재 단계의 모듈 찾기
            current_module = next((m for m in modules if m["step"] == current_step), None)
            
            # 모든 단계 완료 확인
            if not current_module:
                # 실행 결과 생성
                answer = state.context.get("answer", "")
                sources = state.context.get("sources", [])
                
                end_time = asyncio.get_event_loop().time()
                processing_time = end_time - state.context.get("start_time", end_time)
                
                result = {
                    "answer": answer,
                    "sources": sources,
                    "metadata": {
                        "processing_time": processing_time,
                        "pipeline_id": state.pipeline_id
                    }
                }
                
                # 토큰 사용량 정보가 있으면 포함
                if "token_usage" in state.context:
                    result["metadata"]["token_usage"] = state.context["token_usage"]
                
                return state.model_copy(
                    update={
                        "current_step": current_step,
                        "completed": True,
                        "context": {**state.context, "result": result}
                    }
                )
            
            # 모듈 어댑터 인스턴스 가져오기
            module_id = current_module["module_id"]
            module_type = current_module["module_type"]
            
            adapter = await self._get_adapter(
                module_id=module_id,
                module_type=module_type,
                module_config=current_module["module_config"]
            )
            
            if not adapter:
                return state.model_copy(update={
                    "error": f"어댑터 생성 실패: {module_id} ({module_type})"
                })
            
            # 모듈 설정 가져오기
            module_config = current_module["config"]
            
            # 요청 매개변수 구성
            request_params = self._build_request_params(
                state=state,
                module_config=module_config
            )
            
            # 요청 생성
            request = MCPRequest(
                source_id=module_id,
                operation=module_config.get("operation", "query"),
                params=request_params,
                request_id=str(uuid.uuid4())
            )
            
            # 요청 실행
            response = await adapter.process_request(request)
            
            # 응답 처리
            if response.status == "error":
                return state.model_copy(update={
                    "error": f"모듈 실행 오류 (단계 {current_step}): {response.error}"
                })
            
            # 결과 컨텍스트에 저장
            output_field = module_config.get("output_field", f"step_{current_step}_output")
            
            return state.model_copy(
                update={
                    "current_step": current_step,
                    "context": {**state.context, output_field: response.data}
                }
            )
        except Exception as e:
            return state.model_copy(update={"error": f"단계 실행 오류: {str(e)}"})
    
    async def _finalize(self, state: ControllerState) -> ControllerState:
        """
        파이프라인 실행 완료
        
        Args:
            state: 컨트롤러 상태
            
        Returns:
            업데이트된 상태
        """
        # 어댑터 연결 해제
        for adapter in self._adapters.values():
            try:
                await adapter.disconnect()
            except:
                pass
        
        # 완료 표시
        return state.model_copy(update={"completed": True})
    
    async def _get_adapter(self, module_id: str, module_type: str, module_config: Dict[str, Any]) -> Optional[Any]:
        """
        모듈 어댑터 인스턴스 가져오기
        
        Args:
            module_id: 모듈 ID
            module_type: 모듈 유형
            module_config: 모듈 설정
            
        Returns:
            어댑터 인스턴스
        """
        # 캐시된 어댑터 확인
        if module_id in self._adapters:
            return self._adapters[module_id]
        
        # 어댑터 생성
        from mcp_rag_control.adapters.base_adapter import AdapterRegistry
        
        adapter_type = module_config.get("adapter_type", module_type)
        adapter = AdapterRegistry.create_adapter(adapter_type, module_id, module_config)
        
        if not adapter:
            return None
        
        # 어댑터 연결
        connected = await adapter.connect()
        if not connected:
            return None
        
        # 어댑터 캐싱
        self._adapters[module_id] = adapter
        
        return adapter
    
    def _build_request_params(self, state: ControllerState, module_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        요청 매개변수 구성
        
        Args:
            state: 컨트롤러 상태
            module_config: 모듈 설정
            
        Returns:
            요청 매개변수
        """
        # 기본 매개변수
        params = {
            "query": state.query,
            **state.parameters
        }
        
        # 입력 필드 처리
        input_field = module_config.get("input_field")
        if input_field and input_field in state.context:
            params["input"] = state.context[input_field]
        
        # 모듈 설정의 params 필드가 있으면 병합
        if "params" in module_config:
            params.update(module_config["params"])
        
        return params


class AsyncController:
    """비동기 컨트롤러 래퍼"""
    
    def __init__(self, controller: Controller):
        """
        비동기 컨트롤러 래퍼 초기화
        
        Args:
            controller: 컨트롤러 인스턴스
        """
        self.controller = controller
        self._jobs: Dict[str, Dict[str, Any]] = {}
    
    async def start_job(self, pipeline_id: str, query: str, parameters: Dict[str, Any], callback_url: Optional[str] = None) -> str:
        """
        비동기 작업 시작
        
        Args:
            pipeline_id: 파이프라인 ID
            query: 사용자 질문
            parameters: 실행 매개변수
            callback_url: 콜백 URL
            
        Returns:
            작업 ID
        """
        job_id = str(uuid.uuid4())
        
        # 작업 정보 저장
        self._jobs[job_id] = {
            "pipeline_id": pipeline_id,
            "query": query,
            "parameters": parameters,
            "callback_url": callback_url,
            "status": "pending",
            "created_at": asyncio.get_event_loop().time(),
            "updated_at": asyncio.get_event_loop().time(),
            "result": None,
            "error": None
        }
        
        # 작업 실행 (비동기)
        asyncio.create_task(self._execute_job(job_id))
        
        return job_id
    
    async def _execute_job(self, job_id: str) -> None:
        """
        비동기 작업 실행
        
        Args:
            job_id: 작업 ID
        """
        job = self._jobs.get(job_id)
        if not job:
            return
        
        # 상태 업데이트
        job["status"] = "processing"
        job["updated_at"] = asyncio.get_event_loop().time()
        
        try:
            # 파이프라인 실행
            result = await self.controller.execute_pipeline(
                pipeline_id=job["pipeline_id"],
                query=job["query"],
                parameters=job["parameters"]
            )
            
            # 실행 결과 저장
            if result["status"] == "success":
                job["status"] = "completed"
                job["result"] = result["result"]
            else:
                job["status"] = "failed"
                job["error"] = result.get("error", "알 수 없는 오류")
        except Exception as e:
            # 오류 발생시 상태 업데이트
            job["status"] = "failed"
            job["error"] = str(e)
        
        # 업데이트 시각 갱신
        job["updated_at"] = asyncio.get_event_loop().time()
        
        # 콜백 처리
        if job["callback_url"]:
            try:
                await self._send_callback(job_id, job)
            except:
                pass
    
    async def _send_callback(self, job_id: str, job: Dict[str, Any]) -> None:
        """
        콜백 URL로 결과 전송
        
        Args:
            job_id: 작업 ID
            job: 작업 정보
        """
        import aiohttp
        
        callback_url = job["callback_url"]
        
        # 전송할 데이터 준비
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "result": job["result"],
            "error": job["error"]
        }
        
        # HTTP 요청 전송
        async with aiohttp.ClientSession() as session:
            await session.post(callback_url, json=payload)
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        작업 상태 조회
        
        Args:
            job_id: 작업 ID
            
        Returns:
            작업 상태 정보
        """
        job = self._jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "result": job["result"] if job["status"] == "completed" else None,
            "error": job["error"] if job["status"] == "failed" else None
        }
    
    def cleanup_jobs(self, max_age_seconds: int = 3600) -> int:
        """
        오래된 작업 정리
        
        Args:
            max_age_seconds: 최대 보존 기간 (초)
            
        Returns:
            정리된 작업 수
        """
        current_time = asyncio.get_event_loop().time()
        to_remove = []
        
        for job_id, job in self._jobs.items():
            job_age = current_time - job["created_at"]
            if job_age > max_age_seconds:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self._jobs[job_id]
        
        return len(to_remove)
```