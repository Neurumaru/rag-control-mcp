"""Data flow schema definitions for RAG system ModuleTypes."""

from typing import Dict, Optional
from .module import DataType, DataSchema, ModuleType, ModuleCapabilities


class ModuleSchemaRegistry:
    """각 ModuleType에 대한 표준 입출력 스키마 레지스트리"""
    
    # Text Processing Modules
    TEXT_PREPROCESSOR_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.TEXT,
            format="plain_text",
            required_fields=["content"],
            constraints={"min_length": 1, "max_length": 100000}
        ),
        "output": DataSchema(
            data_type=DataType.TEXT,
            format="processed_text",
            required_fields=["content", "metadata"],
            optional_fields=["chunks", "entities"]
        )
    }
    
    QUERY_ANALYZER_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.TEXT,
            format="user_query",
            required_fields=["query"],
            constraints={"max_length": 1000}
        ),
        "output": DataSchema(
            data_type=DataType.STRUCTURED,
            format="query_analysis",
            required_fields=["intent", "entities", "query_type"],
            optional_fields=["confidence", "suggested_filters"]
        )
    }
    
    # Embedding Modules
    EMBEDDING_ENCODER_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.TEXT,
            format="plain_text",
            required_fields=["text"],
            constraints={"max_length": 8192}
        ),
        "output": DataSchema(
            data_type=DataType.EMBEDDINGS,
            format="vector_array",
            required_fields=["embeddings"],
            vector_dimension=1536,  # 기본값, 모델에 따라 변경
            distance_metric="cosine"
        )
    }
    
    # Vector Operations
    VECTOR_STORE_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.EMBEDDINGS,
            format="vector_array",
            required_fields=["embeddings", "metadata"],
            vector_dimension=1536
        ),
        "output": DataSchema(
            data_type=DataType.STRUCTURED,
            format="operation_result",
            required_fields=["success", "id"],
            optional_fields=["error_message"]
        )
    }
    
    SIMILARITY_SEARCH_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.EMBEDDINGS,
            format="query_vector",
            required_fields=["query_embedding"],
            optional_fields=["top_k", "threshold"],
            vector_dimension=1536
        ),
        "output": DataSchema(
            data_type=DataType.VECTORS,
            format="similarity_results",
            required_fields=["results"],
            schema_definition={
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "score": {"type": "number"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            }
        )
    }
    
    # Document Processing
    DOCUMENT_LOADER_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.STRUCTURED,
            format="file_reference",
            required_fields=["file_path"],
            optional_fields=["file_type", "encoding"]
        ),
        "output": DataSchema(
            data_type=DataType.DOCUMENTS,
            format="document_collection",
            required_fields=["documents"],
            schema_definition={
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "metadata": {"type": "object"},
                            "source": {"type": "string"}
                        }
                    }
                }
            }
        )
    }
    
    CONTEXT_BUILDER_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.DOCUMENTS,
            format="document_collection",
            required_fields=["documents", "query"]
        ),
        "output": DataSchema(
            data_type=DataType.CONTEXT,
            format="rag_context",
            required_fields=["context", "sources"],
            optional_fields=["relevance_scores", "token_count"]
        )
    }
    
    # Generation Modules
    LLM_GENERATOR_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.CONTEXT,
            format="rag_context",
            required_fields=["context", "query"]
        ),
        "output": DataSchema(
            data_type=DataType.RESPONSE,
            format="llm_response",
            required_fields=["response"],
            optional_fields=["token_usage", "model_info", "finish_reason"]
        )
    }
    
    # Multimodal Support
    IMAGE_PROCESSOR_SCHEMA = {
        "input": DataSchema(
            data_type=DataType.MULTIMODAL,
            format="image_data",
            required_fields=["image"],
            optional_fields=["format", "quality"]
        ),
        "output": DataSchema(
            data_type=DataType.TEXT,
            format="image_description",
            required_fields=["description"],
            optional_fields=["detected_objects", "text_content"]
        )
    }
    
    @classmethod
    def get_schema_for_type(cls, module_type: ModuleType) -> Optional[Dict[str, DataSchema]]:
        """특정 ModuleType에 대한 스키마 반환"""
        schema_map = {
            ModuleType.TEXT_PREPROCESSOR: cls.TEXT_PREPROCESSOR_SCHEMA,
            ModuleType.QUERY_ANALYZER: cls.QUERY_ANALYZER_SCHEMA,
            ModuleType.EMBEDDING_ENCODER: cls.EMBEDDING_ENCODER_SCHEMA,
            ModuleType.VECTOR_STORE: cls.VECTOR_STORE_SCHEMA,
            ModuleType.SIMILARITY_SEARCH: cls.SIMILARITY_SEARCH_SCHEMA,
            ModuleType.DOCUMENT_LOADER: cls.DOCUMENT_LOADER_SCHEMA,
            ModuleType.CONTEXT_BUILDER: cls.CONTEXT_BUILDER_SCHEMA,
            ModuleType.LLM_GENERATOR: cls.LLM_GENERATOR_SCHEMA,
            ModuleType.IMAGE_PROCESSOR: cls.IMAGE_PROCESSOR_SCHEMA,
        }
        return schema_map.get(module_type)
    
    @classmethod
    def create_module_capabilities(cls, module_type: ModuleType, **kwargs) -> Optional[ModuleCapabilities]:
        """ModuleType에 맞는 기본 ModuleCapabilities 생성"""
        schema = cls.get_schema_for_type(module_type)
        if not schema:
            return None
            
        return ModuleCapabilities(
            input_schema=schema["input"],
            output_schema=schema["output"],
            transformation_type=cls._get_transformation_type(module_type),
            is_stateful=cls._is_stateful(module_type),
            supports_streaming=cls._supports_streaming(module_type),
            supports_batch=cls._supports_batch(module_type),
            **kwargs
        )
    
    @staticmethod
    def _get_transformation_type(module_type: ModuleType) -> str:
        """ModuleType에 따른 변환 타입 반환"""
        type_mapping = {
            ModuleType.TEXT_PREPROCESSOR: "text_processing",
            ModuleType.QUERY_ANALYZER: "semantic_analysis",
            ModuleType.EMBEDDING_ENCODER: "text_to_vector",
            ModuleType.VECTOR_STORE: "vector_storage",
            ModuleType.SIMILARITY_SEARCH: "vector_search",
            ModuleType.DOCUMENT_LOADER: "data_ingestion",
            ModuleType.CONTEXT_BUILDER: "context_aggregation",
            ModuleType.LLM_GENERATOR: "text_generation",
            ModuleType.IMAGE_PROCESSOR: "multimodal_processing",
        }
        return type_mapping.get(module_type, "unknown")
    
    @staticmethod
    def _is_stateful(module_type: ModuleType) -> bool:
        """ModuleType이 상태를 유지하는지 확인"""
        stateful_types = {
            ModuleType.VECTOR_STORE,
            ModuleType.MEMORY_STORE,
            ModuleType.CACHE_MANAGER,
            ModuleType.SESSION_MANAGER
        }
        return module_type in stateful_types
    
    @staticmethod
    def _supports_streaming(module_type: ModuleType) -> bool:
        """ModuleType이 스트리밍을 지원하는지 확인"""
        streaming_types = {
            ModuleType.LLM_GENERATOR,
            ModuleType.TEXT_PREPROCESSOR,
            ModuleType.WEB_SCRAPER
        }
        return module_type in streaming_types
    
    @staticmethod
    def _supports_batch(module_type: ModuleType) -> bool:
        """ModuleType이 배치 처리를 지원하는지 확인"""
        # 대부분의 모듈이 배치 처리를 지원, 예외적인 경우만 False
        non_batch_types = {
            ModuleType.SESSION_MANAGER,
            ModuleType.CONDITIONAL_ROUTER
        }
        return module_type not in non_batch_types


# RAG 데이터 플로우 패턴 정의
class RAGDataFlowPatterns:
    """RAG 시스템의 전형적인 데이터 플로우 패턴들"""
    
    # 기본 RAG 파이프라인 플로우
    BASIC_RAG_FLOW = [
        (DataType.TEXT, ModuleType.TEXT_PREPROCESSOR, DataType.TEXT),
        (DataType.TEXT, ModuleType.EMBEDDING_ENCODER, DataType.EMBEDDINGS),
        (DataType.EMBEDDINGS, ModuleType.SIMILARITY_SEARCH, DataType.VECTORS),
        (DataType.VECTORS, ModuleType.DOCUMENT_LOADER, DataType.DOCUMENTS),
        (DataType.DOCUMENTS, ModuleType.CONTEXT_BUILDER, DataType.CONTEXT),
        (DataType.CONTEXT, ModuleType.LLM_GENERATOR, DataType.RESPONSE)
    ]
    
    # 고급 RAG 파이프라인 (재순위화 포함)
    ADVANCED_RAG_FLOW = [
        (DataType.TEXT, ModuleType.QUERY_ANALYZER, DataType.STRUCTURED),
        (DataType.TEXT, ModuleType.TEXT_PREPROCESSOR, DataType.TEXT),
        (DataType.TEXT, ModuleType.EMBEDDING_ENCODER, DataType.EMBEDDINGS),
        (DataType.EMBEDDINGS, ModuleType.SIMILARITY_SEARCH, DataType.VECTORS),
        (DataType.VECTORS, ModuleType.DOCUMENT_RANKER, DataType.DOCUMENTS),
        (DataType.DOCUMENTS, ModuleType.CONTEXT_BUILDER, DataType.CONTEXT),
        (DataType.CONTEXT, ModuleType.LLM_GENERATOR, DataType.RESPONSE),
        (DataType.RESPONSE, ModuleType.RESPONSE_FORMATTER, DataType.RESPONSE)
    ]
    
    # 멀티모달 RAG 플로우
    MULTIMODAL_RAG_FLOW = [
        (DataType.MULTIMODAL, ModuleType.IMAGE_PROCESSOR, DataType.TEXT),
        (DataType.TEXT, ModuleType.TEXT_PREPROCESSOR, DataType.TEXT),
        (DataType.TEXT, ModuleType.EMBEDDING_ENCODER, DataType.EMBEDDINGS),
        (DataType.EMBEDDINGS, ModuleType.SIMILARITY_SEARCH, DataType.VECTORS),
        (DataType.VECTORS, ModuleType.DOCUMENT_LOADER, DataType.DOCUMENTS),
        (DataType.DOCUMENTS, ModuleType.CONTEXT_BUILDER, DataType.CONTEXT),
        (DataType.CONTEXT, ModuleType.LLM_GENERATOR, DataType.RESPONSE)
    ]
    
    @classmethod
    def validate_flow_compatibility(cls, flow_pattern: list) -> Dict[str, bool]:
        """플로우 패턴의 호환성 검증"""
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        for i in range(len(flow_pattern) - 1):
            current_output = flow_pattern[i][2]
            next_input = flow_pattern[i + 1][0]
            
            if current_output != next_input:
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Step {i}: Output type {current_output} doesn't match next input type {next_input}"
                )
        
        return validation_results
    
    @classmethod
    def get_flow_pattern(cls, pattern_name: str) -> Optional[list]:
        """패턴 이름으로 플로우 패턴 반환"""
        patterns = {
            "basic_rag": cls.BASIC_RAG_FLOW,
            "advanced_rag": cls.ADVANCED_RAG_FLOW,
            "multimodal_rag": cls.MULTIMODAL_RAG_FLOW
        }
        return patterns.get(pattern_name)