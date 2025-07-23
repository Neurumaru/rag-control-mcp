"""RAG 시스템의 새로운 ModuleType 사용 예시들"""

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from uuid import uuid4
from models.module import (
    Module, ModuleType, ModuleConfig, DataType, DataSchema, ModuleCapabilities
)
from models.data_flow_schemas import ModuleSchemaRegistry, RAGDataFlowPatterns


def create_text_preprocessor_module():
    """텍스트 전처리 모듈 생성 예시"""
    
    # 자동으로 capabilities 생성
    capabilities = ModuleSchemaRegistry.create_module_capabilities(
        ModuleType.TEXT_PREPROCESSOR,
        expected_latency_ms=50.0,
        max_batch_size=100,
        metrics_collected=["processing_time", "text_length", "chunk_count"]
    )
    
    config = ModuleConfig(
        max_tokens=4096,
        batch_size=50,
        custom_params={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "remove_stopwords": True,
            "normalize_unicode": True
        }
    )
    
    module = Module(
        name="Advanced Text Preprocessor",
        module_type=ModuleType.TEXT_PREPROCESSOR,
        description="텍스트 정제, 분할, 정규화를 수행하는 모듈",
        mcp_server_url="http://localhost:8001",
        config=config,
        capabilities=capabilities,
        tags=["preprocessing", "text", "chunking"]
    )
    
    return module


def create_embedding_encoder_module():
    """임베딩 인코더 모듈 생성 예시"""
    
    capabilities = ModuleSchemaRegistry.create_module_capabilities(
        ModuleType.EMBEDDING_ENCODER,
        expected_latency_ms=100.0,
        max_batch_size=32,
        memory_requirements_mb=2048.0,
        supports_streaming=False,
        metrics_collected=["embedding_time", "token_count", "model_load_time"]
    )
    
    config = ModuleConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_version="1.0",
        dimensions=384,
        batch_size=16,
        max_memory_mb=4096,
        custom_params={
            "normalize_embeddings": True,
            "pooling_mode": "mean",
            "device": "cuda"
        }
    )
    
    module = Module(
        name="Sentence Transformer Encoder",
        module_type=ModuleType.EMBEDDING_ENCODER,
        description="텍스트를 고밀도 벡터 임베딩으로 변환하는 모듈",
        mcp_server_url="http://localhost:8002",
        config=config,
        capabilities=capabilities,
        tags=["embedding", "vector", "sentence-transformers"]
    )
    
    return module


def create_vector_store_module():
    """벡터 저장소 모듈 생성 예시"""
    
    capabilities = ModuleSchemaRegistry.create_module_capabilities(
        ModuleType.VECTOR_STORE,
        expected_latency_ms=20.0,
        max_batch_size=1000,
        is_stateful=True,
        supports_batch=True,
        metrics_collected=["insert_time", "vector_count", "storage_size"]
    )
    
    config = ModuleConfig(
        host="localhost",
        port=6333,
        database_name="rag_vectors",
        collection_name="documents",
        dimensions=384,
        similarity_threshold=0.7,
        max_results=50,
        custom_params={
            "distance_metric": "cosine",
            "ef_construction": 200,
            "m": 16,
            "index_type": "hnsw"
        }
    )
    
    module = Module(
        name="Qdrant Vector Store",
        module_type=ModuleType.VECTOR_STORE,
        description="고성능 벡터 검색을 위한 Qdrant 벡터 데이터베이스",
        mcp_server_url="http://localhost:8003",
        config=config,
        capabilities=capabilities,
        tags=["vector-db", "qdrant", "search"]
    )
    
    return module


def create_llm_generator_module():
    """LLM 생성 모듈 생성 예시"""
    
    capabilities = ModuleSchemaRegistry.create_module_capabilities(
        ModuleType.LLM_GENERATOR,
        expected_latency_ms=2000.0,
        max_batch_size=1,
        memory_requirements_mb=8192.0,
        supports_streaming=True,
        metrics_collected=["generation_time", "token_count", "prompt_tokens", "completion_tokens"]
    )
    
    config = ModuleConfig(
        model_name="gpt-3.5-turbo",
        model_version="0125",
        max_tokens=1500,
        temperature=0.7,
        top_p=0.9,
        api_key="${OPENAI_API_KEY}",
        timeout_seconds=30.0,
        retry_attempts=3,
        custom_params={
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop_sequences": ["\\n\\n"],
            "system_prompt": "You are a helpful AI assistant that provides accurate and relevant information."
        }
    )
    
    module = Module(
        name="OpenAI GPT-3.5 Generator",
        module_type=ModuleType.LLM_GENERATOR,
        description="OpenAI GPT-3.5를 사용한 텍스트 생성 모듈",
        mcp_server_url="http://localhost:8004",
        config=config,
        capabilities=capabilities,
        tags=["llm", "openai", "generation", "gpt-3.5"]
    )
    
    return module


def create_image_processor_module():
    """이미지 처리 모듈 생성 예시 (멀티모달)"""
    
    capabilities = ModuleSchemaRegistry.create_module_capabilities(
        ModuleType.IMAGE_PROCESSOR,
        expected_latency_ms=500.0,
        max_batch_size=10,
        memory_requirements_mb=1024.0,
        metrics_collected=["processing_time", "image_size", "detection_count"]
    )
    
    config = ModuleConfig(
        model_name="blip2-opt-2.7b",
        max_memory_mb=2048,
        batch_size=4,
        custom_params={
            "max_image_size": 1024,
            "quality": "high",
            "detect_objects": True,
            "extract_text": True,
            "generate_captions": True
        }
    )
    
    module = Module(
        name="BLIP2 Image Processor",
        module_type=ModuleType.IMAGE_PROCESSOR,
        description="이미지를 분석하고 텍스트 설명을 생성하는 멀티모달 모듈",
        mcp_server_url="http://localhost:8005",
        config=config,
        capabilities=capabilities,
        tags=["multimodal", "image", "blip2", "vision"]
    )
    
    return module


def demonstrate_rag_pipeline():
    """전체 RAG 파이프라인 구성 예시"""
    
    # 각 모듈 생성
    modules = {
        "preprocessor": create_text_preprocessor_module(),
        "encoder": create_embedding_encoder_module(),
        "vector_store": create_vector_store_module(),
        "llm": create_llm_generator_module()
    }
    
    print("=== RAG 파이프라인 모듈 구성 ===")
    for name, module in modules.items():
        print(f"\\n{name.upper()}:")
        print(f"  이름: {module.name}")
        print(f"  타입: {module.module_type}")
        print(f"  입력: {module.capabilities.input_schema.data_type if module.capabilities else 'N/A'}")
        print(f"  출력: {module.capabilities.output_schema.data_type if module.capabilities else 'N/A'}")
        print(f"  변환: {module.capabilities.transformation_type if module.capabilities else 'N/A'}")
        print(f"  스트리밍: {module.capabilities.supports_streaming if module.capabilities else 'N/A'}")
        print(f"  배치: {module.capabilities.supports_batch if module.capabilities else 'N/A'}")
    
    # 데이터 플로우 검증
    print("\\n=== 데이터 플로우 검증 ===")
    basic_flow = RAGDataFlowPatterns.BASIC_RAG_FLOW
    validation = RAGDataFlowPatterns.validate_flow_compatibility(basic_flow)
    
    print(f"기본 RAG 플로우 유효성: {validation['is_valid']}")
    if validation['errors']:
        for error in validation['errors']:
            print(f"  오류: {error}")
    
    # 플로우 단계별 출력
    print("\\n=== 기본 RAG 플로우 단계 ===")
    for i, (input_type, module_type, output_type) in enumerate(basic_flow, 1):
        print(f"  {i}. {input_type} → {module_type} → {output_type}")
    
    return modules


def demonstrate_multimodal_rag():
    """멀티모달 RAG 파이프라인 예시"""
    
    print("\\n\\n=== 멀티모달 RAG 파이프라인 ===")
    
    # 멀티모달 특화 모듈들
    image_processor = create_image_processor_module()
    text_processor = create_text_preprocessor_module()
    encoder = create_embedding_encoder_module()
    
    modules = [image_processor, text_processor, encoder]
    
    for module in modules:
        print(f"\\n{module.name}:")
        if module.capabilities:
            print(f"  입력: {module.capabilities.input_schema.data_type} ({module.capabilities.input_schema.format})")
            print(f"  출력: {module.capabilities.output_schema.data_type} ({module.capabilities.output_schema.format})")
            print(f"  지연시간: {module.capabilities.expected_latency_ms}ms")
            print(f"  메모리: {module.capabilities.memory_requirements_mb}MB")
    
    # 멀티모달 플로우 검증
    multimodal_flow = RAGDataFlowPatterns.MULTIMODAL_RAG_FLOW
    validation = RAGDataFlowPatterns.validate_flow_compatibility(multimodal_flow)
    
    print(f"\\n멀티모달 RAG 플로우 유효성: {validation['is_valid']}")
    
    print("\\n멀티모달 RAG 플로우 단계:")
    for i, (input_type, module_type, output_type) in enumerate(multimodal_flow, 1):
        print(f"  {i}. {input_type} → {module_type} → {output_type}")


def demonstrate_schema_validation():
    """스키마 검증 예시"""
    
    print("\\n\\n=== 스키마 검증 데모 ===")
    
    # 텍스트 전처리 스키마 확인
    schema = ModuleSchemaRegistry.get_schema_for_type(ModuleType.TEXT_PREPROCESSOR)
    if schema:
        input_schema = schema["input"]
        output_schema = schema["output"]
        
        print("텍스트 전처리 모듈 스키마:")
        print(f"  입력 타입: {input_schema.data_type}")
        print(f"  입력 포맷: {input_schema.format}")
        print(f"  필수 필드: {input_schema.required_fields}")
        print(f"  선택 필드: {input_schema.optional_fields}")
        print(f"  제약사항: {input_schema.constraints}")
        
        print(f"  출력 타입: {output_schema.data_type}")
        print(f"  출력 포맷: {output_schema.format}")
        print(f"  필수 필드: {output_schema.required_fields}")
        print(f"  선택 필드: {output_schema.optional_fields}")
    
    # 벡터 검색 스키마 확인
    vector_schema = ModuleSchemaRegistry.get_schema_for_type(ModuleType.SIMILARITY_SEARCH)
    if vector_schema:
        input_schema = vector_schema["input"]
        output_schema = vector_schema["output"]
        
        print("\\n벡터 검색 모듈 스키마:")
        print(f"  입력 벡터 차원: {input_schema.vector_dimension}")
        print(f"  거리 메트릭: {input_schema.distance_metric}")
        print(f"  출력 스키마 정의: {output_schema.schema_definition}")


if __name__ == "__main__":
    # 기본 RAG 파이프라인 데모
    modules = demonstrate_rag_pipeline()
    
    # 멀티모달 RAG 데모
    demonstrate_multimodal_rag()
    
    # 스키마 검증 데모
    demonstrate_schema_validation()