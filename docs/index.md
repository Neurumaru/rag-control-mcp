# MCP-RAG-Control 문서

MCP-RAG-Control은 에이전트 기반 RAG(Retrieval-Augmented Generation) 시스템을 위한 맵 기반 제어 아키텍처입니다. 이 문서에서는 프로젝트의 구현에 필요한 모든 정보를 제공합니다.

## 문서 목록

1. [구현 작업 분해](implementation_tasks.md) - 프로젝트 구현을 위한 세부 작업 목록
2. [프로젝트 구조](project_structure.md) - 프로젝트 디렉토리 및 파일 구조
3. [API 설계](api_design.md) - API 엔드포인트 및 요청/응답 형식
4. [의존성 관리](dependencies.md) - 프로젝트 의존성 및 설치 방법
5. [모델 설계](models_design.md) - 데이터 모델 및 스키마 정의
6. [어댑터 설계](adapters_design.md) - MCP 어댑터 구현
7. [컨트롤러 설계](controller_design.md) - LangGraph 기반 컨트롤러

## 빠른 시작

1. **환경 설정**
   ```bash
   # 가상 환경 생성
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

   # 의존성 설치
   pip install -e .
   ```

2. **개발 환경 설치**
   ```bash
   pip install -e ".[dev]"
   ```

3. **서버 실행**
   ```bash
   uvicorn mcp_rag_control.api.app:app --reload
   ```

## 개발 워크플로우

1. **모듈 개발**
   - 새로운 어댑터 구현
   - 모듈 등록 및 테스트

2. **파이프라인 구성**
   - 모듈 연결
   - 파이프라인 등록
   - 실행 테스트

3. **테스트**
   - 단위 테스트 실행
   - 통합 테스트 실행

## 기여 방법

1. 저장소 포크
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 사항 커밋 (`git commit -m 'Add some amazing feature'`)
4. 브랜치 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 오픈