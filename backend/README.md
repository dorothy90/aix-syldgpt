# 백엔드 (FastAPI)

RAG Chatbot의 FastAPI 백엔드 서버입니다.

## 사전 요구사항

- Python 3.11 이상
- OpenSearch 서버 (RAG 기능 사용 시 필요)

### OpenSearch 설치 및 실행

RAG 기능을 사용하려면 OpenSearch 서버가 필요합니다.

**Docker를 사용한 실행:**
```bash
docker run -d \
  -p 9200:9200 \
  -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin" \
  --name opensearch-node \
  opensearchproject/opensearch:latest
```

**또는 Docker Compose 사용:**
```yaml
version: '3'
services:
  opensearch:
    image: opensearchproject/opensearch:latest
    container_name: opensearch-node
    environment:
      - discovery.type=single-node
      - OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin
    ports:
      - "9200:9200"
      - "9600:9600"
    volumes:
      - opensearch-data:/usr/share/opensearch/data

volumes:
  opensearch-data:
```

## 설치

```bash
cd backend
pip install -r requirements.txt
```

## 환경 변수 설정

`.env.example` 파일을 참고하여 `.env` 파일을 생성하고 필요한 환경 변수를 설정하세요.

필수 환경 변수:
- `OPENROUTER_API_KEY`: OpenRouter API 키
- `OPENROUTER_BASE_URL`: OpenRouter API 베이스 URL
- `RETRIEVE_CHAIN_MODEL`: 사용할 LLM 모델 이름
- `EMBEDDINGS_MODEL_NAME`: 임베딩 모델 이름

OpenSearch 설정 (기본값):
- `OPENSEARCH_HOST`: localhost
- `OPENSEARCH_PORT`: 9200
- `OPENSEARCH_USER`: admin
- `OPENSEARCH_PASSWORD`: admin
- `OPENSEARCH_INDEX`: document_embeddings

## 실행

```bash
# 개발 모드
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는
python -m app.main
```

**참고:** OpenSearch 서버가 실행되지 않은 상태에서 채팅 요청을 보내면 연결 오류가 발생합니다. RAG 기능을 사용하려면 OpenSearch 서버를 먼저 실행해야 합니다.

## API 엔드포인트

### 채팅

- `POST /api/chat/stream`: 스트리밍 채팅 요청 (Server-Sent Events)

### 세션 관리

- `POST /api/sessions/`: 새 세션 생성
- `GET /api/sessions/`: 모든 세션 목록 조회
- `GET /api/sessions/{session_id}`: 특정 세션 조회
- `DELETE /api/sessions/{session_id}`: 세션 삭제

### 기타

- `GET /health`: 헬스 체크

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 문제 해결

### OpenSearch 연결 오류

채팅 요청 시 "OpenSearch 서버에 연결할 수 없습니다" 오류가 발생하는 경우:

1. OpenSearch 서버가 실행 중인지 확인:
   ```bash
   curl http://localhost:9200
   ```

2. 환경 변수 확인:
   - `OPENSEARCH_HOST`
   - `OPENSEARCH_PORT`
   - `OPENSEARCH_USER`
   - `OPENSEARCH_PASSWORD`

3. OpenSearch 서버 로그 확인

