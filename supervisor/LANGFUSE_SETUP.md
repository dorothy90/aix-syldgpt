## Langfuse 연동 (csv_supervisor 기준)

이 문서는 `supervisor/csv_supervisor.ipynb` 실행 결과가 Langfuse에 Trace로 쌓이도록 만드는 최소 설정을 정리합니다.

### 1) Langfuse UI 접속

이 리포의 self-hosted Langfuse는 `langfuse/docker-compose.yml`에서 **호스트 포트 3001**로 매핑되어 있습니다.

- **Langfuse Web UI**: `http://localhost:3001`

### 2) 프로젝트 생성 + 키 발급

Langfuse UI에서 프로젝트를 만들고 아래 키를 확인합니다.

- **Public Key**: `LANGFUSE_PUBLIC_KEY`
- **Secret Key**: `LANGFUSE_SECRET_KEY`

### 3) 환경변수 설정

이 환경은 `.env*` 파일 생성이 막혀 있을 수 있습니다. 대신 예시 파일을 제공합니다:

- `supervisor/env.example`

로컬에서 실행할 때는 아래처럼 복사해서 사용하세요.

```bash
cp supervisor/env.example supervisor/.env
```

그리고 `.env`에 아래 값을 채웁니다.

- `LANGFUSE_HOST=http://localhost:3001`
- `LANGFUSE_PUBLIC_KEY=...`
- `LANGFUSE_SECRET_KEY=...`

### 4) 노트북 실행 후 확인 포인트

`supervisor/csv_supervisor.ipynb`에서 질의를 1번 실행한 뒤 Langfuse에서 아래를 확인합니다.

- **Traces**: 질의 1건이 Trace 1건으로 생성되는지
- **Timeline**: supervisor/agent/tool 실행이 단계별로(Span) 보이는지
- **Generations**: LLM 호출이 모델명/지연/프롬프트/응답과 함께 기록되는지

추가로, 본 실습 코드(`supervisor/langfuse_instrumentation.py`)는 아래 메타데이터/태그를 자동으로 넣습니다.

- **session_id**: `csv-supervisor-YYYYMMDD-HHMMSS` 형태(또는 `LANGFUSE_SESSION_ID` 지정값)
  - Langfuse에서 `session_id`로 필터하면 노트북 실행 묶음이 한 번에 보입니다.
- **tags**: `csv`, `supervisor`, `local`(환경), `csv-supervisor`(릴리즈), `intent`(filter/viz/relation)
- **metadata**: `query`, (가능하면) `oper`, `para`, `chart_type`

### 5) Score(평가) 붙이기

최소 점수 체계는 아래 문서를 참고하세요:

- `supervisor/LANGFUSE_SCORING.md`


