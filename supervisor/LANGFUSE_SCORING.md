## Langfuse 점수(Score) 체계 (csv_supervisor 기준)

목표는 “나중에 개선/비교가 가능한 최소한의 기준”을 먼저 만드는 것입니다. 처음엔 UI에서 수동으로 몇 개만 찍고, 익숙해지면 자동화(룰/평가 스크립트)로 확장하면 됩니다.

### 추천 Score 3종 (최소 세트)

- `routing_correct` (BOOLEAN / 0 또는 1)
  - **의미**: Supervisor가 올바른 에이전트로 라우팅했는가?
  - **예시**
    - `OP001의 연관 파라 알려줘` → relation_agent면 1
    - `OP001 TEMP 그래프` → viz_agent면 1
    - 목록 요청인데 viz로 갔다 → 0

- `tool_success` (BOOLEAN / 0 또는 1)
  - **의미**: 최종적으로 툴 실행이 성공했는가?
  - **판정 기준(예시)**:
    - `filter_csv` 결과가 “조건에 맞는 데이터가 없습니다”가 아니고 표/행이 나왔으면 1
    - `generate_chart` 결과에 차트 경로가 나오고 실제 파일이 생성됐으면 1
    - 예외/빈 데이터/파일 미생성 → 0

- `answer_helpful` (NUMERIC / 1~5)
  - **의미**: 사용자 관점에서 최종 답변이 유용했는가?
  - **예시 가이드**
    - 5: 바로 실행/활용 가능한 답변
    - 3: 대체로 맞지만 누락/애매함 있음
    - 1: 의도와 다름/쓸 수 없음

### Langfuse UI에서 수동으로 찍는 위치

- Traces 목록에서 특정 Trace 클릭
- 상세 페이지에서 **Scores**(또는 Feedback/Score 섹션)에서 추가

### 운영 팁(실습에서 특히 유용)

- `supervisor/langfuse_instrumentation.py`가 **tags/metadata(intent, oper, para 등)**를 넣어주므로,
  - `intent=viz`만 모아 `answer_helpful` 평균을 보면 “그래프 요청 품질”만 따로 볼 수 있습니다.


