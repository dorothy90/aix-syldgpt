def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source><page>{int(doc.metadata['page'])+1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):

    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata}</source></document>"
            for doc in docs
        ]
    )


def format_api_documents(api_docs: list[dict]) -> str:
    """
    API에서 받은 문서들을 format_searched_docs와 동일한 형식으로 포맷팅

    Args:
        api_docs: [{"filename": "...", "content": "...", "url": "..."}, ...]

    Returns:
        포맷팅된 문서 문자열
    """
    formatted = []
    for doc in api_docs:
        formatted.append(
            f"<document><content>{doc.get('content', '')}</content>"
            f"<source>filename={doc.get('filename', '')}, url={doc.get('url', '')}</source></document>"
        )
    return "\n".join(formatted)


def format_task(tasks):
    # 결과를 저장할 빈 리스트 생성
    task_time_pairs = []

    # 리스트를 순회하면서 각 항목을 처리
    for item in tasks:
        # 콜론(:) 기준으로 문자열을 분리
        task, time_str = item.rsplit(":", 1)
        # '시간' 문자열을 제거하고 정수로 변환
        time = int(time_str.replace("시간", "").strip())
        # 할 일과 시간을 튜플로 만들어 리스트에 추가
        task_time_pairs.append((task, time))

    # 결과 출력
    return task_time_pairs
