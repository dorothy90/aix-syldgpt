const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  plotImage?: string; // Base64 인코딩된 그래프 이미지
}

export interface Session {
  session_id: string;
  title: string;
  messages: Message[];
  created_at: string;
  updated_at?: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
}

/**
 * 스트리밍 채팅 메시지 전송
 */
export async function* streamChatMessage(
  message: string,
  sessionId?: string
): AsyncGenerator<{ type: string; content?: string; session_id?: string; error?: string; image?: string }, void, unknown> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('Response body is not readable');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            yield data;
          } catch (e) {
            console.error('Failed to parse SSE data:', e);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * 세션 생성
 */
export async function createSession(title?: string): Promise<Session> {
  const response = await fetch(`${API_URL}/api/sessions/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ title }),
  });

  if (!response.ok) {
    throw new Error(`Failed to create session: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 모든 세션 조회
 */
export async function getSessions(): Promise<Session[]> {
  const response = await fetch(`${API_URL}/api/sessions/`);

  if (!response.ok) {
    throw new Error(`Failed to get sessions: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 특정 세션 조회
 */
export async function getSession(sessionId: string): Promise<Session> {
  const response = await fetch(`${API_URL}/api/sessions/${sessionId}`);

  if (!response.ok) {
    throw new Error(`Failed to get session: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 세션 삭제
 */
export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/sessions/${sessionId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete session: ${response.statusText}`);
  }
}

