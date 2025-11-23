import { useState, useCallback, useEffect } from 'react';
import { Message, Session } from '../services/api';
import * as api from '../services/api';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 세션 목록 로드
  const loadSessions = useCallback(async () => {
    try {
      const loadedSessions = await api.getSessions();
      setSessions(loadedSessions);

      // 현재 세션이 없거나 삭제된 경우, 첫 번째 세션 선택
      if (!currentSession && loadedSessions.length > 0) {
        setCurrentSession(loadedSessions[0]);
        setMessages(loadedSessions[0].messages);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '세션 목록을 불러오는데 실패했습니다.');
    }
  }, [currentSession]);

  // 초기 로드
  useEffect(() => {
    loadSessions();
  }, []);

  // 새 세션 생성
  const createNewSession = useCallback(async () => {
    try {
      const newSession = await api.createSession();
      setCurrentSession(newSession);
      setMessages([]);
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : '새 세션을 생성하는데 실패했습니다.');
    }
  }, [loadSessions]);

  // 세션 선택
  const selectSession = useCallback(async (sessionId: string) => {
    try {
      const session = await api.getSession(sessionId);
      setCurrentSession(session);
      setMessages(session.messages);
    } catch (err) {
      setError(err instanceof Error ? err.message : '세션을 불러오는데 실패했습니다.');
    }
  }, []);

  // 세션 삭제
  const removeSession = useCallback(async (sessionId: string) => {
    try {
      await api.deleteSession(sessionId);

      // 삭제된 세션이 현재 세션이면 다른 세션 선택 또는 새로 생성
      if (currentSession?.session_id === sessionId) {
        const remainingSessions = sessions.filter(s => s.session_id !== sessionId);
        if (remainingSessions.length > 0) {
          await selectSession(remainingSessions[0].session_id);
        } else {
          await createNewSession();
        }
      }

      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : '세션을 삭제하는데 실패했습니다.');
    }
  }, [currentSession, sessions, selectSession, createNewSession, loadSessions]);

  // 메시지 전송 (스트리밍)
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return;

    const sessionId = currentSession?.session_id;

    // 사용자 메시지 추가
    const userMessage: Message = { role: 'user', content };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    // 어시스턴트 메시지 플레이스홀더 추가
    let assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      let fullContent = '';
      let receivedSessionId = sessionId;
      let documents: api.DocumentReference[] = [];

      // 스트리밍 처리
      for await (const chunk of api.streamChatMessage(content, sessionId)) {
        if (chunk.type === 'session_id' && chunk.session_id) {
          receivedSessionId = chunk.session_id;
          // 새 세션이 생성된 경우
          if (!currentSession && receivedSessionId) {
            const newSession = await api.getSession(receivedSessionId);
            setCurrentSession(newSession);
            await loadSessions();
          }
        } else if (chunk.type === 'documents' && chunk.documents) {
          // 문서 정보 받기
          documents = chunk.documents;
          // 메시지에 문서 정보 추가
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantMessageIndex] = {
              role: 'assistant',
              content: fullContent,
              documents: documents
            };
            return updated;
          });
        } else if (chunk.type === 'token' && chunk.content) {
          fullContent += chunk.content;
          // 실시간으로 메시지 업데이트
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantMessageIndex] = {
              role: 'assistant',
              content: fullContent,
              documents: documents.length > 0 ? documents : undefined
            };
            return updated;
          });
        } else if (chunk.type === 'done' && chunk.content) {
          fullContent = chunk.content;
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantMessageIndex] = {
              role: 'assistant',
              content: fullContent,
              documents: documents.length > 0 ? documents : undefined
            };
            return updated;
          });
        } else if (chunk.type === 'error') {
          throw new Error(chunk.error || '알 수 없는 오류가 발생했습니다.');
        }
      }

      // 세션 목록 새로고침 (메시지가 업데이트되었을 수 있음)
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : '메시지 전송에 실패했습니다.');
      // 에러 발생 시 어시스턴트 메시지 제거
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  }, [currentSession, messages.length, isLoading, loadSessions]);

  return {
    messages,
    currentSession,
    sessions,
    isLoading,
    error,
    sendMessage,
    createNewSession,
    selectSession,
    removeSession,
    loadSessions,
  };
}

