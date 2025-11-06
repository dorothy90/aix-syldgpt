import { Session } from '../services/api';

interface SidebarProps {
  sessions: Session[];
  currentSession: Session | null;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
}

export default function Sidebar({
  sessions,
  currentSession,
  onNewChat,
  onSelectSession,
  onDeleteSession,
}: SidebarProps) {
  return (
    <div className="w-64 bg-gray-100 border-r border-gray-300 flex flex-col h-screen">
      <div className="p-4 border-b border-gray-300">
        <h2 className="text-xl font-bold mb-4">채팅 관리</h2>
        <button
          onClick={onNewChat}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          ➕ 새 채팅
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-sm font-semibold text-gray-600 mb-2">채팅 히스토리</h3>
        {sessions.length === 0 ? (
          <p className="text-gray-500 text-sm">채팅 히스토리가 없습니다.</p>
        ) : (
          <div className="space-y-2">
            {sessions.map((session) => {
              const isActive = session.session_id === currentSession?.session_id;
              return (
                <div
                  key={session.session_id}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    isActive ? 'bg-blue-100 border-2 border-blue-500' : 'bg-white hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div
                      className="flex-1 min-w-0"
                      onClick={() => onSelectSession(session.session_id)}
                    >
                      <div className="font-medium text-sm truncate">
                        {session.title}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        📅 {session.created_at}
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteSession(session.session_id);
                      }}
                      className="text-red-500 hover:text-red-700 text-sm px-2 py-1"
                    >
                      삭제
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

