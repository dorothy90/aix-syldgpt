import { useChat } from './hooks/useChat';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import Sidebar from './components/Sidebar';

function App() {
  const {
    messages,
    currentSession,
    sessions,
    isLoading,
    error,
    sendMessage,
    createNewSession,
    selectSession,
    removeSession,
  } = useChat();

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar
        sessions={sessions}
        currentSession={currentSession}
        onNewChat={createNewSession}
        onSelectSession={selectSession}
        onDeleteSession={removeSession}
      />

      <div className="flex-1 flex flex-col">
        <div className="bg-white border-b border-gray-300 p-4">
          <h1 className="text-2xl font-bold">🤖 RAG Chatbot</h1>
          {currentSession && (
            <p className="text-sm text-gray-600 mt-1">
              💬 {currentSession.title} | 📅 {currentSession.created_at}
            </p>
          )}
        </div>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 m-4">
            <p className="font-bold">오류</p>
            <p>{error}</p>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 mt-20">
              <p className="text-lg">안녕하세요! 무엇을 도와드릴까요?</p>
            </div>
          ) : (
            <div>
              {messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))}
              {isLoading && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-200 rounded-lg px-4 py-2">
                    <div className="flex gap-1">
                      <span className="animate-bounce">●</span>
                      <span className="animate-bounce" style={{ animationDelay: '0.1s' }}>●</span>
                      <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>●</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}

export default App;

