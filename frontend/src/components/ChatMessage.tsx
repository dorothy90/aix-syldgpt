import { Message } from '../services/api';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 ${
          isUser
            ? 'bg-blue-500 text-white'
            : 'bg-gray-200 text-gray-800'
        }`}
      >
        <div className="whitespace-pre-wrap break-words">
          {message.content || <span className="text-gray-400">...</span>}
        </div>
        {!isUser && message.artifacts?.length ? (
          <div className="mt-3 space-y-3">
            {message.artifacts.map((a, idx) => {
              if (a.type === 'html') {
                return (
                  <div
                    key={idx}
                    className="bg-white rounded-lg p-2 border border-gray-300 overflow-x-auto"
                    dangerouslySetInnerHTML={{ __html: a.data }}
                  />
                );
              }
              if (a.type === 'image') {
                return (
                  <img
                    key={idx}
                    className="rounded-lg border border-gray-300 max-w-full"
                    alt={a.title || 'artifact'}
                    src={`data:${a.mime};base64,${a.data}`}
                  />
                );
              }
              return null;
            })}
          </div>
        ) : null}
      </div>
    </div>
  );
}

