import { Message } from './services/api';

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

        {/* LOT 그래프 이미지 표시 */}
        {!isUser && message.plotImage && (
          <div className="mt-3 pt-3 border-t border-gray-300">
            <img
              src={`data:image/png;base64,${message.plotImage}`}
              alt="LOT 그래프"
              className="max-w-full h-auto rounded-lg shadow-md"
              style={{ maxHeight: '500px' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

