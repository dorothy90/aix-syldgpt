import { Message } from '../services/api';

interface ChatMessageProps {
  message: Message;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * 원본 URL을 프록시 엔드포인트 URL로 변환
 */
function getSecureDocumentUrl(url: string): string {
  // 원본 URL을 인코딩하여 프록시 엔드포인트에 전달
  const encodedUrl = encodeURIComponent(url);
  return `${API_URL}/api/chat/documents/proxy?url=${encodedUrl}`;
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

        {/* 문서 링크 표시 */}
        {!isUser && message.documents && message.documents.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-300">
            <div className="text-sm font-semibold mb-2 text-gray-700">
              관련 문서:
            </div>
            <div className="space-y-2">
              {message.documents.map((doc, index) => (
                <div key={index} className="text-sm">
                  <a
                    href={getSecureDocumentUrl(doc.url)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 underline break-all"
                  >
                    📄 {doc.filename}
                  </a>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

