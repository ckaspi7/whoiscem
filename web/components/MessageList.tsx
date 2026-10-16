import type { ChatMessage } from "@/lib/chat-types";

import { MessageItem } from "./MessageItem";

interface MessageListProps {
  messages: ChatMessage[];
  latestAssistantId: string | null;
  /** Offered only on the newest message, and only if it came back degraded. */
  onRetryDegraded?: () => void;
}

export function MessageList({ messages, latestAssistantId, onRetryDegraded }: MessageListProps) {
  return (
    <>
      {messages.map((message, index) => (
        <MessageItem
          key={message.id}
          message={message}
          reveal={message.id === latestAssistantId}
          onRetry={index === messages.length - 1 ? onRetryDegraded : undefined}
        />
      ))}
    </>
  );
}
