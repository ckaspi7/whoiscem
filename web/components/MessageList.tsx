import type { ChatMessage } from "@/lib/chat-types";

import { MessageItem } from "./MessageItem";

interface MessageListProps {
  messages: ChatMessage[];
  latestAssistantId: string | null;
}

export function MessageList({ messages, latestAssistantId }: MessageListProps) {
  return (
    <>
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} reveal={message.id === latestAssistantId} />
      ))}
    </>
  );
}
