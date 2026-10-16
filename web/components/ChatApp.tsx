"use client";

import { useEffect, useRef } from "react";

import { useChat } from "@/hooks/useChat";

import { ChatInput } from "./ChatInput";
import { ErrorPanel } from "./ErrorPanel";
import { Masthead } from "./Masthead";
import { MessageList } from "./MessageList";
import { QuickPrompts } from "./QuickPrompts";
import { ThinkingIndicator } from "./ThinkingIndicator";

export function ChatApp() {
  const { messages, sendState, freshAssistantId, send, retry, retryDegraded, clear, hydrated } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);

  const isLoading = sendState.status === "loading";
  const showEmptyState = hydrated && messages.length === 0 && sendState.status === "idle";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, sendState.status]);

  return (
    <div className="app-shell">
      <div className="app-chat">
        <Masthead onClear={clear} canClear={messages.length > 0 && !isLoading} />
        <main className="flex min-h-0 flex-1 flex-col">
          <div className={`flex flex-1 flex-col overflow-y-auto ${showEmptyState ? "justify-center" : ""}`}>
            <div
              className={`mx-auto flex w-full max-w-[720px] flex-col gap-6 px-4 pb-8 sm:px-6 ${
                showEmptyState ? "pb-[12vh]" : ""
              }`}
            >
              {showEmptyState && <QuickPrompts onPick={send} />}
              <MessageList
                messages={messages}
                latestAssistantId={freshAssistantId}
                onRetryDegraded={isLoading ? undefined : retryDegraded}
              />
              {isLoading && <ThinkingIndicator isFirstMessage={sendState.isFirstMessage} />}
              {sendState.status === "error" && <ErrorPanel error={sendState.error} onRetry={retry} />}
            </div>
          </div>
          <ChatInput onSend={send} disabled={isLoading} />
          {/* After the pinned composer, so scrolling here lands the last message just above it, not under it. */}
          <div ref={bottomRef} aria-hidden />
        </main>
      </div>
    </div>
  );
}
