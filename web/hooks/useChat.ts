"use client";

import { useCallback, useRef, useState } from "react";

import { sendChatMessage } from "@/lib/api-client";
import type { ChatMessage, ChatResult } from "@/lib/chat-types";

import { useChatHistory } from "./useChatHistory";

export type ChatErrorResult = Exclude<ChatResult, { kind: "success" }>;

export type SendState =
  | { status: "idle" }
  | { status: "loading"; isFirstMessage: boolean }
  | { status: "error"; error: ChatErrorResult };

interface PendingAttempt {
  text: string;
  historyBefore: ChatMessage[];
}

export function useChat() {
  const { messages, addMessage, removeMessage, clear: clearHistory, hydrated } = useChatHistory();
  const [sendState, setSendState] = useState<SendState>({ status: "idle" });
  // The one assistant message that arrived during this page's lifetime and
  // should get the word-by-word reveal. Messages restored from sessionStorage
  // are never "fresh", so a page reload never replays an old answer's reveal.
  const [freshAssistantId, setFreshAssistantId] = useState<string | null>(null);
  const pendingAttempt = useRef<PendingAttempt | null>(null);

  const attempt = useCallback(
    async (text: string, historyBefore: ChatMessage[]) => {
      pendingAttempt.current = { text, historyBefore };
      setSendState({ status: "loading", isFirstMessage: historyBefore.length === 0 });

      const result = await sendChatMessage(text, historyBefore);

      if (result.kind === "success") {
        pendingAttempt.current = null;
        const id = crypto.randomUUID();
        addMessage({
          id,
          role: "assistant",
          content: result.response,
          route: result.route,
          faithfulnessScore: result.faithfulnessScore,
          costUsd: result.costUsd,
          degraded: result.degraded,
        });
        setFreshAssistantId(id);
        setSendState({ status: "idle" });
        return;
      }

      setSendState({ status: "error", error: result });
    },
    [addMessage],
  );

  const send = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || sendState.status === "loading") return;

      const historyBefore = messages;
      addMessage({ id: crypto.randomUUID(), role: "user", content: trimmed });
      void attempt(trimmed, historyBefore);
    },
    [messages, addMessage, attempt, sendState.status],
  );

  // Retries the exact same attempt (same text, same prior history) without
  // appending a second copy of the user's message - the first copy is
  // already sitting in the message list from the original send().
  const retry = useCallback(() => {
    if (!pendingAttempt.current) return;
    const { text, historyBefore } = pendingAttempt.current;
    void attempt(text, historyBefore);
  }, [attempt]);

  // A degraded answer (a backing tool failed, so it may be missing evidence)
  // is replaced rather than added to: drop it and ask the same question again
  // with the history that preceded it. Only the latest message can be retried.
  const retryDegraded = useCallback(() => {
    if (sendState.status === "loading") return;
    const answer = messages[messages.length - 1];
    const question = messages[messages.length - 2];
    if (!answer?.degraded || answer.role !== "assistant" || question?.role !== "user") return;

    removeMessage(answer.id);
    setFreshAssistantId(null);
    void attempt(question.content, messages.slice(0, -2));
  }, [messages, removeMessage, attempt, sendState.status]);

  const clear = useCallback(() => {
    pendingAttempt.current = null;
    setSendState({ status: "idle" });
    setFreshAssistantId(null);
    clearHistory();
  }, [clearHistory]);

  return { messages, sendState, freshAssistantId, send, retry, retryDegraded, clear, hydrated };
}
