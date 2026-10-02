"use client";

import { useCallback, useEffect, useState } from "react";

import type { ChatMessage } from "@/lib/chat-types";

const STORAGE_KEY = "whoiscem:chat-history:v1";

function readStoredHistory(): ChatMessage[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ChatMessage[]) : [];
  } catch {
    return [];
  }
}

/**
 * Conversation state lives in sessionStorage only - the backend is fully
 * stateless per request and sends its own `history` on every call. We render
 * an empty list on the very first paint (matching server-rendered output)
 * and hydrate from sessionStorage in an effect, so a visitor with a stored
 * conversation never sees a server/client hydration mismatch.
 */
export function useChatHistory() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    // A deliberate exception to "don't setState synchronously in an effect":
    // sessionStorage doesn't exist during SSR, and reading it during the
    // lazy useState initializer instead would make the client's first render
    // disagree with the server-rendered (empty) HTML, which is the
    // hydration-mismatch bug this two-pass approach exists to avoid. This
    // runs once, on mount, purely to pull in a browser-only API's value.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMessages(readStoredHistory());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // sessionStorage can throw (private browsing, storage full); the
      // conversation still works in-memory for the rest of the tab's life.
    }
  }, [messages, hydrated]);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  return { messages, addMessage, clear, hydrated };
}
