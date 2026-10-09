import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { sendChatMessage } from "@/lib/api-client";
import type { ChatSuccessResult } from "@/lib/chat-types";

import { useChat } from "./useChat";

vi.mock("@/lib/api-client", () => ({ sendChatMessage: vi.fn() }));

const reply = (overrides: Partial<ChatSuccessResult> = {}): ChatSuccessResult => ({
  kind: "success",
  response: "Electrical Engineering at UBC.",
  route: "resume",
  faithfulnessScore: 5,
  costUsd: 0.0004,
  degraded: false,
  ...overrides,
});

async function sendFirstQuestion(result: { current: ReturnType<typeof useChat> }) {
  await waitFor(() => expect(result.current.hydrated).toBe(true));
  act(() => result.current.send("What did Cem study?"));
  await waitFor(() => expect(result.current.messages).toHaveLength(2));
}

describe("retryDegraded", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.mocked(sendChatMessage).mockReset();
  });

  it("replaces a degraded answer with a fresh one instead of adding to it", async () => {
    vi.mocked(sendChatMessage).mockResolvedValueOnce(
      reply({ response: "I couldn't reach my knowledge base just now.", faithfulnessScore: null, degraded: true }),
    );
    const { result } = renderHook(() => useChat());
    await sendFirstQuestion(result);
    expect(result.current.messages[1].degraded).toBe(true);

    vi.mocked(sendChatMessage).mockResolvedValueOnce(reply());
    act(() => result.current.retryDegraded());

    await waitFor(() => expect(result.current.messages[1]?.content).toBe("Electrical Engineering at UBC."));
    expect(result.current.messages.map((message) => message.role)).toEqual(["user", "assistant"]);
    expect(result.current.messages[1].degraded).toBe(false);
  });

  it("asks the same question again with only the history that preceded it", async () => {
    vi.mocked(sendChatMessage).mockResolvedValueOnce(reply({ degraded: true }));
    const { result } = renderHook(() => useChat());
    await sendFirstQuestion(result);

    vi.mocked(sendChatMessage).mockResolvedValueOnce(reply());
    act(() => result.current.retryDegraded());
    await waitFor(() => expect(sendChatMessage).toHaveBeenCalledTimes(2));

    expect(sendChatMessage).toHaveBeenLastCalledWith("What did Cem study?", []);
  });

  it("does nothing when the latest answer is healthy", async () => {
    vi.mocked(sendChatMessage).mockResolvedValueOnce(reply());
    const { result } = renderHook(() => useChat());
    await sendFirstQuestion(result);

    act(() => result.current.retryDegraded());

    expect(sendChatMessage).toHaveBeenCalledTimes(1);
    expect(result.current.messages).toHaveLength(2);
  });
});
