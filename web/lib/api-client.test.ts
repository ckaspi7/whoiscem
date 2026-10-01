import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sendChatMessage, toWireRole } from "./api-client";
import type { ChatMessage } from "./chat-types";

function jsonResponse(status: number, body: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  });
}

describe("toWireRole", () => {
  it("maps our internal 'user' to the wire literal 'human'", () => {
    expect(toWireRole("user")).toBe("human");
  });

  it("maps our internal 'assistant' to the wire literal 'ai', never 'human'", () => {
    // The backend treats anything other than the exact string "human" as an
    // AIMessage, so this is the one direction that must never be wrong.
    expect(toWireRole("assistant")).toBe("ai");
    expect(toWireRole("assistant")).not.toBe("human");
  });
});

describe("sendChatMessage", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("calls the same-origin proxy, never a direct backend URL", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(200, { response: "hi", route: "conversation", faithfulness_score: 5, cost_usd: 0.001 }),
    );

    await sendChatMessage("hello", []);

    expect(fetch).toHaveBeenCalledTimes(1);
    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/chat");
  });

  it("translates history through the role chokepoint before sending it on the wire", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(200, { response: "ok", route: "resume", faithfulness_score: 4, cost_usd: 0.002 }),
    );

    const history: ChatMessage[] = [
      { id: "1", role: "user", content: "What startup did Cem co-found?" },
      { id: "2", role: "assistant", content: "NeoWise." },
    ];

    await sendChatMessage("How long was that?", history);

    const [, init] = vi.mocked(fetch).mock.calls[0];
    const body = JSON.parse(init!.body as string);

    expect(body.message).toBe("How long was that?");
    expect(body.history).toEqual([
      { role: "human", content: "What startup did Cem co-found?" },
      { role: "ai", content: "NeoWise." },
    ]);
    // Never the raw internal role strings on the wire.
    expect(body.history.some((turn: { role: string }) => turn.role === "user")).toBe(false);
    expect(body.history.some((turn: { role: string }) => turn.role === "assistant")).toBe(false);
  });

  it("returns a success result with the faithfulness score and cost on 200", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(200, {
        response: "Cem works at TELUS.",
        route: "resume",
        faithfulness_score: 5,
        cost_usd: 0.0021,
      }),
    );

    const result = await sendChatMessage("Where does Cem work?", []);

    expect(result).toEqual({
      kind: "success",
      response: "Cem works at TELUS.",
      route: "resume",
      faithfulnessScore: 5,
      costUsd: 0.0021,
    });
  });

  it("reads the retry countdown from the Retry-After header, not the detail string, on 429", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-01T15:30:00.000Z"));
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(429, { detail: "Rate limit exceeded, retry in 17s" }, { "Retry-After": "17" }),
    );

    const result = await sendChatMessage("hi", []);

    expect(result).toEqual({
      kind: "rate-limited",
      retryAfterSeconds: 17,
      retryAt: new Date("2026-10-01T15:30:17.000Z").getTime(),
    });
  });

  it("falls back to a sane default if a 429 arrives with no Retry-After header", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(429, { detail: "Rate limit exceeded" }));

    const result = await sendChatMessage("hi", []);

    expect(result.kind).toBe("rate-limited");
    expect((result as { retryAfterSeconds: number }).retryAfterSeconds).toBeGreaterThan(0);
  });

  it("computes the next UTC midnight as the reset time on a 402 spend cap", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-01T15:30:00.000Z"));
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(402, { detail: "Daily spend cap of $5.00 reached ($5.0021 spent today)" }),
    );

    const result = await sendChatMessage("hi", []);

    expect(result.kind).toBe("spend-capped");
    const spendCapped = result as { detail: string; resetAt: Date };
    expect(spendCapped.detail).toBe("Daily spend cap of $5.00 reached ($5.0021 spent today)");
    expect(spendCapped.resetAt.toISOString()).toBe("2026-10-02T00:00:00.000Z");
  });

  it("extracts every message from a 422's array-shaped detail", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(
      jsonResponse(422, {
        detail: [
          { type: "string_too_long", loc: ["body", "message"], msg: "String should have at most 1000 characters", input: "x".repeat(1001) },
        ],
      }),
    );

    const result = await sendChatMessage("x".repeat(1001), []);

    expect(result).toEqual({
      kind: "validation-error",
      messages: ["body.message: String should have at most 1000 characters"],
    });
  });

  it("treats a 401 as a service misconfiguration, not a user input error", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(401, { detail: "Invalid or missing API key" }));

    const result = await sendChatMessage("hi", []);

    expect(result.kind).toBe("unexpected-auth-error");
  });

  it("returns a network-error result when fetch itself rejects", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("Failed to fetch"));

    const result = await sendChatMessage("hi", []);

    expect(result).toEqual({
      kind: "network-error",
      reason: "unreachable",
      message: expect.any(String),
    });
  });

  it("returns a timeout network-error result once the client-side deadline elapses", async () => {
    vi.useFakeTimers();
    vi.mocked(fetch).mockImplementationOnce(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          const signal = (init as RequestInit).signal as AbortSignal;
          signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        }),
    );

    const pending = sendChatMessage("hi", []);
    await vi.advanceTimersByTimeAsync(60_000);
    const result = await pending;

    expect(result.kind).toBe("network-error");
    expect((result as { reason: string }).reason).toBe("timeout");
  });

  it("falls back to a generic network-error for an unlisted status code", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(500, { detail: "boom" }));

    const result = await sendChatMessage("hi", []);

    expect(result.kind).toBe("network-error");
    expect((result as { reason: string }).reason).toBe("unexpected");
  });
});
