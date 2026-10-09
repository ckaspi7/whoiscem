import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "@/lib/chat-types";

import { MessageItem } from "./MessageItem";

vi.mock("motion/react", async (importOriginal) => ({
  ...(await importOriginal<typeof import("motion/react")>()),
  useReducedMotion: () => false,
}));

const answer = (overrides: Partial<ChatMessage> = {}): ChatMessage => ({
  id: "a1",
  role: "assistant",
  content: "Cem studied **Electrical Engineering**.",
  route: "resume",
  faithfulnessScore: 5,
  costUsd: 0.0004,
  ...overrides,
});

describe("MessageItem", () => {
  it("renders an assistant answer as markdown", () => {
    const { container } = render(<MessageItem message={answer()} />);

    expect(container.querySelector("strong")?.textContent).toBe("Electrical Engineering");
    expect(container.textContent).not.toContain("**");
  });

  it("shows no degraded note on a healthy answer", () => {
    render(<MessageItem message={answer()} onRetry={() => {}} />);

    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
  });

  it("flags a degraded answer and offers a retry that calls back", () => {
    const onRetry = vi.fn();
    render(<MessageItem message={answer({ degraded: true })} onRetry={onRetry} />);

    expect(screen.getByRole("status").textContent).toContain("knowledge base was unreachable");
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("still flags a degraded answer when no retry is offered, without a button", () => {
    render(<MessageItem message={answer({ degraded: true })} />);

    expect(screen.getByRole("status")).not.toBeNull();
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
  });

  it("renders a visitor's message as plain text, not markdown", () => {
    const user: ChatMessage = { id: "u1", role: "user", content: "what is **this**?" };
    const { container } = render(<MessageItem message={user} />);

    expect(container.querySelector("strong")).toBeNull();
    expect(container.textContent).toContain("what is **this**?");
  });
});
