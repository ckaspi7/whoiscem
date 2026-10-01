"use client";

import { useEffect, useState } from "react";

interface Stage {
  at: number;
  text: string;
}

// A visitor's very first message of the session is the one most likely to
// hit a cold backend (lazy-loaded retrieval singletons, plus the host's own
// wake-from-idle delay), so the cold-start reassurance line appears much
// sooner here than it does on later messages.
const FIRST_MESSAGE_STAGES: Stage[] = [
  { at: 0, text: "Thinking." },
  {
    at: 1400,
    text: "This is your first question this visit, so the backend may be cold-starting. That can take up to 30 seconds.",
  },
  { at: 12_000, text: "Still going. A cold start is slow but it is working." },
  { at: 28_000, text: "Almost there, thanks for sticking around." },
];

const RETURNING_MESSAGE_STAGES: Stage[] = [
  { at: 0, text: "Thinking." },
  { at: 2800, text: "Searching Cem's resume and background." },
  {
    at: 10_000,
    text: "Taking longer than usual. The backend may have gone idle again and is waking back up.",
  },
  { at: 28_000, text: "Still working, almost there." },
];

/**
 * A timed, staged copy ladder for the loading state: something generic
 * immediately, something more specific a few seconds in, and a cold-start
 * reassurance line. Discrete text swaps at fixed timestamps, not a
 * continuously-animated value, so plain state + timers (not useMotionValue)
 * is the right tool here.
 */
export function useLoadingLadder(active: boolean, isFirstMessage: boolean): string {
  const stages = isFirstMessage ? FIRST_MESSAGE_STAGES : RETURNING_MESSAGE_STAGES;
  // Stage 0's text ("Thinking.") is identical in both ladders, and the
  // caller (ThinkingIndicator) only mounts this hook for the duration of one
  // loading cycle, so the lazy initial value is already correct - no effect
  // needs to reset it.
  const [text, setText] = useState(() => stages[0].text);

  useEffect(() => {
    if (!active) return;
    // Subscribing to a sequence of timers and setting state from their
    // callbacks, not a synchronous set on mount - stage 0 is already covered
    // by the initial state above.
    const timers = stages
      .slice(1)
      .map((stage) => setTimeout(() => setText(stage.text), stage.at));
    return () => {
      timers.forEach(clearTimeout);
    };
    // `stages` is derived from isFirstMessage each render; depending on the
    // flag directly avoids re-triggering the ladder on unrelated re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, isFirstMessage]);

  return active ? text : "";
}
