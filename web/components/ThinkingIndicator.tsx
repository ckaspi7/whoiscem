"use client";

import { motion, useReducedMotion } from "motion/react";

import { useLoadingLadder } from "@/hooks/useLoadingLadder";

/**
 * The loading state: an ember dot that pulses (feedback that work is
 * happening) next to a staged copy ladder (useLoadingLadder) that gets more
 * specific, and more reassuring about cold starts, the longer the wait runs.
 */
export function ThinkingIndicator({ isFirstMessage }: { isFirstMessage: boolean }) {
  const text = useLoadingLadder(true, isFirstMessage);
  const reduceMotion = useReducedMotion();

  return (
    <div
      className="flex max-w-[68ch] items-center gap-3 border-l-2 border-ember/40 pl-4 sm:pl-5"
      role="status"
      aria-live="polite"
    >
      <span className="relative flex h-2.5 w-2.5 shrink-0">
        {!reduceMotion && (
          <motion.span
            className="absolute inline-flex h-full w-full rounded-full bg-ember"
            animate={{ scale: [1, 1.9, 1], opacity: [0.55, 0, 0.55] }}
            transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
          />
        )}
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-ember" />
      </span>
      <p className="font-serif text-[15px] italic leading-relaxed text-stone-400">{text}</p>
    </div>
  );
}
