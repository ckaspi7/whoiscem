"use client";

import { motion, useReducedMotion } from "motion/react";
import { useMemo } from "react";

interface RevealTextProps {
  text: string;
  className?: string;
}

const MAX_TOTAL_STAGGER_SECONDS = 1.1;
const MAX_PER_WORD_DELAY_SECONDS = 0.03;

/**
 * Reveals `text` word by word on mount to simulate aliveness, since the
 * backend has no real streaming - the full answer lands in one response and
 * an instant pop-in would read as inert. Staggering is done with per-word
 * transition delays (opacity/transform only, GPU-cheap) rather than
 * setInterval + setState, so there is exactly one commit per word instead of
 * one React re-render per tick.
 */
export function RevealText({ text, className }: RevealTextProps) {
  const reduceMotion = useReducedMotion();
  // Split on whitespace but keep the separators, so re-joining the segments
  // reproduces the original spacing exactly.
  const segments = useMemo(() => text.split(/(\s+)/), [text]);

  // Each non-whitespace segment's position among *only* the non-whitespace
  // segments, or -1 for whitespace. Computed with no mutable counter (each
  // entry counts the non-whitespace segments before it) so nothing is
  // reassigned inside the map callback - trivial cost at chat-message length.
  const wordIndices = useMemo(
    () =>
      segments.map((segment, i) =>
        segment.trim().length === 0
          ? -1
          : segments.slice(0, i).filter((s) => s.trim().length > 0).length,
      ),
    [segments],
  );

  if (reduceMotion) {
    return <span className={className}>{text}</span>;
  }

  const wordCount = segments.filter((segment) => segment.trim().length > 0).length || 1;
  const perWordDelay = Math.min(
    MAX_PER_WORD_DELAY_SECONDS,
    MAX_TOTAL_STAGGER_SECONDS / wordCount,
  );

  return (
    <span className={className}>
      {segments.map((segment, i) => {
        if (wordIndices[i] === -1) {
          return segment;
        }
        return (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: 0.4,
              delay: wordIndices[i] * perWordDelay,
              ease: [0.16, 1, 0.3, 1],
            }}
            style={{ display: "inline-block" }}
          >
            {segment}
          </motion.span>
        );
      })}
    </span>
  );
}
