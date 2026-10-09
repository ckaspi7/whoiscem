"use client";

import { ArrowClockwise, WarningCircle } from "@phosphor-icons/react";
import { motion } from "motion/react";

import type { ChatMessage } from "@/lib/chat-types";

import { AnswerBody } from "./AnswerBody";
import { TransparencyCaption } from "./TransparencyCaption";

const ENTRY_TRANSITION = { duration: 0.3, ease: [0.16, 1, 0.3, 1] as const };

/**
 * Assistant messages read flush-left like an article, in serif type, with no
 * bubble or fill - a deliberate departure from generic chat-UI convention.
 * User messages keep a conventional right-aligned sans pill so the two
 * voices stay visually distinct without either one looking templated.
 */
interface MessageItemProps {
  message: ChatMessage;
  reveal?: boolean;
  /** Offered only when the answer came back degraded and can be asked again. */
  onRetry?: () => void;
}

export function MessageItem({ message, reveal, onRetry }: MessageItemProps) {
  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={ENTRY_TRANSITION}
        className="flex justify-end"
      >
        <p className="max-w-[75%] rounded-2xl rounded-br-sm bg-surface-raised px-4 py-2.5 font-sans text-[15px] leading-relaxed text-paper">
          <span className="sr-only">You said: </span>
          {message.content}
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={ENTRY_TRANSITION}
      className="max-w-[68ch] border-l-2 border-ember/40 pl-4 sm:pl-5"
    >
      <div className="font-serif text-[17px] leading-[1.7] text-paper/95">
        <span className="sr-only">Cem&rsquo;s assistant said: </span>
        <AnswerBody text={message.content} reveal={reveal} />
      </div>
      {message.degraded && (
        <p
          role="status"
          className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 font-sans text-[13px] leading-snug text-stone-400"
        >
          <span className="inline-flex items-center gap-1.5">
            <WarningCircle size={15} weight="bold" className="shrink-0 text-ember-bright" aria-hidden />
            Part of the knowledge base was unreachable, so this answer may be incomplete.
          </span>
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="inline-flex items-center gap-1.5 rounded-full border border-ember/50 px-3 py-1 font-medium text-ember-bright transition-colors hover:bg-ember/10 active:scale-[0.97]"
            >
              <ArrowClockwise size={13} weight="bold" aria-hidden />
              Try again
            </button>
          )}
        </p>
      )}
      <TransparencyCaption
        route={message.route ?? ""}
        faithfulnessScore={message.faithfulnessScore}
        costUsd={message.costUsd}
      />
    </motion.div>
  );
}
