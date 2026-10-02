"use client";

import { motion } from "motion/react";

import type { ChatMessage } from "@/lib/chat-types";

import { RevealText } from "./RevealText";
import { TransparencyCaption } from "./TransparencyCaption";

const ENTRY_TRANSITION = { duration: 0.3, ease: [0.16, 1, 0.3, 1] as const };

/**
 * Assistant messages read flush-left like an article, in serif type, with no
 * bubble or fill - a deliberate departure from generic chat-UI convention.
 * User messages keep a conventional right-aligned sans pill so the two
 * voices stay visually distinct without either one looking templated.
 */
export function MessageItem({ message, reveal }: { message: ChatMessage; reveal?: boolean }) {
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
      <p className="font-serif text-[17px] leading-[1.7] text-paper/95">
        <span className="sr-only">Cem&rsquo;s assistant said: </span>
        {reveal ? <RevealText text={message.content} /> : message.content}
      </p>
      <TransparencyCaption
        route={message.route ?? ""}
        faithfulnessScore={message.faithfulnessScore}
        costUsd={message.costUsd}
      />
    </motion.div>
  );
}
