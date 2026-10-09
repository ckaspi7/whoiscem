"use client";

import { motion } from "motion/react";

const PROMPTS = ["Where does Cem work?", "What has he been listening to?", "What's on his LinkedIn?"];

/** Shown only in the empty state, to demonstrate the assistant's range without making a visitor think one up. */
export function QuickPrompts({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {PROMPTS.map((prompt, i) => (
        <motion.button
          key={prompt}
          type="button"
          onClick={() => onPick(prompt)}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.15 + i * 0.07, ease: [0.16, 1, 0.3, 1] }}
          whileTap={{ scale: 0.96 }}
          className="pill px-3.5 py-1.5 font-sans text-[13px] text-paper-dim transition-colors hover:border-accent/60 hover:text-accent-bright"
        >
          {prompt}
        </motion.button>
      ))}
    </div>
  );
}
