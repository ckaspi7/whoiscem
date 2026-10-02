"use client";

import { PaperPlaneRight } from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useRef, useState, type KeyboardEvent } from "react";

const MAX_CHARS = 1000;

export function ChatInput({ onSend, disabled }: { onSend: (text: string) => void; disabled: boolean }) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const nearLimit = value.length > MAX_CHARS * 0.9;

  return (
    <div className="border-t border-stone-800/80 bg-ink/95 px-4 py-3 backdrop-blur sm:px-6 sm:py-4">
      <div className="mx-auto flex max-w-[720px] items-end gap-2.5">
        <textarea
          ref={textareaRef}
          value={value}
          maxLength={MAX_CHARS}
          disabled={disabled}
          rows={1}
          placeholder="Ask about my work, resume, or projects."
          aria-label="Message"
          onChange={(event) => {
            setValue(event.target.value);
            const el = event.target;
            el.style.height = "auto";
            el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
          }}
          onKeyDown={handleKeyDown}
          className="max-h-40 flex-1 resize-none rounded-2xl border border-stone-800 bg-surface px-4 py-2.5 font-sans text-[15px] text-paper placeholder:text-stone-400 focus:border-ember/60 focus:outline-none disabled:opacity-50"
        />
        <motion.button
          type="button"
          onClick={submit}
          disabled={disabled || !value.trim()}
          whileTap={{ scale: 0.93 }}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-ember text-ink transition-opacity disabled:opacity-30"
          aria-label="Send message"
        >
          <PaperPlaneRight size={17} weight="fill" aria-hidden />
        </motion.button>
      </div>
      <div className="mx-auto mt-1.5 flex max-w-[720px] justify-end">
        <span className={`font-mono text-[10.5px] ${nearLimit ? "text-ember-bright" : "text-stone-400"}`}>
          {value.length} / {MAX_CHARS}
        </span>
      </div>
    </div>
  );
}
