"use client";

import { Trash } from "@phosphor-icons/react";

/**
 * Shares the chat column's width and gutters (max-w-[720px], px-4/sm:px-6) so
 * the title lines up with the conversation below it instead of hugging the
 * viewport edge while the messages sit in the middle.
 */
export function Masthead({ onClear, canClear }: { onClear: () => void; canClear: boolean }) {
  return (
    <header className="px-4 pb-6 pt-8 sm:px-6 sm:pt-10">
      <div className="mx-auto flex w-full max-w-[720px] items-start justify-between gap-4">
        <div>
          <h1 className="font-sans text-2xl font-semibold tracking-tight text-paper sm:text-3xl">
            Cem Kaspi<span className="text-accent">.</span>
          </h1>
          <p className="mt-1.5 max-w-md font-serif text-[15px] italic leading-snug text-paper-faint sm:text-base">
            Software engineer. An assistant wired into Cem&rsquo;s real resume and work history, not a canned
            script.
          </p>
        </div>
        {canClear && (
          <button
            type="button"
            onClick={onClear}
            className="pill mt-1 flex shrink-0 items-center gap-1.5 border-line px-3 py-1.5 font-sans text-[12.5px] text-paper-faint transition-colors hover:border-line-strong hover:text-paper-dim"
          >
            <Trash size={13} aria-hidden />
            Clear chat
          </button>
        )}
      </div>
    </header>
  );
}
