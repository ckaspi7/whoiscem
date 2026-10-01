"use client";

import { Trash } from "@phosphor-icons/react";

export function Masthead({ onClear, canClear }: { onClear: () => void; canClear: boolean }) {
  return (
    <header className="flex items-start justify-between gap-4 px-4 pb-6 pt-8 sm:px-6 sm:pt-10">
      <div>
        <h1 className="font-sans text-2xl font-semibold tracking-tight text-paper sm:text-3xl">
          Cem Kaspi<span className="text-ember">.</span>
        </h1>
        <p className="mt-1.5 max-w-md font-serif text-[15px] italic leading-snug text-stone-400 sm:text-base">
          Software engineer. Wired into my real resume and work history, not a canned script.
        </p>
      </div>
      {canClear && (
        <button
          type="button"
          onClick={onClear}
          className="mt-1 flex shrink-0 items-center gap-1.5 rounded-full border border-stone-800 px-3 py-1.5 font-sans text-[12.5px] text-stone-400 transition-colors hover:border-stone-600 hover:text-stone-300"
        >
          <Trash size={13} aria-hidden />
          Clear chat
        </button>
      )}
    </header>
  );
}
