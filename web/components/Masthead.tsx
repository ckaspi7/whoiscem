"use client";

import { Trash } from "@phosphor-icons/react";

import { FACTS } from "@/lib/profile";

import { ModeToggle } from "./ModeToggle";

/**
 * Lines up with the conversation below it instead of hugging the viewport edge
 * while the messages sit in the middle. The chat column is 720px with its
 * gutters inside that, so its content is 42rem (max-w-2xl); this header and the
 * composer keep their gutters outside a box of that width, so all three share
 * the same left and right edges at every viewport.
 *
 * The at-a-glance facts are always in the DOM and hidden by CSS
 * (.masthead-facts); a theme that wants hero chips reveals them.
 */
export function Masthead({ onClear, canClear }: { onClear: () => void; canClear: boolean }) {
  return (
    <header className="masthead px-4 pb-6 pt-8 sm:px-6 sm:pt-10">
      <div className="masthead-inner mx-auto flex w-full max-w-2xl items-start justify-between gap-4">
        <div className="masthead-text">
          <h1 className="masthead-title font-display text-2xl font-semibold tracking-tight text-paper sm:text-3xl">
            Cem Kaspi<span className="text-accent">.</span>
          </h1>
          <p className="masthead-sub mt-1.5 max-w-md font-prose text-[15px] italic leading-snug text-paper-faint sm:text-base">
            <span className="masthead-role">Software engineer. </span>
            An assistant wired into Cem&rsquo;s real resume and work history, not a canned script.
          </p>
          <ul className="masthead-facts" aria-label="At a glance">
            {FACTS.slice(0, 3).map((fact) => (
              <li key={fact.id}>{fact.label}</li>
            ))}
          </ul>
        </div>
        {/* Only a box when it has something to hold: an empty one would still add a gap to the row. A theme with a switch to show reveals it regardless. */}
        <div className={`masthead-actions mt-1 shrink-0 items-center gap-2 ${canClear ? "flex" : "hidden"}`}>
          <ModeToggle />
          {canClear && (
            <button
              type="button"
              onClick={onClear}
              className="pill flex shrink-0 items-center gap-1.5 border-line px-3 py-1.5 font-sans text-[12.5px] text-paper-faint transition-colors hover:border-line-strong hover:text-paper-dim"
            >
              <Trash size={13} aria-hidden />
              <span className="clear-label">Clear chat</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
