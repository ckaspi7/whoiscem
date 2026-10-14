"use client";

import { chooseTheme, useActiveTheme } from "@/themes/client";
import { THEMES } from "@/themes/registry";

/**
 * A small floating switcher for comparing themes in one app, one URL. Shown in
 * development, and in a deployed preview when NEXT_PUBLIC_THEME_PICKER=1;
 * visitors to the real site never see it. It reads the theme off <html> rather
 * than keeping its own copy, so it can never disagree with the page.
 */
export function ThemePicker() {
  const active = useActiveTheme();

  return (
    <div
      role="group"
      aria-label="Theme"
      className="fixed bottom-28 left-3 z-50 flex max-w-[calc(100vw-1.5rem)] gap-1 overflow-x-auto rounded-full border border-line-strong bg-surface p-1 font-mono text-[11px] shadow-lg [scrollbar-width:none] sm:bottom-auto sm:left-auto sm:right-3 sm:top-2"
    >
      {THEMES.map((theme) => (
        <button
          key={theme.id}
          type="button"
          aria-pressed={active === theme.id}
          onClick={() => chooseTheme(theme.id, "always")}
          className={`rounded-full px-2.5 py-1 transition-colors ${
            active === theme.id ? "bg-accent text-on-accent" : "text-paper-dim hover:text-paper"
          }`}
        >
          {theme.label}
        </button>
      ))}
    </div>
  );
}
