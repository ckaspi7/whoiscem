"use client";

import { useSyncExternalStore } from "react";

import { DEFAULT_THEME, THEMES, THEME_STORAGE_KEY, isThemeId, type ThemeId } from "@/themes/registry";

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => observer.disconnect();
}

function currentTheme(): ThemeId {
  const theme = document.documentElement.dataset.theme;
  return isThemeId(theme) ? theme : DEFAULT_THEME;
}

function serverTheme(): ThemeId {
  return DEFAULT_THEME;
}

function choose(id: ThemeId) {
  document.documentElement.dataset.theme = id;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, id);
  } catch {
    // Storage can be blocked; the choice still applies for this page.
  }
  const url = new URL(window.location.href);
  url.searchParams.set("theme", id);
  window.history.replaceState(null, "", url.toString());
}

/**
 * A small floating switcher for comparing themes in one app, one URL. Shown in
 * development, and in a deployed preview when NEXT_PUBLIC_THEME_PICKER=1;
 * visitors to the real site never see it. It reads the theme off <html> rather
 * than keeping its own copy, so it can never disagree with the page.
 */
export function ThemePicker() {
  const active = useSyncExternalStore(subscribe, currentTheme, serverTheme);

  return (
    <div
      role="group"
      aria-label="Theme"
      className="fixed bottom-3 left-3 z-50 flex gap-1 rounded-full border border-line-strong bg-surface p-1 font-mono text-[11px] shadow-lg max-sm:bottom-28"
    >
      {THEMES.map((theme) => (
        <button
          key={theme.id}
          type="button"
          aria-pressed={active === theme.id}
          onClick={() => choose(theme.id)}
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
