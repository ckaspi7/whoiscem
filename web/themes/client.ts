import { useSyncExternalStore } from "react";

import { DEFAULT_THEME, THEME_STORAGE_KEY, isThemeId, type ThemeId } from "./registry";

function subscribe(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  return () => observer.disconnect();
}

/** The theme on <html> right now, read from the page at the moment of asking. */
export function currentTheme(): ThemeId {
  const theme = document.documentElement.dataset.theme;
  return isThemeId(theme) ? theme : DEFAULT_THEME;
}

function serverTheme(): ThemeId {
  return DEFAULT_THEME;
}

/** The theme on <html>. Read from the page rather than copied, so it can never disagree with it. */
export function useActiveTheme(): ThemeId {
  return useSyncExternalStore(subscribe, currentTheme, serverTheme);
}

/** Chosen, but not yet on the page: a theme change waits for the browser to capture the old look first. */
let pendingTheme: ThemeId | null = null;

/**
 * The theme the page is showing or is about to show. A switch pressed twice in
 * quick succession reads this, so it toggles twice instead of toggling to the
 * same place both times.
 */
export function latestTheme(): ThemeId {
  return pendingTheme ?? currentTheme();
}

function remember(id: ThemeId, url: "always" | "if-present") {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, id);
  } catch {
    // Storage can be blocked; the choice still applies for this page.
  }
  const location = new URL(window.location.href);
  if (url === "always" || location.searchParams.has("theme")) {
    location.searchParams.set("theme", id);
    window.history.replaceState(null, "", location.toString());
  }
}

/**
 * Apply a theme and remember it. The page cross-fades to the new look (the View
 * Transitions API snapshots the old page, applies the change, and fades between
 * the two, which also carries the gradient backgrounds that a CSS transition
 * cannot) and simply switches where that is unsupported or the visitor has asked
 * for reduced motion.
 *
 * The picker always writes `?theme=` so that a link shows the same design; the
 * light/dark switch only keeps the URL in step when it already carries one,
 * because the URL wins over a remembered choice on the next load and would
 * otherwise undo the switch.
 */
export function chooseTheme(id: ThemeId, url: "always" | "if-present") {
  pendingTheme = id;
  remember(id, url);

  const apply = () => {
    document.documentElement.dataset.theme = id;
    if (pendingTheme === id) pendingTheme = null;
  };
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

  if (typeof document.startViewTransition === "function" && !reducedMotion) {
    document.startViewTransition(apply);
  } else {
    apply();
  }
}
