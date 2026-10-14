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

/**
 * Apply a theme and remember it. The picker always writes `?theme=` so that a
 * link shows the same design; the light/dark switch only keeps the URL in step
 * when it already carries one, because the URL wins over a remembered choice on
 * the next load and would otherwise undo the switch.
 */
export function chooseTheme(id: ThemeId, url: "always" | "if-present") {
  document.documentElement.dataset.theme = id;
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
