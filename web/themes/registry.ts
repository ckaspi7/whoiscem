/**
 * The theme registry, and the contract a theme must meet.
 *
 * A theme is a CSS file, themes/<id>.css, that reassigns the colour tokens
 * documented in app/globals.css (and, optionally, the shape and material
 * variables: radii, --pill-bg, --pill-shadow and friends) under
 * `:root[data-theme="<id>"]`. The default theme's tokens live in globals.css
 * itself, because Tailwind generates its utilities from there.
 *
 * To add one: add an entry below, create themes/<id>.css, and @import it from
 * globals.css. themes/themes.test.ts then checks that it defines every token,
 * declares the right color-scheme, and that its text colours clear WCAG AA on
 * every surface they sit on. Rules that keep text readable on glass: text sits
 * on solid tokens, never on a translucent layer; texture and blur belong to
 * the page ground and to chrome, not behind words; and a highlight that
 * lightens a surface must not be used where the text is light.
 */
export const THEMES = [
  { id: "ember", label: "Ember", scheme: "dark" },
  { id: "porcelain", label: "Porcelain", scheme: "light" },
  { id: "dusk", label: "Dusk", scheme: "dark" },
] as const;

export type ThemeId = (typeof THEMES)[number]["id"];

export const DEFAULT_THEME: ThemeId = "ember";
export const THEME_STORAGE_KEY = "whoiscem:theme";

export function isThemeId(value: unknown): value is ThemeId {
  return THEMES.some((theme) => theme.id === value);
}

/**
 * Runs in <head>, before first paint, so a chosen theme never flashes the
 * default first. `?theme=<id>` wins and is remembered; otherwise the
 * remembered choice; otherwise the default. Any failure (storage blocked,
 * odd URL) leaves the default in place.
 */
export function themeInitScript(): string {
  const ids = JSON.stringify(THEMES.map((theme) => theme.id));
  const key = JSON.stringify(THEME_STORAGE_KEY);
  const fallback = JSON.stringify(DEFAULT_THEME);
  return (
    `(function(){try{var ids=${ids};` +
    `var q=new URLSearchParams(location.search).get("theme");` +
    `var s=localStorage.getItem(${key});` +
    `var t=ids.indexOf(q)>-1?q:ids.indexOf(s)>-1?s:${fallback};` +
    `if(ids.indexOf(q)>-1)localStorage.setItem(${key},q);` +
    `document.documentElement.dataset.theme=t;}catch(e){}})();`
  );
}
