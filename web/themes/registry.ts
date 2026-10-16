/**
 * The theme registry, and the contract a theme must meet.
 *
 * A theme is a CSS file, themes/<id>.css, that reassigns the colour tokens
 * documented in app/globals.css (and, optionally, the shape and material
 * variables: radii, --pill-bg, --pill-shadow and friends) under
 * `:root[data-theme="<id>"]`. The baseline tokens every theme overrides live in
 * globals.css itself, because Tailwind generates its utilities from there; the
 * baseline is not a theme and cannot be chosen.
 *
 * To add one: add an entry below, create themes/<id>.css, and @import it from
 * globals.css. themes/themes.test.ts then checks that it defines every token,
 * declares the right color-scheme, and that its text colours clear WCAG AA on
 * every surface they sit on. Rules that keep text readable on glass: text sits
 * on solid tokens, never on a translucent layer; texture and blur belong to
 * the page ground and to chrome, not behind words; and a highlight that
 * lightens a surface must not be used where the text is light.
 *
 * Two themes with the same `family` and opposite schemes are a light/dark pair:
 * the switch in the header moves between them, and a visitor who has not chosen
 * gets the half that matches their operating system.
 */
interface ThemeEntry {
  readonly id: string;
  readonly label: string;
  readonly scheme: "light" | "dark";
  readonly family?: string;
}

export const THEMES = [
  { id: "aurora", label: "Aurora", scheme: "light", family: "aurora" },
  { id: "aurora-night", label: "Aurora Night", scheme: "dark", family: "aurora" },
] as const satisfies readonly ThemeEntry[];

export type ThemeId = (typeof THEMES)[number]["id"];

/** What is assumed before the page has chosen one (the server render) and what an unknown id falls back to: the light half. */
export const DEFAULT_THEME: ThemeId = "aurora";

/** What a visitor who has chosen nothing sees: the half of the pair that matches their system. */
export const VISITOR_DEFAULTS = { light: "aurora", dark: "aurora-night" } as const satisfies Record<
  "light" | "dark",
  ThemeId
>;

export const THEME_STORAGE_KEY = "whoiscem:theme";

export function isThemeId(value: unknown): value is ThemeId {
  return THEMES.some((theme) => theme.id === value);
}

/** The other half of a light/dark pair, or null for a theme that has none. */
export function counterpart(id: ThemeId): ThemeId | null {
  const entries: readonly ThemeEntry[] = THEMES;
  const current = entries.find((theme) => theme.id === id);
  if (!current?.family) return null;
  const other = entries.find((theme) => theme.family === current.family && theme.scheme !== current.scheme);
  return other && isThemeId(other.id) ? other.id : null;
}

export function schemeOf(id: ThemeId): "light" | "dark" {
  const entries: readonly ThemeEntry[] = THEMES;
  return entries.find((theme) => theme.id === id)?.scheme ?? "dark";
}

/**
 * Runs in <head>, before first paint, so a chosen theme never flashes the
 * default first. `?theme=<id>` wins and is remembered; otherwise the
 * remembered choice; otherwise the half of the default pair that matches the
 * visitor's system. Each lookup fails on its own (storage blocked, no
 * matchMedia, odd URL) without taking the others down, so a visitor with
 * storage blocked still gets the right half instead of the fallback theme.
 */
export function themeInitScript(): string {
  const ids = JSON.stringify(THEMES.map((theme) => theme.id));
  const key = JSON.stringify(THEME_STORAGE_KEY);
  const light = JSON.stringify(VISITOR_DEFAULTS.light);
  const dark = JSON.stringify(VISITOR_DEFAULTS.dark);
  return (
    `(function(){var ids=${ids},q=null,s=null,d=false;` +
    `try{q=new URLSearchParams(location.search).get("theme")}catch(e){}` +
    `try{s=localStorage.getItem(${key})}catch(e){}` +
    `try{d=matchMedia("(prefers-color-scheme: dark)").matches}catch(e){}` +
    `var t=ids.indexOf(q)>-1?q:ids.indexOf(s)>-1?s:d?${dark}:${light};` +
    `try{if(ids.indexOf(q)>-1)localStorage.setItem(${key},q)}catch(e){}` +
    `document.documentElement.dataset.theme=t;})();`
  );
}
