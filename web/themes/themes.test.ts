import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { contrastRatio } from "./contrast";
import {
  DEFAULT_THEME,
  THEMES,
  THEME_STORAGE_KEY,
  VISITOR_DEFAULTS,
  counterpart,
  isThemeId,
  schemeOf,
  themeInitScript,
  type ThemeId,
} from "./registry";

const webRoot = path.join(import.meta.dirname, "..");
const read = (relative: string) => readFileSync(path.join(webRoot, relative), "utf8");

const REQUIRED_TOKENS = [
  "ink",
  "surface",
  "surface-raised",
  "paper",
  "paper-dim",
  "paper-faint",
  "line",
  "line-strong",
  "accent",
  "accent-bright",
  "accent-dim",
  "on-accent",
] as const;

type Tokens = Record<string, string>;

function declarations(block: string): Tokens {
  const out: Tokens = {};
  for (const match of block.matchAll(/--(?:color-)?([\w-]+):\s*([^;]+);/g)) {
    out[match[1]] = match[2].trim();
  }
  return out;
}

/** The default theme's colour tokens: the first plain @theme block in globals.css. */
function defaultBlock(): string {
  const match = read("app/globals.css").match(/@theme\s*\{([^}]*)\}/);
  if (!match) throw new Error("globals.css has no @theme block");
  return match[1];
}

function themeBlock(id: ThemeId): string {
  if (id === DEFAULT_THEME) return defaultBlock();
  const match = read(`themes/${id}.css`).match(new RegExp(`:root\\[data-theme="${id}"\\]\\s*\\{([^}]*)\\}`));
  if (!match) throw new Error(`themes/${id}.css has no :root[data-theme="${id}"] block`);
  return match[1];
}

function resolved(id: ThemeId): Tokens {
  return { ...declarations(defaultBlock()), ...declarations(themeBlock(id)) };
}

// [foreground, background, minimum ratio, where it appears]. 4.5 is WCAG AA
// for body text, 3 for large text and UI.
const PAIRS: Array<[string, string, number, string]> = [
  ["paper", "ink", 4.5, "answers on the page"],
  ["paper", "surface", 4.5, "typed text in the field"],
  ["paper", "surface-raised", 4.5, "the visitor's bubble"],
  ["paper-dim", "ink", 4.5, "secondary text"],
  ["paper-dim", "surface", 4.5, "secondary text in the field"],
  ["paper-dim", "surface-raised", 4.5, "secondary text on a chip"],
  ["paper-faint", "ink", 4.5, "captions and the counter"],
  ["paper-faint", "surface", 4.5, "the placeholder"],
  ["paper-faint", "surface-raised", 4.5, "captions on a chip"],
  ["accent-bright", "ink", 4.5, "links and hover text"],
  ["accent-bright", "surface", 4.5, "links on a surface"],
  ["accent-bright", "surface-raised", 4.5, "links on a chip"],
  ["accent", "ink", 3, "list markers and the title dot"],
  ["on-accent", "accent", 4.5, "the send icon"],
];

describe("the contrast helper", () => {
  it("matches the WCAG reference values", () => {
    expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrastRatio("#123456", "#123456")).toBeCloseTo(1, 5);
    // #767676 on white is the well-known lightest grey that still passes AA.
    expect(contrastRatio("#767676", "#fff")).toBeCloseTo(4.54, 1);
  });

  it("accepts short hex and is symmetric", () => {
    expect(contrastRatio("#fff", "#000")).toBeCloseTo(contrastRatio("#000000", "#ffffff"), 8);
  });

  it("refuses a colour it cannot compute rather than passing silently", () => {
    expect(() => contrastRatio("rgb(0 0 0 / 0.5)", "#fff")).toThrow(/opaque hex/);
  });

  it("would have caught the first round's real failure", () => {
    // The Retro build set its quick-prompt text in dusty blue (#7a9fbf) on a
    // pale blue-grey pill (about #e6eaec). That readability problem shipped
    // because nothing measured it; this is the pair the gate exists to fail.
    expect(contrastRatio("#7a9fbf", "#e6eaec")).toBeLessThan(4.5);
  });
});

describe("the registry", () => {
  it("lists unique themes, one of them the default", () => {
    const ids = THEMES.map((theme) => theme.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(isThemeId(DEFAULT_THEME)).toBe(true);
  });

  it("rejects an id it does not know", () => {
    expect(isThemeId("neon")).toBe(false);
    expect(isThemeId(null)).toBe(false);
  });

  it("pairs a theme with its other half, and only where a family has exactly one of each", () => {
    expect(counterpart("aurora")).toBe("aurora-night");
    expect(counterpart("aurora-night")).toBe("aurora");
    expect(counterpart("ember")).toBeNull();

    const families = new Map<string, string[]>();
    for (const theme of THEMES as readonly { id: string; scheme: string; family?: string }[]) {
      if (theme.family) families.set(theme.family, [...(families.get(theme.family) ?? []), theme.scheme]);
    }
    for (const [family, schemes] of families) {
      expect(schemes.slice().sort(), `family ${family}`).toEqual(["dark", "light"]);
    }
  });

  it("gives a visitor who chose nothing a half of the pair that matches their system", () => {
    expect(schemeOf(VISITOR_DEFAULTS.light)).toBe("light");
    expect(schemeOf(VISITOR_DEFAULTS.dark)).toBe("dark");
    expect(counterpart(VISITOR_DEFAULTS.light)).toBe(VISITOR_DEFAULTS.dark);
  });
});

describe.each(THEMES.map((theme) => [theme.id, theme.scheme] as const))("theme %s", (id, scheme) => {
  const tokens = resolved(id);

  it("defines every token itself, so none is silently inherited from another theme", () => {
    const own = declarations(themeBlock(id));
    expect(REQUIRED_TOKENS.filter((token) => !(token in own))).toEqual([]);
  });

  it(`declares color-scheme: ${scheme}, matching the registry`, () => {
    const declared = /color-scheme:\s*(\w+)/.exec(
      id === DEFAULT_THEME ? read("app/globals.css").match(/:root\s*\{([^}]*)\}/)![1] : themeBlock(id),
    );
    expect(declared?.[1]).toBe(scheme);
  });

  it.each(PAIRS)("%s on %s reaches %s:1 (%s)", (foreground, background, minimum) => {
    expect(contrastRatio(tokens[foreground], tokens[background])).toBeGreaterThanOrEqual(minimum);
  });

  // A theme whose page ground is a gradient or a mesh declares each stop as
  // --color-ground-N (the lightest or darkest colour it can reach, for the
  // direction text runs the other way). Text sits straight on the ground, so
  // every text colour must hold up against every stop, not just against --ink.
  const groundStops = Object.keys(tokens).filter((key) => /^ground-\d+$/.test(key));
  it.skipIf(groundStops.length === 0).each(["paper", "paper-dim", "paper-faint"])(
    "%s stays at AA over every stop of the page ground",
    (text) => {
      for (const stop of groundStops) {
        const ratio = contrastRatio(tokens[text], tokens[stop]);
        expect(ratio, `${text} on ${stop} (${tokens[stop]}) is ${ratio.toFixed(2)}:1`).toBeGreaterThanOrEqual(4.5);
      }
    },
  );
});

describe("themeInitScript", () => {
  function run(search: string, stored: string | null, options: { dark?: boolean; storageBlocked?: boolean } = {}) {
    const store = new Map<string, string>(stored ? [[THEME_STORAGE_KEY, stored]] : []);
    const root = { dataset: {} as Record<string, string> };
    const blocked = () => {
      throw new Error("storage blocked");
    };
    new Function("location", "localStorage", "document", "matchMedia", themeInitScript())(
      { search },
      options.storageBlocked
        ? { getItem: blocked, setItem: blocked }
        : { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) },
      { documentElement: root },
      (query: string) => ({ matches: Boolean(options.dark) && query.includes("dark") }),
    );
    return { theme: root.dataset.theme, remembered: store.get(THEME_STORAGE_KEY) };
  }

  it("gives a first-time visitor the half of the default pair that matches their system", () => {
    expect(run("", null).theme).toBe(VISITOR_DEFAULTS.light);
    expect(run("", null, { dark: true }).theme).toBe(VISITOR_DEFAULTS.dark);
  });

  it("still does, with storage blocked, rather than falling back to the page's own default", () => {
    expect(run("", null, { storageBlocked: true }).theme).toBe(VISITOR_DEFAULTS.light);
    expect(run("", null, { storageBlocked: true, dark: true }).theme).toBe(VISITOR_DEFAULTS.dark);
    expect(run("?theme=porcelain", null, { storageBlocked: true }).theme).toBe("porcelain");
  });

  it("applies and remembers a theme asked for in the URL", () => {
    expect(run("?theme=porcelain", null)).toEqual({ theme: "porcelain", remembered: "porcelain" });
  });

  it("uses the remembered theme when the URL says nothing, whatever the system prefers", () => {
    expect(run("", "porcelain").theme).toBe("porcelain");
    expect(run("", "aurora", { dark: true }).theme).toBe("aurora");
  });

  it("lets the URL override what was remembered", () => {
    expect(run("?theme=ember", "porcelain").theme).toBe("ember");
  });

  it("ignores a theme it does not know, whether from the URL or storage", () => {
    expect(run("?theme=neon", null).theme).toBe(VISITOR_DEFAULTS.light);
    expect(run("", "neon", { dark: true }).theme).toBe(VISITOR_DEFAULTS.dark);
  });

  it("falls back to the light half when the browser has no matchMedia", () => {
    const root = { dataset: {} as Record<string, string> };
    new Function("location", "localStorage", "document", themeInitScript())(
      { search: "" },
      { getItem: () => null, setItem: () => undefined },
      { documentElement: root },
    );
    expect(root.dataset.theme).toBe(VISITOR_DEFAULTS.light);
  });
});
