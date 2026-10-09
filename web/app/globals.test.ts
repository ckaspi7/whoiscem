import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const css = readFileSync(path.join(import.meta.dirname, "globals.css"), "utf8");

interface ThemeBlock {
  inline: boolean;
  body: string;
}

function themeBlocks(source: string): ThemeBlock[] {
  return [...source.matchAll(/@theme(\s+inline)?\s*\{([^}]*)\}/g)].map((match) => ({
    inline: Boolean(match[1]),
    body: match[2],
  }));
}

describe("font tokens", () => {
  it("are declared in an `@theme inline` block, because next/font defines its variables on <body>", () => {
    // A plain @theme resolves var() at :root, where --font-geist-sans and
    // friends do not exist. The whole declaration is then invalid at
    // computed-value time and every font utility silently falls back to the
    // system stack, which is how a webfont never loaded in six builds.
    const referencing = themeBlocks(css).filter((block) => /--font-[\w-]+:\s*var\(--font-/.test(block.body));

    expect(referencing.length).toBeGreaterThan(0);
    for (const block of referencing) {
      expect(block.inline).toBe(true);
    }
  });

  it("declare all three families the components use", () => {
    const inline = themeBlocks(css)
      .filter((block) => block.inline)
      .map((block) => block.body)
      .join("\n");

    for (const family of ["--font-sans", "--font-serif", "--font-mono"]) {
      expect(inline).toContain(`${family}:`);
    }
  });
});
