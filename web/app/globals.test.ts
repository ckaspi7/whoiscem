import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const webRoot = path.join(import.meta.dirname, "..");
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

const stripComments = (source: string) => source.replace(/\/\*[\s\S]*?\*\//g, "");

describe("chat layout", () => {
  // The page scrolls as a document, so without these a conversation of a few
  // answers pushes the input below the fold: the composer had to be scrolled to
  // after every reply. design-review's "long" state checks the same in a browser.
  it("pins the composer to the bottom of the viewport", () => {
    const rule = css.match(/\.composer\s*\{([^}]*)\}/)?.[1] ?? "";

    expect(rule).toMatch(/position:\s*sticky/);
    expect(rule).toMatch(/bottom:\s*0/);
  });

  it("scrolls to a marker after the composer, so the last answer lands above it rather than under it", () => {
    const source = readFileSync(path.join(webRoot, "components/ChatApp.tsx"), "utf8");
    const composerAt = source.indexOf("<ChatInput");

    expect(composerAt).toBeGreaterThan(-1);
    expect(source.indexOf("ref={bottomRef}")).toBeGreaterThan(composerAt);
  });
});

describe("font tokens", () => {
  it("are declared in an `@theme inline` block, because a font family is defined below :root", () => {
    // A plain @theme resolves var() at :root, where the families do not exist
    // when they are defined on a descendant. The whole declaration is then
    // invalid at computed-value time and every font utility silently falls back
    // to the system stack, which is how a webfont never loaded in six builds.
    const referencing = themeBlocks(css).filter((block) => /--font-[\w-]+:\s*var\(--(?:font|ff)-/.test(block.body));

    expect(referencing.length).toBeGreaterThan(0);
    for (const block of referencing) {
      expect(block.inline).toBe(true);
    }
  });

  it("cover the four roles the components use", () => {
    const inline = themeBlocks(css)
      .filter((block) => block.inline)
      .map((block) => block.body)
      .join("\n");

    for (const role of ["--font-sans", "--font-prose", "--font-display", "--font-mono"]) {
      expect(inline).toContain(`${role}:`);
    }
  });

  it("each map to a --ff-* family role that is defined on :root, where a theme can reassign it", () => {
    const root = css.match(/:root\s*\{([^}]*)\}/)?.[1] ?? "";

    for (const role of ["--ff-sans", "--ff-prose", "--ff-display", "--ff-mono"]) {
      expect(root, `${role} must be defined on :root`).toContain(`${role}:`);
    }
  });

  it("are never read as var(--font-*) by CSS outside a utility", () => {
    // Inline tokens are inlined into utilities and NOT emitted as custom
    // properties, so var(--font-sans) in a plain rule resolves to nothing and
    // the declaration silently falls back. Plain CSS must use the --ff-* roles.
    // (This is how table, heading and code fonts quietly rendered as the prose
    // serif until a browser check caught it.)
    const files = [
      path.join(webRoot, "app/globals.css"),
      ...readdirSync(path.join(webRoot, "themes"))
        .filter((name) => name.endsWith(".css"))
        .map((name) => path.join(webRoot, "themes", name)),
    ];

    for (const file of files) {
      const offenders = stripComments(readFileSync(file, "utf8")).match(/var\(--font-(?:sans|prose|display|mono|serif)\)/g);
      expect(offenders, `${path.basename(file)} reads an inline font token directly`).toBeNull();
    }
  });
});
