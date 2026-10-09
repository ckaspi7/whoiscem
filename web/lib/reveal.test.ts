import type { Element, Root } from "hast";
import { describe, expect, it } from "vitest";

import {
  MAX_PER_WORD_DELAY_SECONDS,
  MAX_TOTAL_STAGGER_SECONDS,
  revealStepSeconds,
  rehypeWordReveal,
} from "./reveal";

describe("revealStepSeconds", () => {
  it("never goes slower than the per-word cap on a short answer", () => {
    expect(revealStepSeconds("Short answer.")).toBe(MAX_PER_WORD_DELAY_SECONDS);
  });

  it("compresses a long answer so the whole reveal stays near the total cap", () => {
    const longAnswer = Array.from({ length: 400 }, (_, i) => `word${i}`).join(" ");

    const step = revealStepSeconds(longAnswer);

    expect(step).toBeLessThan(MAX_PER_WORD_DELAY_SECONDS);
    expect(step * 400).toBeCloseTo(MAX_TOTAL_STAGGER_SECONDS, 5);
  });

  it("handles empty text without dividing by zero", () => {
    expect(Number.isFinite(revealStepSeconds(""))).toBe(true);
  });
});

describe("rehypeWordReveal", () => {
  const run = (tree: Root) => {
    rehypeWordReveal()(tree);
    return tree;
  };

  it("splits text into word spans and keeps the whitespace between them", () => {
    const tree: Root = {
      type: "root",
      children: [
        {
          type: "element",
          tagName: "p",
          properties: {},
          children: [{ type: "text", value: "Hello  brave world" }],
        },
      ],
    };

    const paragraph = run(tree).children[0] as Element;

    expect(paragraph.children.map((child) => (child.type === "text" ? child.value : "<span>"))).toEqual([
      "<span>",
      "  ",
      "<span>",
      " ",
      "<span>",
    ]);
  });

  it("numbers words continuously across elements", () => {
    const tree: Root = {
      type: "root",
      children: [
        { type: "element", tagName: "p", properties: {}, children: [{ type: "text", value: "one two" }] },
        { type: "element", tagName: "p", properties: {}, children: [{ type: "text", value: "three" }] },
      ],
    };

    run(tree);

    const styles = tree.children.flatMap((paragraph) =>
      (paragraph as Element).children.flatMap((child) =>
        child.type === "element" ? [String(child.properties.style)] : [],
      ),
    );
    expect(styles).toEqual(["--i:0", "--i:1", "--i:2"]);
  });
});
