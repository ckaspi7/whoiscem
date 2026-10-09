import type { Element, ElementContent, Root, Text } from "hast";

export const MAX_TOTAL_STAGGER_SECONDS = 1.1;
export const MAX_PER_WORD_DELAY_SECONDS = 0.03;

/**
 * Seconds between consecutive words, chosen so a long answer finishes
 * revealing in about MAX_TOTAL_STAGGER_SECONDS instead of dragging on, while a
 * short one never goes slower than MAX_PER_WORD_DELAY_SECONDS per word.
 */
export function revealStepSeconds(text: string): number {
  const words = text.split(/\s+/).filter(Boolean).length || 1;
  return Math.min(MAX_PER_WORD_DELAY_SECONDS, MAX_TOTAL_STAGGER_SECONDS / words);
}

/** Text inside these is read as a block, so it is never split into words. */
const KEEP_WHOLE = new Set(["code", "pre", "script", "style"]);

function wordSpan(word: string, index: number): Element {
  return {
    type: "element",
    tagName: "span",
    properties: { className: ["reveal-word"], style: `--i:${index}` },
    children: [{ type: "text", value: word }],
  };
}

/**
 * A rehype plugin that wraps every word of ordinary text in
 * `<span class="reveal-word" style="--i:N">`, N counting words in reading
 * order. The stagger itself is pure CSS (`animation-delay: calc(var(--i) *
 * var(--reveal-step))`), so revealing an answer costs one tree pass and no
 * React state, rather than one component and one timer per word. Works on the
 * parsed markdown tree, which is why bold, lists and tables reveal correctly
 * where splitting the raw string could not.
 */
export function rehypeWordReveal() {
  return (tree: Root) => {
    let index = 0;

    const wrap = (parent: Root | Element) => {
      const children: ElementContent[] = [];
      for (const child of parent.children as ElementContent[]) {
        if (child.type === "text") {
          for (const piece of child.value.split(/(\s+)/)) {
            if (!piece) continue;
            children.push(/^\s+$/.test(piece) ? ({ type: "text", value: piece } as Text) : wordSpan(piece, index++));
          }
          continue;
        }
        if (child.type === "element" && !KEEP_WHOLE.has(child.tagName)) wrap(child);
        children.push(child);
      }
      parent.children = children as typeof parent.children;
    };

    wrap(tree);
  };
}
