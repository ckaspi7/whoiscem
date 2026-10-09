"use client";

import { useReducedMotion } from "motion/react";
import { useMemo, useState, type CSSProperties } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { rehypeWordReveal, revealStepSeconds } from "@/lib/reveal";

const REMARK_PLUGINS = [remarkGfm];
const REVEAL_PLUGINS = [rehypeWordReveal];
const NO_PLUGINS: never[] = [];

/**
 * Model output is untrusted text: raw HTML in it is rendered as text (no
 * rehype-raw), and react-markdown's default URL transform drops javascript:
 * links. Links open in a new tab without leaking the opener.
 */
const COMPONENTS: Components = {
  a({ node, ...props }) {
    void node;
    return <a {...props} target="_blank" rel="noopener noreferrer" />;
  },
  table({ node, ...props }) {
    void node;
    return (
      <div className="answer-table">
        <table {...props} />
      </div>
    );
  },
};

interface AnswerBodyProps {
  text: string;
  /** Fade the words in once, on mount. Only the freshest answer should. */
  reveal?: boolean;
}

/**
 * An assistant answer rendered as markdown. The backend returns one finished
 * answer rather than a token stream, so `reveal` fades the words in to keep a
 * response from popping in inert; clicking the answer skips straight to the
 * end. Block elements live in a div, never a p, so lists and tables nest
 * validly.
 */
export function AnswerBody({ text, reveal = false }: AnswerBodyProps) {
  const reduceMotion = useReducedMotion();
  const [skipped, setSkipped] = useState(false);
  const animate = reveal && !reduceMotion && !skipped;
  const step = useMemo(() => revealStepSeconds(text), [text]);

  return (
    <div
      className="answer-prose"
      style={animate ? ({ "--reveal-step": `${step}s` } as CSSProperties) : undefined}
      onClick={animate ? () => setSkipped(true) : undefined}
    >
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={animate ? REVEAL_PLUGINS : NO_PLUGINS}
        components={COMPONENTS}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
