/**
 * Design read (taste-skill Section 0.B): reading this as a solo developer's
 * portfolio chat product for recruiters and technical visitors evaluating
 * Cem's engineering taste, with a warm editorial-meets-terminal language,
 * leaning toward Tailwind + Motion + hand-styled components over a heavy
 * design system.
 *
 * Dials (Section 1), deliberately above baseline per the brief - the thing
 * being fixed is "safe and generic, no personality":
 *   DESIGN_VARIANCE: 8  - asymmetric masthead, unbubbled article-style
 *                         assistant replies against conventional right-aligned
 *                         user bubbles, no centered hero.
 *   MOTION_INTENSITY: 7 - per-word reveal on answers, staggered entries,
 *                         a motivated pulsing "thinking" indicator; all
 *                         transform/opacity only, all gated behind
 *                         prefers-reduced-motion (see useReducedMotion calls
 *                         across components/).
 *   VISUAL_DENSITY: 3   - airy, single-column reading layout; this is a
 *                         conversation to read, not a cockpit.
 */
import { ChatApp } from "@/components/ChatApp";

export default function Home() {
  return <ChatApp />;
}
