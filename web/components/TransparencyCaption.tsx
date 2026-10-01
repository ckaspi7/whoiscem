"use client";

import * as Collapsible from "@radix-ui/react-collapsible";
import { CaretDown } from "@phosphor-icons/react";
import { useId, useState } from "react";

interface TransparencyCaptionProps {
  route: string;
  faithfulnessScore: number | null | undefined;
  costUsd: number | undefined;
}

function faithfulnessLabel(score: number | null | undefined): string {
  if (score === null || score === undefined) return "not scored";
  if (score >= 4) return `${score} / 5, confident`;
  if (score >= 2) return `${score} / 5, partially confident`;
  return `${score} / 5, low confidence`;
}

/**
 * A collapsed-by-default caption showing the route taken and the
 * faithfulness score behind an answer. Deliberately understated (small type,
 * closed by default) - this is a transparency touch, not a debug panel.
 */
export function TransparencyCaption({ route, faithfulnessScore, costUsd }: TransparencyCaptionProps) {
  const [open, setOpen] = useState(false);
  const contentId = useId();

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} className="mt-2.5">
      <Collapsible.Trigger
        className="group inline-flex cursor-pointer items-center gap-1 rounded-sm font-mono text-[11px] tracking-wide text-stone-400 transition-colors hover:text-ember-bright focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ember-bright/60"
        aria-controls={contentId}
      >
        <CaretDown
          size={11}
          weight="bold"
          className="transition-transform duration-200 group-data-[state=open]:rotate-180"
          aria-hidden
        />
        How this was answered
      </Collapsible.Trigger>
      <Collapsible.Content
        id={contentId}
        className="overflow-hidden data-[state=closed]:animate-collapsible-up data-[state=open]:animate-collapsible-down"
      >
        <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px] leading-relaxed text-stone-400">
          <dt>Route</dt>
          <dd className="text-stone-300">{route || "unknown"}</dd>
          <dt>Faithfulness</dt>
          <dd className="text-stone-300">{faithfulnessLabel(faithfulnessScore)}</dd>
          {costUsd !== undefined && (
            <>
              <dt>Cost</dt>
              <dd className="text-stone-300">${costUsd.toFixed(4)}</dd>
            </>
          )}
        </dl>
      </Collapsible.Content>
    </Collapsible.Root>
  );
}
