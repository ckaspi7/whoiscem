"use client";

import { ArrowClockwise, WarningCircle } from "@phosphor-icons/react";
import { useEffect, useState, type ReactNode } from "react";

import type { ChatErrorResult } from "@/hooks/useChat";

/**
 * Ticks down to a fixed deadline. Subscribes to the passage of time (an
 * external clock) and calls setState from that subscription's callback,
 * rather than resetting state from a prop inside the effect body - the
 * deadline itself is computed once, where the 429 is first handled
 * (api-client.ts), not re-derived here.
 */
function useCountdown(deadlineMs: number): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [deadlineMs]);

  return Math.max(0, Math.ceil((deadlineMs - now) / 1000));
}

function formatResetTime(resetAt: Date): string {
  const hh = String(resetAt.getUTCHours()).padStart(2, "0");
  const mm = String(resetAt.getUTCMinutes()).padStart(2, "0");
  const msUntil = Math.max(0, resetAt.getTime() - Date.now());
  const hoursUntil = Math.floor(msUntil / (1000 * 60 * 60));
  const minutesUntil = Math.floor((msUntil % (1000 * 60 * 60)) / (1000 * 60));
  return `${hh}:${mm} UTC, in about ${hoursUntil}h ${minutesUntil}m`;
}

function PanelShell({ children, onRetry }: { children: ReactNode; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="panel flex max-w-[68ch] items-start gap-3 px-4 py-3.5 font-sans text-[14px] leading-relaxed text-paper-dim"
    >
      <WarningCircle size={18} weight="bold" className="mt-0.5 shrink-0 text-accent-bright" aria-hidden />
      <div className="flex-1">
        {children}
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="pill mt-2.5 inline-flex items-center gap-1.5 border-accent/50 px-3 py-1 font-sans text-[13px] font-medium text-accent-bright transition-colors hover:bg-accent/10 active:scale-[0.97]"
          >
            <ArrowClockwise size={13} weight="bold" aria-hidden />
            Try again
          </button>
        )}
      </div>
    </div>
  );
}

function RateLimitedPanel({
  error,
  onRetry,
}: {
  error: Extract<ChatErrorResult, { kind: "rate-limited" }>;
  onRetry: () => void;
}) {
  const remaining = useCountdown(error.retryAt);
  return (
    <PanelShell onRetry={remaining <= 0 ? onRetry : undefined}>
      <p>
        {remaining > 0
          ? `You're sending messages a little too fast. You can try again in ${remaining}s.`
          : "You can try again now."}
      </p>
    </PanelShell>
  );
}

/** Renders every ChatErrorResult kind with copy and controls suited to it. */
export function ErrorPanel({ error, onRetry }: { error: ChatErrorResult; onRetry: () => void }) {
  switch (error.kind) {
    case "rate-limited":
      return <RateLimitedPanel error={error} onRetry={onRetry} />;

    case "spend-capped":
      return (
        <PanelShell>
          <p>Cem&rsquo;s daily budget for this assistant has been used up for now.</p>
          <p className="mt-1 text-paper-faint">
            It resets at {formatResetTime(error.resetAt)}.
          </p>
        </PanelShell>
      );

    case "validation-error":
      return (
        <PanelShell onRetry={onRetry}>
          <p>That message could not be sent.</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-paper-faint">
            {error.messages.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </PanelShell>
      );

    case "unexpected-auth-error":
      return (
        <PanelShell>
          <p>{error.message}</p>
        </PanelShell>
      );

    case "network-error":
      return (
        <PanelShell onRetry={onRetry}>
          <p>{error.message}</p>
        </PanelShell>
      );

    default: {
      const exhaustive: never = error;
      return exhaustive;
    }
  }
}
