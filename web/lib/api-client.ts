import type {
  ChatMessage,
  ChatResult,
  WireChatResponse,
  WireChatTurn,
  WireRole,
} from "./chat-types";

const CHAT_ENDPOINT = "/api/chat";

// A real call normally takes several seconds; a cold backend can take
// 15-30+ seconds. 60s is a generous ceiling past which we stop waiting and
// show a friendly timeout state rather than hang indefinitely.
export const REQUEST_TIMEOUT_MS = 60_000;

/**
 * The single chokepoint that translates our internal role vocabulary into
 * the backend's wire vocabulary. Nothing else in this codebase may produce a
 * WireRole value - see chat-types.ts for why that matters.
 */
export function toWireRole(role: ChatMessage["role"]): WireRole {
  return role === "user" ? "human" : "ai";
}

function toWireHistory(messages: ChatMessage[]): WireChatTurn[] {
  return messages.map((message) => ({
    role: toWireRole(message.role),
    content: message.content,
  }));
}

/** The spend cap resets at UTC midnight (confirmed from spend_tracker.py). */
function nextUtcMidnight(from: Date = new Date()): Date {
  return new Date(
    Date.UTC(from.getUTCFullYear(), from.getUTCMonth(), from.getUTCDate() + 1, 0, 0, 0, 0),
  );
}

async function safeJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

function extractValidationMessages(detail: unknown): string[] {
  if (!Array.isArray(detail) || detail.length === 0) {
    return ["The message could not be validated."];
  }
  return detail.map((entry) => {
    if (entry && typeof entry === "object" && "msg" in entry) {
      const record = entry as Record<string, unknown>;
      const loc = Array.isArray(record.loc) ? record.loc.join(".") : undefined;
      const msg = typeof record.msg === "string" ? record.msg : "Invalid request.";
      return loc ? `${loc}: ${msg}` : msg;
    }
    return "Invalid request.";
  });
}

/**
 * Sends one chat turn through our own same-origin proxy (never the backend
 * URL directly) and parses the response into a discriminated result the UI
 * can switch on. `history` is the conversation so far, in our internal
 * vocabulary; this function is the only caller of toWireHistory/toWireRole
 * on the way out.
 */
export async function sendChatMessage(
  message: string,
  history: ChatMessage[],
): Promise<ChatResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, history: toWireHistory(history) }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof DOMException && err.name === "AbortError") {
      return {
        kind: "network-error",
        reason: "timeout",
        message: "That took far longer than it should have. The backend may be stuck.",
      };
    }
    return {
      kind: "network-error",
      reason: "unreachable",
      message: "Could not reach the assistant. Check your connection and try again.",
    };
  }
  clearTimeout(timeoutId);

  switch (response.status) {
    case 200: {
      const body = (await safeJson(response)) as WireChatResponse | null;
      if (!body || typeof body.response !== "string") {
        return {
          kind: "network-error",
          reason: "unexpected",
          message: "The assistant sent back something unreadable.",
        };
      }
      return {
        kind: "success",
        response: body.response,
        route: body.route,
        faithfulnessScore: body.faithfulness_score,
        costUsd: body.cost_usd,
      };
    }

    case 429: {
      // The real countdown comes from the Retry-After header, never parsed
      // out of the detail string.
      const header = response.headers.get("Retry-After");
      const parsed = header ? Number.parseInt(header, 10) : NaN;
      const retryAfterSeconds = Number.isFinite(parsed) && parsed > 0 ? parsed : 30;
      return {
        kind: "rate-limited",
        retryAfterSeconds,
        retryAt: Date.now() + retryAfterSeconds * 1000,
      };
    }

    case 402: {
      const body = (await safeJson(response)) as { detail?: string } | null;
      return {
        kind: "spend-capped",
        detail: body?.detail ?? "The daily budget for this assistant has been reached.",
        resetAt: nextUtcMidnight(),
      };
    }

    case 422: {
      const body = (await safeJson(response)) as { detail?: unknown } | null;
      return { kind: "validation-error", messages: extractValidationMessages(body?.detail) };
    }

    case 401: {
      // Should never happen in normal operation: the proxy always sends the
      // configured key. Treat it as "service misconfigured", not a
      // user-fixable input error.
      return {
        kind: "unexpected-auth-error",
        message: "The assistant is temporarily misconfigured. Retrying will not fix this.",
      };
    }

    default: {
      return {
        kind: "network-error",
        reason: "unexpected",
        message: `Something went wrong talking to the assistant (status ${response.status}).`,
      };
    }
  }
}
