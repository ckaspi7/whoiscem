/**
 * Shared types for the chat feature. The split between InternalRole and
 * WireRole is deliberate, not decorative: the backend (api.py) dispatches on
 * the literal string "human" to build a LangChain HumanMessage, and treats
 * every other string as an AIMessage:
 *
 *   (HumanMessage if turn.role == "human" else AIMessage)(...)
 *
 * A typo or a "natural" choice like "user"/"assistant" on the wire would
 * silently become an AIMessage instead of failing loudly. Keeping WireRole as
 * its own type (not a rename of InternalRole) means assigning an InternalRole
 * value into a WireRole-typed field is a compile error, not just a style
 * nit - see toWireRole() in api-client.ts, the only function allowed to
 * produce a WireRole.
 */
export type InternalRole = "user" | "assistant";
export type WireRole = "human" | "ai";

export interface ChatMessage {
  id: string;
  role: InternalRole;
  content: string;
  /** Populated on assistant messages once the backend has answered. */
  route?: string;
  faithfulnessScore?: number | null;
  costUsd?: number;
}

export interface WireChatTurn {
  role: WireRole;
  content: string;
}

export interface WireChatRequest {
  message: string;
  history: WireChatTurn[];
}

export interface WireChatResponse {
  response: string;
  route: string;
  faithfulness_score: number | null;
  cost_usd: number;
}

export interface WireValidationErrorItem {
  type: string;
  loc: Array<string | number>;
  msg: string;
  input: unknown;
}

// ---- The discriminated result the UI switches on. One kind per branch the
// product brief actually distinguishes: success, rate-limited (with a real
// countdown), spend-capped (with a real UTC reset time), a validation
// failure (message shape rejected), a network/timeout failure, and the
// should-never-happen auth misconfiguration. ----

export interface ChatSuccessResult {
  kind: "success";
  response: string;
  route: string;
  faithfulnessScore: number | null;
  costUsd: number;
}

export interface ChatRateLimitedResult {
  kind: "rate-limited";
  retryAfterSeconds: number;
  /** Epoch ms deadline, computed once when the 429 arrived - lets the UI
   *  tick a countdown by subscribing to a clock instead of resetting state
   *  from a prop every time a new rate-limit response comes in. */
  retryAt: number;
}

export interface ChatSpendCappedResult {
  kind: "spend-capped";
  detail: string;
  /** Next UTC midnight, computed client-side - the cap is known to reset then. */
  resetAt: Date;
}

export interface ChatValidationErrorResult {
  kind: "validation-error";
  messages: string[];
}

export interface ChatNetworkErrorResult {
  kind: "network-error";
  reason: "timeout" | "unreachable" | "unexpected";
  message: string;
}

export interface ChatAuthErrorResult {
  kind: "unexpected-auth-error";
  message: string;
}

export type ChatResult =
  | ChatSuccessResult
  | ChatRateLimitedResult
  | ChatSpendCappedResult
  | ChatValidationErrorResult
  | ChatNetworkErrorResult
  | ChatAuthErrorResult;
