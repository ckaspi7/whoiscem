import { NextResponse, type NextRequest } from "next/server";

// A cold backend (lazy-loaded cross-encoder/retrieval singletons, plus the
// host's own wake-from-idle delay) can legitimately take 15-30+ seconds. The
// client's own hard timeout is 60s (see lib/api-client.ts); this is set a
// little higher so that, in the normal case, the client's friendlier timeout
// UI fires first and this is only a backstop against the server function
// itself hanging forever.
const BACKEND_FETCH_TIMEOUT_MS = 65_000;

/**
 * The only thing standing between a browser and the real backend. The
 * browser never sees BACKEND_API_URL or BACKEND_API_KEY - both are plain
 * server-side env vars (no NEXT_PUBLIC_ prefix), so they never reach a
 * client bundle.
 *
 * This also re-attaches the real visitor IP as X-Forwarded-For, because the
 * backend's rate limiter keys on it specifically so that many visitors
 * behind this one shared proxy don't collapse into a single bucket. Skipping
 * that would silently break per-visitor rate limiting for everyone the
 * moment this ships.
 */
export async function POST(request: NextRequest) {
  const backendUrl = process.env.BACKEND_API_URL;
  const backendApiKey = process.env.BACKEND_API_KEY;

  if (!backendUrl || !backendApiKey) {
    return NextResponse.json(
      { detail: "The chat backend is not configured on this deployment." },
      { status: 500 },
    );
  }

  // Netlify sets x-nf-client-connection-ip reliably. Fall back to the first
  // hop of x-forwarded-for if that's absent (e.g. local dev through some
  // other proxy). Never fabricate one - an absent header just means the
  // backend falls back to the connecting socket, same as it does today.
  const forwardedFor = request.headers.get("x-forwarded-for");
  const visitorIp =
    request.headers.get("x-nf-client-connection-ip") ?? forwardedFor?.split(",")[0]?.trim();

  const requestBody = await request.text();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Api-Key": backendApiKey,
  };
  if (visitorIp) {
    headers["X-Forwarded-For"] = visitorIp;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), BACKEND_FETCH_TIMEOUT_MS);

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${backendUrl}/chat`, {
      method: "POST",
      headers,
      body: requestBody,
      signal: controller.signal,
    });
  } catch {
    return NextResponse.json(
      { detail: "Could not reach the chat backend." },
      { status: 502 },
    );
  } finally {
    clearTimeout(timeoutId);
  }

  // Relay status, body, and headers (especially Retry-After on a 429)
  // unchanged - the frontend's error handling branches on the real shapes
  // the backend sends, so this must not reshape or swallow anything.
  const responseBody = await backendResponse.text();
  const responseHeaders = new Headers();
  responseHeaders.set(
    "Content-Type",
    backendResponse.headers.get("Content-Type") ?? "application/json",
  );
  const retryAfter = backendResponse.headers.get("Retry-After");
  if (retryAfter) {
    responseHeaders.set("Retry-After", retryAfter);
  }

  return new NextResponse(responseBody, {
    status: backendResponse.status,
    headers: responseHeaders,
  });
}
