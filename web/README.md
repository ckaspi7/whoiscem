# whoiscem frontend

Standalone Next.js chat UI for the whoiscem portfolio chatbot. Talks to the
FastAPI backend (`api.py` at the repo root) only through this app's own
server-side proxy route (`app/api/chat/route.ts`) - the browser never sees
the backend's URL or API key.

## Local development

```bash
npm install
cp .env.example .env.local   # then fill in BACKEND_API_URL / BACKEND_API_KEY
npm run dev
```

`BACKEND_API_URL` / `BACKEND_API_KEY` are plain (non-`NEXT_PUBLIC_`) server
env vars read only by the proxy route - see `.env.example`. For local work
against the real backend, point `BACKEND_API_URL` at wherever `uvicorn
api:app` (or the containerized backend) is listening.

## Commands

```bash
npm run dev      # dev server, http://localhost:3000
npm run build    # production build
npm run lint     # eslint
npm test         # vitest (lib/api-client.test.ts: role mapping + every error-shape branch)
npm run test:watch
```

## Structure

- `app/page.tsx` + `components/ChatApp.tsx` - the chat UI itself.
- `app/api/chat/route.ts` - the proxy: injects the backend API key and the
  real visitor IP (`X-Forwarded-For`), relays the response unchanged.
- `lib/api-client.ts` - the only thing that calls `/api/chat`. Owns the
  internal-role -> wire-role translation (`toWireRole`) and parses every
  response into a discriminated `ChatResult`.
- `hooks/useChat.ts` / `hooks/useChatHistory.ts` - conversation state
  (sessionStorage-backed) and the send/retry flow.

## Deployment

Intended for Netlify, building from this subdirectory (`netlify.toml` at the
repo root sets `base = "web"`). `BACKEND_API_URL` and `BACKEND_API_KEY` are
configured as real environment variables in the Netlify UI, not committed.
