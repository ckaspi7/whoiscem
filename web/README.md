# whoiscem frontend

Standalone Next.js chat UI for the whoiscem portfolio chatbot. Talks to the
FastAPI backend (`api.py` at the repo root) only through this app's own
server-side proxy route (`app/api/chat/route.ts`) - the browser never sees
the backend's URL or API key.

## Local development

```bash
npm ci
cp .env.example .env.local   # then fill in BACKEND_API_URL / BACKEND_API_KEY
npm run dev
```

`BACKEND_API_URL` / `BACKEND_API_KEY` are plain (non-`NEXT_PUBLIC_`) server
env vars read only by the proxy route - see `.env.example`. For local work
against the real backend, point `BACKEND_API_URL` at wherever `uvicorn
api:app` (or the containerized backend) is listening, and set the same value
as the backend's `API_KEY`.

## Commands

```bash
npm run dev      # dev server, http://localhost:3000
npm run build    # production build
npm run start    # serve the production build
npm run lint     # eslint
npm test         # vitest: API client, markdown, reveal, chat hook, themes and contrast
npm run test:watch
npm run design-review -- --base http://127.0.0.1:3200   # see below
```

## Structure

- `app/page.tsx` + `components/ChatApp.tsx` - the chat UI itself.
- `app/api/chat/route.ts` - the proxy: injects the backend API key and the
  real visitor IP (`X-Forwarded-For`), relays the response unchanged.
- `lib/api-client.ts` - the only thing that calls `/api/chat`. Owns the
  internal-role -> wire-role translation (`toWireRole`) and parses every
  response into a discriminated `ChatResult`.
- `hooks/useChat.ts` / `hooks/useChatHistory.ts` - conversation state
  (sessionStorage-backed) and the send/retry flow, including retrying a
  degraded answer.
- `components/AnswerBody.tsx` + `lib/reveal.ts` - answers as markdown with the
  word-by-word reveal. Model output is untrusted: raw HTML is inert and
  `javascript:` links are dropped.
- `themes/` - the design. `registry.ts` lists the themes (Aurora and Aurora
  Night), `aurora.css` and `aurora-night.css` reassign the colour tokens, and
  `client.ts` switches themes with a cross-fade.

## Design and themes

There is one light/dark pair. An inline script in `<head>` chooses a theme
before first paint: `?theme=aurora` or `?theme=aurora-night` wins, then the
choice remembered in `localStorage` (`whoiscem:theme`), then the half that
matches the visitor's system setting. Any other value is ignored. The header
switch moves between the halves, fading over 0.6 s (the View Transitions API;
instant under reduced motion or where the API is missing).

A theme is a CSS file that reassigns the tokens documented at the top of
`app/globals.css` under `:root[data-theme="<id>"]`. To add one, add it to
`themes/registry.ts`, create `themes/<id>.css` and `@import` it from
`globals.css`. `themes/themes.test.ts` then fails unless it defines every token
and every text pair clears WCAG AA, including over each stop of a gradient
background.

`npm run design-review` drives the built app in the installed Edge with the
chat API mocked and walks every UI state at desktop and phone width. It
writes screenshots and contact sheets to `design-review/` (gitignored), can
pixel-diff against an earlier run with `--compare-to <dir>`, and measures
text contrast on the rendered pixels. Serve a production build first
(`npm run build && npx next start -p 3200`): `next dev` can time out fetching
the web fonts behind a corporate proxy and render fallback fonts.

## Deployment

Intended for Netlify, building from this subdirectory (`netlify.toml` at the
repo root sets `base = "web"`). `BACKEND_API_URL` and `BACKEND_API_KEY` are
configured as real environment variables in the Netlify UI, not committed.
Nothing is deployed yet.
