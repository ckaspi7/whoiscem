# Frontend (web/)

A standalone Next.js app (`web/`, its own `package.json`) that is the public
face of `api.py`. It is meant for Netlify (`netlify.toml` sets `base = "web"`);
as of this writing neither it nor the API is deployed anywhere. The browser
never talks to the backend: it posts to the app's own `app/api/chat/route.ts`,
which holds `BACKEND_API_URL` and `BACKEND_API_KEY` (plain server env vars,
never `NEXT_PUBLIC_`), adds `X-Api-Key`, and re-attaches the real visitor IP as
`X-Forwarded-For` so the backend's per-IP rate limit still sees visitors
separately. `lib/api-client.ts` is the only caller of `/api/chat`.

- **Answers** render as markdown (`react-markdown` + `remark-gfm`) in
  `components/AnswerBody.tsx`. Model output is untrusted: raw HTML renders as
  inert text and `javascript:` links are dropped. A `degraded` answer (the API
  says a tool failed) shows a note and, on the newest message, a "Try again"
  button that replaces the answer rather than adding one.
- **Design** is one light/dark pair, Aurora and Aurora Night (`themes/`). A theme
  is a CSS file that reassigns the colour tokens under
  `:root[data-theme="<id>"]`; the registry (`themes/registry.ts`) is the only
  list of themes, and an inline script in `<head>` picks one before first paint:
  `?theme=` wins, then the choice stored under `whoiscem:theme`, then the half
  matching the visitor's system setting. Only `aurora` and `aurora-night` are
  accepted; the other candidates that were evaluated were deleted, and
  `themes.test.ts` fails if one becomes selectable again. The tokens in
  `globals.css` are a baseline every theme overrides, not a selectable theme.
  The header switch fades between the halves over 0.6 s with the View
  Transitions API, and switches at once under reduced motion or where the API
  is missing.
- **Fonts** come from `next/font`; the `--ff-*` roles on `:root` name the
  families. In plain CSS use those, never `var(--font-sans)`: the `@theme inline`
  tokens are not emitted as custom properties, and `globals.test.ts` fails on a
  direct reference.
- **Readability is a build gate.** `themes/themes.test.ts` computes WCAG contrast
  for every text pair in each theme (and over every stop of a gradient ground),
  so a colour change that drops below AA fails the tests.
- `npm run design-review` drives the built app in the installed Edge with the
  chat API mocked, through every UI state at desktop and phone width, and can
  pixel-diff against an earlier run (`--compare-to`) and audit contrast on the
  rendered pixels. Output goes to the gitignored `web/design-review/`. Run it on
  a production build, not `next dev`: on this machine the dev server's font
  download times out behind the corporate proxy, so it renders fallback fonts.
- As of this writing CI (`.github/workflows/ci.yml`) has no job for `web/`; check it before relying on this: the tests, lint and
  build run only when someone runs them, and on Netlify's build once it exists.

## Commands

Run from inside `web/`:

```bash
npm ci && npm test && npm run lint && npm run build

# design-review needs the built app serving first (and Microsoft Edge installed); use two terminals:
npx next start -p 3200
npm run design-review -- --base http://127.0.0.1:3200
```
