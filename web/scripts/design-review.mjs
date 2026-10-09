#!/usr/bin/env node
/**
 * Design review harness: evidence for a theme, not claims about it.
 *
 * For each theme and viewport it drives the real app (the chat API is mocked
 * in the browser, so no backend or OpenAI credit is needed) through eight
 * states, saves a screenshot of each, builds a contact sheet, and audits
 * readability against the pixels actually rendered.
 *
 *   npm run build && npx next start -p 3200      # in one terminal
 *   npm run design-review -- --base http://127.0.0.1:3200
 *
 *   --themes ember,porcelain   which themes (default: every theme in the registry)
 *   --out design-review        output directory (gitignored)
 *   --compare-to <dir>         pixel-diff the screenshots against an earlier run
 *   --no-audit                 skip the contrast audit
 *
 * Uses the Microsoft Edge already on the machine (BROWSER_CHANNEL=chrome for
 * Chrome), so no browser download is needed.
 *
 * Runs are deterministic (reduced motion, fixed clock) so that two runs differ
 * only where the design does. The consequence is that these screenshots show
 * the settled end state; motion itself is something to see in a browser.
 *
 * The contrast audit is the point. Token tests can only check the colours a
 * theme declares; glass, gradients and translucent layers change what text
 * actually sits on. So the audit hides every piece of text, screenshots the
 * result, and measures each text run's colour against the worst pixel behind it
 * (AA: 4.5, or 3 for large text).
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-core";

const VIEWPORTS = { desktop: { width: 1440, height: 900 }, mobile: { width: 390, height: 844 } };
const STATES = ["empty", "answer", "thinking", "refusal", "degraded", "rate-limit", "spend-cap", "network"];
const SCENARIO = {
  answer: "answer",
  thinking: "slow",
  refusal: "refusal",
  degraded: "degraded",
  "rate-limit": "rate-limit",
  "spend-cap": "spend-cap",
  network: "network",
};
const AUDITED = new Set(["empty", "answer", "refusal", "degraded", "rate-limit", "spend-cap"]);
const EXPECTED_FAILURE_STATES = new Set(["rate-limit", "spend-cap", "network"]);
const FIXED_NOW = new Date("2026-06-15T19:45:00Z");
const SETTLE_MS = 1000;

const LINKEDIN = [
  "Cem's LinkedIn profile includes:",
  "",
  "**Headline:** AI/ML Engineer at TELUS",
  "",
  "**Career History:**",
  "- **AI/ML Engineer at TELUS Communications Inc. (2022-01 - Present)**",
  "  - Building GenAI-powered products and automations.",
  "  - Roles within this position:",
  "    - AI/ML Engineer (2024-09 - Present)",
  "    - Software Developer, GenAI & Chatbot Development (2024-04 - 2024-09)",
  "- **Co-Founder at NeoWise (2019-09 - 2021-05)**",
].join("\n");
const SKILLS = [
  "Cem's technical skills, by area:",
  "",
  "| Area | Tools |",
  "|---|---|",
  "| Languages | Python, TypeScript, SQL |",
  "| Cloud | GCP, Azure, Terraform |",
  "",
  "Run `pytest tests/unit/` for the guardrail tests. See [his LinkedIn](https://linkedin.com/in/cemkaspi).",
].join("\n");
const REFUSAL = "I don't have reliable information about that in my knowledge base.";
const DEGRADED =
  "I couldn't reach my knowledge base just now, so I can't answer that reliably. Please try again in a moment.";

function parseArgs(argv) {
  const args = { base: "http://127.0.0.1:3200", themes: null, out: "design-review", compareTo: null, audit: true };
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === "--base") args.base = argv[++i];
    else if (flag === "--themes") args.themes = argv[++i].split(",");
    else if (flag === "--out") args.out = argv[++i];
    else if (flag === "--compare-to") args.compareTo = argv[++i];
    else if (flag === "--no-audit") args.audit = false;
    else throw new Error(`unknown option ${flag}`);
  }
  return args;
}

async function registryThemeIds() {
  const source = await readFile(new URL("../themes/registry.ts", import.meta.url), "utf8");
  return [...source.matchAll(/\{\s*id:\s*"([\w-]+)"/g)].map((match) => match[1]);
}

async function mockChat(page, scenario) {
  await page.route("**/api/chat", async (route) => {
    const { message } = JSON.parse(route.request().postData() ?? "{}");
    const ok = (body) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ route: "resume", cost_usd: 0.0004, degraded: false, faithfulness_score: 5, ...body }),
      });
    try {
      switch (scenario) {
        case "answer":
          return await ok({ response: /skill|tool/i.test(message) ? SKILLS : LINKEDIN });
        case "slow":
          await new Promise((resolve) => setTimeout(resolve, 9000));
          return await ok({ response: LINKEDIN });
        case "refusal":
          return await ok({ response: REFUSAL, faithfulness_score: 1 });
        case "degraded":
          return await ok({ response: DEGRADED, degraded: true, faithfulness_score: null });
        case "rate-limit":
          return await route.fulfill({
            status: 429,
            headers: { "Retry-After": "30" },
            contentType: "application/json",
            body: JSON.stringify({ detail: "Rate limit exceeded, retry in 30s" }),
          });
        case "spend-cap":
          return await route.fulfill({
            status: 402,
            contentType: "application/json",
            body: JSON.stringify({ detail: "Daily spend cap of $5.00 reached ($5.0021 spent today)" }),
          });
        case "network":
          return await route.abort("connectionrefused");
        default:
          return await ok({ response: "ok" });
      }
    } catch {
      // The page was closed while a deliberately slow response was pending.
    }
  });
}

const answers = (page) => page.locator(".answer-prose");

async function drive(page, state) {
  const chip = (name) => page.getByRole("button", { name }).click();
  const ask = async (text) => {
    await page.getByRole("textbox", { name: "Message" }).fill(text);
    await page.getByRole("button", { name: "Send message" }).click();
  };
  switch (state) {
    case "empty":
      break;
    case "answer":
      await chip("What's on his LinkedIn?");
      await answers(page).first().waitFor();
      await ask("What are his skills and tools?");
      await page.waitForFunction(() => document.querySelectorAll(".answer-prose").length >= 2);
      await page.waitForTimeout(2400);
      break;
    case "thinking":
      await chip("Where does Cem work?");
      await page.getByRole("status").first().waitFor();
      await page.waitForTimeout(1800);
      break;
    case "refusal":
    case "degraded":
      await chip("Where does Cem work?");
      await answers(page).first().waitFor();
      await page.waitForTimeout(2400);
      break;
    default:
      await chip("Where does Cem work?");
      await page.getByRole("alert").first().waitFor();
      await page.waitForTimeout(400);
  }
}

/** In the page: every visible text run, its colour, size and client rects. */
function collectTextRuns() {
  const sentinel = "#123456";
  const toRgba = (css) => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = sentinel;
    ctx.fillStyle = css;
    if (ctx.fillStyle === sentinel && css.toLowerCase() !== sentinel) return null;
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
    return [r, g, b, a / 255];
  };
  const effectiveOpacity = (el) => {
    let opacity = 1;
    for (let node = el; node; node = node.parentElement) opacity *= Number.parseFloat(getComputedStyle(node).opacity);
    return opacity;
  };
  const runs = [];
  const unparsed = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = node.nodeValue.replace(/\s+/g, " ").trim();
    const el = node.parentElement;
    if (!text || !el || el.closest("script,style,noscript,.sr-only,[aria-hidden=true]")) continue;
    const style = getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden") continue;
    const opacity = effectiveOpacity(el);
    if (opacity < 0.05) continue;
    const range = document.createRange();
    range.selectNodeContents(node);
    const rects = [...range.getClientRects()].filter((r) => r.width >= 2 && r.height >= 4);
    if (!rects.length) continue;
    const color = toRgba(style.color);
    if (!color) {
      unparsed.push(`${text.slice(0, 30)} (${style.color})`);
      continue;
    }
    const size = Number.parseFloat(style.fontSize);
    const weight = Number.parseInt(style.fontWeight, 10);
    runs.push({
      text: text.slice(0, 40),
      color: [color[0], color[1], color[2], color[3] * opacity],
      large: size >= 24 || (size >= 18.66 && weight >= 700),
      rects: rects.map((r) => ({ x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height })),
    });
  }
  for (const field of document.querySelectorAll("textarea")) {
    const box = field.getBoundingClientRect();
    if (field.value || !field.placeholder || box.width < 2) continue;
    const color = toRgba(getComputedStyle(field, "::placeholder").color);
    if (color) {
      runs.push({
        text: "(placeholder)",
        color,
        large: false,
        rects: [{ x: box.left + scrollX + 14, y: box.top + scrollY + 8, w: Math.min(280, box.width - 28), h: box.height - 16 }],
      });
    }
  }
  return { runs, unparsed };
}

/** In a scratch page: decode a screenshot and measure each run against its worst background pixel. */
async function measureContrast({ b64, runs }) {
  const image = new Image();
  image.src = `data:image/png;base64,${b64}`;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const linear = (v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  const luminance = (r, g, b) => 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b);
  const ratio = (a, b) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

  return runs.map((run) => {
    let worst = Infinity;
    let worstBackground = null;
    for (const rect of run.rects) {
      const cols = Math.max(2, Math.min(12, Math.round(rect.w / 8)));
      const rows = Math.max(2, Math.min(4, Math.round(rect.h / 6)));
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = Math.min(image.width - 1, Math.max(0, Math.round(rect.x + ((i + 0.5) * rect.w) / cols)));
          const y = Math.min(image.height - 1, Math.max(0, Math.round(rect.y + ((j + 0.5) * rect.h) / rows)));
          const [br, bg, bb] = ctx.getImageData(x, y, 1, 1).data;
          const alpha = run.color[3];
          const fr = run.color[0] * alpha + br * (1 - alpha);
          const fg = run.color[1] * alpha + bg * (1 - alpha);
          const fb = run.color[2] * alpha + bb * (1 - alpha);
          const value = ratio(luminance(fr, fg, fb), luminance(br, bg, bb));
          if (value < worst) {
            worst = value;
            worstBackground = [br, bg, bb];
          }
        }
      }
    }
    return { text: run.text, ratio: worst, need: run.large ? 3 : 4.5, background: worstBackground, color: run.color };
  });
}

async function auditContrast(page, scratch) {
  const { runs, unparsed } = await page.evaluate(collectTextRuns);
  await page.addStyleTag({
    content:
      "*{color:transparent!important;-webkit-text-fill-color:transparent!important;text-shadow:none!important}" +
      "::placeholder{color:transparent!important;-webkit-text-fill-color:transparent!important}svg{visibility:hidden!important}",
  });
  const shot = await page.screenshot({ fullPage: true });
  const measured = await scratch.evaluate(measureContrast, { b64: shot.toString("base64"), runs });
  const failures = measured
    .filter((m) => m.ratio < m.need)
    .map((m) => ({ text: m.text, ratio: Number(m.ratio.toFixed(2)), need: m.need, background: m.background }));
  // The runs closest to their threshold, by margin: where a theme is tightest.
  const tightest = [...measured]
    .sort((a, b) => a.ratio / a.need - b.ratio / b.need)
    .slice(0, 3)
    .map((m) => ({ text: m.text, ratio: Number(m.ratio.toFixed(2)), need: m.need }));
  return {
    checked: measured.length,
    lowest: Number(Math.min(...measured.map((m) => m.ratio)).toFixed(2)),
    tightest,
    failures,
    unparsed,
  };
}

async function diffImages({ a, b }) {
  const load = async (b64) => {
    const image = new Image();
    image.src = `data:image/png;base64,${b64}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0);
    return { width: image.width, height: image.height, data: ctx.getImageData(0, 0, image.width, image.height).data };
  };
  const [first, second] = await Promise.all([load(a), load(b)]);
  if (first.width !== second.width || first.height !== second.height) {
    return { sameSize: false, size: [first.width, first.height, second.width, second.height] };
  }
  let differing = 0;
  for (let i = 0; i < first.data.length; i += 4) {
    const delta = Math.max(
      Math.abs(first.data[i] - second.data[i]),
      Math.abs(first.data[i + 1] - second.data[i + 1]),
      Math.abs(first.data[i + 2] - second.data[i + 2]),
    );
    if (delta > 2) differing++;
  }
  return { sameSize: true, differing, total: first.width * first.height };
}

async function buildSheet(scratch, file, entries, columns, width) {
  const cells = entries
    .map(
      (e) =>
        `<figure style="margin:0"><figcaption style="color:#fff;padding:2px 0;font:12px system-ui">${e.label}</figcaption>` +
        `<img style="width:100%;display:block" src="data:image/png;base64,${e.b64}"></figure>`,
    )
    .join("");
  await scratch.setViewportSize({ width, height: 800 });
  await scratch.setContent(
    `<body style="margin:0;background:#777"><div style="display:grid;grid-template-columns:repeat(${columns},1fr);gap:8px;padding:8px">${cells}</div></body>`,
  );
  await scratch.waitForFunction(() => [...document.images].every((img) => img.complete));
  await scratch.screenshot({ path: file, fullPage: true });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const themes = args.themes ?? (await registryThemeIds());
  await mkdir(args.out, { recursive: true });
  const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL ?? "msedge", headless: true });
  const scratch = await (await browser.newContext()).newPage();
  await scratch.goto("about:blank");

  const report = {};
  let problems = 0;

  for (const theme of themes) {
    await mkdir(path.join(args.out, theme), { recursive: true });
    report[theme] = {};
    for (const [viewportName, viewport] of Object.entries(VIEWPORTS)) {
      const entry = { overflow: [], console: [], audit: {}, fonts: null };
      const sheet = [];
      for (const state of STATES) {
        // Deterministic on purpose, so two runs differ only where the design does:
        // reduced motion (no entrance animations, no pulsing dot, no word reveal)
        // and a fixed clock (the spend-cap panel prints the time until UTC midnight).
        const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
        const page = await context.newPage();
        await page.clock.setFixedTime(FIXED_NOW);
        page.on("console", (m) => {
          if (!["error", "warning"].includes(m.type())) return;
          // The browser logs every deliberately mocked 429, 402 and refused
          // connection; those are the states under test, not problems.
          if (EXPECTED_FAILURE_STATES.has(state) && /Failed to load resource/.test(m.text())) return;
          // Motion's development-only note about the reduced-motion emulation above.
          if (/You have Reduced Motion enabled/.test(m.text())) return;
          entry.console.push(`${state}: ${m.text()}`);
        });
        page.on("pageerror", (e) => entry.console.push(`${state}: ${e.message}`));
        if (SCENARIO[state]) await mockChat(page, SCENARIO[state]);
        await page.goto(`${args.base}/?theme=${theme}`, { waitUntil: "networkidle" });
        await page.addStyleTag({ content: '[aria-label="Theme"],nextjs-portal{display:none!important}' });
        // Motion animates entrances in JavaScript, which reduced-motion emulation
        // does not stop, so give them time to land before anything is captured.
        await page.waitForTimeout(SETTLE_MS);
        await drive(page, state);
        await page.waitForTimeout(SETTLE_MS);
        // A full-page capture draws a fixed background where the page happens to
        // be scrolled, leaving a band at the top; capture from the top instead.
        await page.evaluate(() => window.scrollTo(0, 0));

        const file = path.join(args.out, theme, `${viewportName}-${state}.png`);
        const png = await page.screenshot({ path: file, fullPage: true });
        sheet.push({ label: state, b64: png.toString("base64") });

        const facts = await page.evaluate(async () => {
          await document.fonts.ready;
          const first = (el) => (el ? getComputedStyle(el).fontFamily.split(",")[0].replaceAll('"', "") : null);
          return {
            overflow: document.documentElement.scrollWidth > window.innerWidth,
            theme: document.documentElement.dataset.theme,
            title: first(document.querySelector("h1")),
            prose: first(document.querySelector(".answer-prose")),
            loaded: [...document.fonts].filter((f) => f.status === "loaded").map((f) => f.family.replaceAll('"', "")),
          };
        });
        if (facts.overflow) entry.overflow.push(state);
        if (facts.theme !== theme) entry.console.push(`${state}: theme resolved to ${facts.theme}`);
        entry.fonts ??= { title: facts.title, loaded: new Set() };
        facts.loaded.forEach((f) => entry.fonts.loaded.add(f));

        if (args.audit && AUDITED.has(state)) entry.audit[state] = await auditContrast(page, scratch);
        await context.close();
      }
      await buildSheet(
        scratch,
        path.join(args.out, theme, `${viewportName}-sheet.png`),
        sheet,
        viewportName === "desktop" ? 2 : 4,
        viewportName === "desktop" ? 1500 : 1600,
      );
      entry.fonts.loaded = [...entry.fonts.loaded].filter((f) => !f.includes("Fallback")).sort();
      report[theme][viewportName] = entry;

      const failed = Object.entries(entry.audit).filter(([, a]) => a.failures.length);
      problems += failed.length + entry.overflow.length + entry.console.length;
      const lowest = Math.min(...Object.values(entry.audit).map((a) => a.lowest), Infinity);
      console.log(
        `${theme.padEnd(10)} ${viewportName.padEnd(8)} overflow:${entry.overflow.length ? entry.overflow : "none"}  ` +
          `console:${entry.console.length || "none"}  lowest contrast:${Number.isFinite(lowest) ? lowest : "n/a"}  ` +
          `title font:${entry.fonts.title}  fonts:${entry.fonts.loaded.join("+")}`,
      );
      const tightest = Object.values(entry.audit)
        .flatMap((a) => a.tightest)
        .sort((a, b) => a.ratio / a.need - b.ratio / b.need)[0];
      if (tightest) {
        console.log(`   tightest margin: "${tightest.text}" ${tightest.ratio}:1 (needs ${tightest.need})`);
      }
      for (const [state, a] of failed) {
        for (const f of a.failures.slice(0, 6)) {
          console.log(`   LOW CONTRAST in ${state}: "${f.text}" ${f.ratio}:1 (needs ${f.need}) on rgb(${f.background})`);
        }
      }
      for (const [state, a] of Object.entries(entry.audit)) {
        if (a.unparsed.length) console.log(`   (could not parse colour in ${state}: ${a.unparsed.join("; ")})`);
      }
    }
  }

  if (args.compareTo) {
    console.log(`\nPixel comparison against ${args.compareTo}:`);
    for (const theme of themes) {
      for (const file of (await readdir(path.join(args.out, theme))).filter((f) => !f.includes("sheet"))) {
        let other;
        try {
          other = await readFile(path.join(args.compareTo, theme, file));
        } catch {
          continue;
        }
        const mine = await readFile(path.join(args.out, theme, file));
        const diff = await scratch.evaluate(diffImages, { a: mine.toString("base64"), b: other.toString("base64") });
        const label = diff.sameSize ? `${diff.differing} of ${diff.total} pixels differ` : `size changed ${diff.size}`;
        console.log(`  ${theme}/${file}: ${label}`);
        if (!diff.sameSize || diff.differing) problems++;
      }
    }
  }

  await writeFile(path.join(args.out, "report.json"), JSON.stringify(report, null, 2));
  await browser.close();
  console.log(`\n${problems ? `${problems} problem(s) found` : "No problems found"}. Screenshots and sheets are in ${args.out}/`);
  process.exitCode = problems ? 1 : 0;
}

main().catch((error) => {
  console.error("design-review failed:", error);
  process.exit(2);
});
