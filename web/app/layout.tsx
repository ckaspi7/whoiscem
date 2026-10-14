import type { Metadata, Viewport } from "next";
import {
  Bricolage_Grotesque,
  Geist,
  Geist_Mono,
  Instrument_Serif,
  Plus_Jakarta_Sans,
  Source_Serif_4,
} from "next/font/google";

import { ThemePicker } from "@/components/ThemePicker";
import { themeInitScript } from "@/themes/registry";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  style: ["normal", "italic"],
});

// Families only some themes use. preload is off so a visitor on another theme
// never downloads them: the @font-face rules are emitted, but the browser only
// fetches a file once a rendered element actually uses that family.
const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  preload: false,
});

const plusJakarta = Plus_Jakarta_Sans({
  variable: "--font-plus-jakarta",
  subsets: ["latin"],
  preload: false,
});

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
  preload: false,
});

// On <html>, not <body>: a theme reassigns the --ff-* font roles on :root, and
// var() resolves where it is declared, so the families must exist at :root.
const FONT_VARIABLES = [
  geistSans,
  geistMono,
  sourceSerif,
  instrumentSerif,
  plusJakarta,
  bricolage,
]
  .map((font) => font.variable)
  .join(" ");

export const metadata: Metadata = {
  title: "Cem Kaspi",
  description: "Ask a RAG chatbot about Cem Kaspi's resume, work, and background.",
};

export const viewport: Viewport = {
  // The page colour of each half of the default pair, picked by the visitor's
  // system setting, which is also what picks their theme until they choose one.
  // Browser chrome keeps it if they then switch or pick another theme: a cosmetic
  // gap rather than a readability one.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f1fb" },
    { media: "(prefers-color-scheme: dark)", color: "#130f19" },
  ],
};

// The switcher is for comparing designs; visitors to the real site never see it.
const SHOW_THEME_PICKER =
  process.env.NODE_ENV !== "production" || process.env.NEXT_PUBLIC_THEME_PICKER === "1";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the init script below sets data-theme before
    // React hydrates, so the server's <html> and the client's legitimately differ.
    <html lang="en" className={FONT_VARIABLES} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript() }} />
      </head>
      <body className="bg-ink font-sans text-paper antialiased">
        {children}
        {SHOW_THEME_PICKER && <ThemePicker />}
      </body>
    </html>
  );
}
