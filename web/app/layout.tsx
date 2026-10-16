import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";

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

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
  preload: false,
});

// On <html>, not <body>: a theme reassigns the --ff-* font roles on :root, and
// var() resolves where it is declared, so the families must exist at :root.
const FONT_VARIABLES = [geistSans, geistMono, sourceSerif, bricolage].map((font) => font.variable).join(" ");

export const metadata: Metadata = {
  title: "Cem Kaspi",
  description: "Ask a RAG chatbot about Cem Kaspi's resume, work, and background.",
};

export const viewport: Viewport = {
  // The page colour of each half of the default pair, picked by the visitor's
  // system setting, which is also what picks their theme until they choose one.
  // Browser chrome keeps it if they then switch to the other half: a cosmetic
  // gap rather than a readability one.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f1fb" },
    { media: "(prefers-color-scheme: dark)", color: "#130f19" },
  ],
};

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
      </body>
    </html>
  );
}
