import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";

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

export const metadata: Metadata = {
  title: "Cem Kaspi",
  description: "Ask a RAG chatbot about Cem Kaspi's resume, work, and background.",
};

export const viewport: Viewport = {
  // The default theme's page colour. Browser chrome keeps it under any theme;
  // a light theme gets a dark address bar on mobile, a cosmetic gap rather
  // than a readability one, so it is left alone until a theme is chosen.
  themeColor: "#0f0d0c",
};

// The switcher is for comparing designs; visitors to the real site never see it.
const SHOW_THEME_PICKER =
  process.env.NODE_ENV !== "production" || process.env.NEXT_PUBLIC_THEME_PICKER === "1";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: the init script below sets data-theme before
    // React hydrates, so the server's <html> and the client's legitimately differ.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript() }} />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} bg-ink font-sans text-paper antialiased`}
      >
        {children}
        {SHOW_THEME_PICKER && <ThemePicker />}
      </body>
    </html>
  );
}
