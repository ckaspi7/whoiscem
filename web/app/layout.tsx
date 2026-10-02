import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Source_Serif_4 } from "next/font/google";

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
  // The page has one theme by design (a dark base with a warm ember accent),
  // not a light/dark toggle, so the chrome stays locked to it too.
  themeColor: "#0f0d0c",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${sourceSerif.variable} bg-ink font-sans text-paper antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
