import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // web/CLAUDE.md is hand-written. In development, when `next dev` detects an AI
  // coding agent and finds no managed agent-rules block, Next 16 generates its own
  // generic AGENTS.md and CLAUDE.md in web/ (documented on `agentRules` in
  // next/dist/server/config-shared.d.ts); keep that off so it cannot collide with
  // this file or the root CLAUDE.md.
  agentRules: false,

  // Next 16's dev server rejects the HMR websocket from any origin it does not
  // recognise, and 127.0.0.1 is not treated as localhost. A page opened at
  // http://127.0.0.1:3000 then renders but never hydrates, so nothing is
  // interactive. Dev-only; has no effect on a production build.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
