import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // This repo already has a hand-maintained root-level CLAUDE.md with
  // project-specific guidance; skip Next's auto-generated, generic
  // AGENTS.md/CLAUDE.md pair in web/ so it doesn't shadow or duplicate that.
  agentRules: false,

  // Next 16's dev server rejects the HMR websocket from any origin it does not
  // recognise, and 127.0.0.1 is not treated as localhost. A page opened at
  // http://127.0.0.1:3000 then renders but never hydrates, so nothing is
  // interactive. Dev-only; has no effect on a production build.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
