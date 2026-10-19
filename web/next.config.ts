import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // web/CLAUDE.md is hand-written, next to a hand-maintained root CLAUDE.md.
  // Skip Next's auto-generated, generic AGENTS.md/CLAUDE.md pair so it can
  // neither overwrite web/CLAUDE.md nor shadow or duplicate the root file.
  agentRules: false,

  // Next 16's dev server rejects the HMR websocket from any origin it does not
  // recognise, and 127.0.0.1 is not treated as localhost. A page opened at
  // http://127.0.0.1:3000 then renders but never hydrates, so nothing is
  // interactive. Dev-only; has no effect on a production build.
  allowedDevOrigins: ["127.0.0.1"],
};

export default nextConfig;
