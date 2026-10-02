import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // This repo already has a hand-maintained root-level CLAUDE.md with
  // project-specific guidance; skip Next's auto-generated, generic
  // AGENTS.md/CLAUDE.md pair in web/ so it doesn't shadow or duplicate that.
  agentRules: false,
};

export default nextConfig;
