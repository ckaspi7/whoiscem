import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { FACTS, PROFILE } from "./profile";

const resume = readFileSync(path.join(import.meta.dirname, "../../data/resume.md"), "utf8");

// What in the resume backs each fact. A fact with no entry here, or whose
// evidence disappears from the resume, fails the build.
const EVIDENCE: Record<string, RegExp> = {
  role: /TELUS Communications — AI\/ML Engineer/,
  place: /Vancouver, BC/,
  tenure: /Nearly five years at TELUS/,
  education: /BASc, Electrical Engineering\*\* — University of British Columbia/,
  mentor: /Mentor 100\+ working professionals[\s\S]*University of Texas at Austin/,
  startup: /Co-founded a wearable thermal-device startup/,
};

describe("profile facts", () => {
  it("each have evidence in the resume the assistant answers from", () => {
    for (const fact of FACTS) {
      expect(EVIDENCE[fact.id], `no evidence listed for fact "${fact.id}"`).toBeDefined();
      expect(resume, `"${fact.label}" is not backed by data/resume.md`).toMatch(EVIDENCE[fact.id]);
    }
  });

  it("have unique ids", () => {
    const ids = FACTS.map((fact) => fact.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("link to the LinkedIn profile the resume itself lists", () => {
    expect(resume).toContain(PROFILE.linkedinLabel);
  });
});
