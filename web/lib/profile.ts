/**
 * Facts shown around the chat by themes that want a profile (a side rail, hero
 * chips). Each is a claim made in data/resume.md, the source the assistant
 * itself answers from; profile.test.ts fails if one stops appearing there, so
 * this page cannot quietly assert something the resume does not.
 */
export const PROFILE = {
  name: "Cem Kaspi",
  initials: "CK",
  role: "AI/ML Engineer",
  linkedinUrl: "https://linkedin.com/in/cemkaspi",
  linkedinLabel: "linkedin.com/in/cemkaspi",
} as const;

export interface Fact {
  id: string;
  label: string;
}

export const FACTS: readonly Fact[] = [
  { id: "role", label: "AI/ML Engineer at TELUS Communications" },
  { id: "place", label: "Vancouver, BC" },
  { id: "tenure", label: "Nearly five years at TELUS" },
  { id: "education", label: "BASc Electrical Engineering, UBC" },
  { id: "mentor", label: "Mentors 100+ professionals in a UT Austin AI/ML programme" },
  { id: "startup", label: "Co-founded NeoWise, a wearable thermal-device startup" },
];
