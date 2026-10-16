import { fireEvent, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AnswerBody } from "./AnswerBody";

const motion = vi.hoisted(() => ({ reduced: false }));
vi.mock("motion/react", () => ({ useReducedMotion: () => motion.reduced }));

beforeEach(() => {
  motion.reduced = false;
});

const LINKEDIN_STYLE_ANSWER = [
  "Cem's LinkedIn profile includes:",
  "",
  "**Headline:** AI/ML Engineer at TELUS",
  "",
  "**Career History:**",
  "- **AI/ML Engineer at TELUS (2022-01 - Present)**",
  "  - Building GenAI-powered products",
  "- **Co-Founder at NeoWise (2019-09 - 2021-05)**",
].join("\n");

describe("markdown rendering", () => {
  it("renders bold as <strong> and leaves no literal asterisks", () => {
    const { container } = render(<AnswerBody text={LINKEDIN_STYLE_ANSWER} />);

    expect(container.querySelector("strong")?.textContent).toBe("Headline:");
    expect(container.textContent).not.toContain("**");
  });

  it("renders bullet lists as real lists, including nested ones", () => {
    const { container } = render(<AnswerBody text={LINKEDIN_STYLE_ANSWER} />);

    expect(container.querySelectorAll("ul > li").length).toBe(3);
    expect(container.querySelector("li > ul > li")?.textContent).toContain("Building GenAI-powered products");
  });

  it("renders numbered lists on their own lines instead of run together", () => {
    const { container } = render(<AnswerBody text={"Top artists:\n\n1. The Weeknd\n2. Drake\n3. Eminem"} />);

    const items = [...container.querySelectorAll("ol > li")].map((li) => li.textContent);
    expect(items).toEqual(["The Weeknd", "Drake", "Eminem"]);
  });

  it("never puts a list or table inside a paragraph", () => {
    const { container } = render(<AnswerBody text={LINKEDIN_STYLE_ANSWER + "\n\n| a | b |\n|---|---|\n| 1 | 2 |"} />);

    expect(container.querySelector("p ul, p ol, p table")).toBeNull();
  });

  it("renders a GFM table inside a horizontally scrollable wrapper", () => {
    const { container } = render(<AnswerBody text={"| Language | Level |\n|---|---|\n| Python | Expert |"} />);

    expect(container.querySelector(".answer-table > table")).not.toBeNull();
    expect([...container.querySelectorAll("th")].map((th) => th.textContent)).toEqual(["Language", "Level"]);
    expect(container.querySelector("td")?.textContent).toBe("Python");
  });

  it("renders headings", () => {
    const { container } = render(<AnswerBody text={"## Experience\n\nSome text"} />);

    expect(container.querySelector("h2")?.textContent).toBe("Experience");
  });
});

describe("untrusted output", () => {
  it("renders raw HTML as inert text, not as elements", () => {
    const { container } = render(<AnswerBody text={"Hello <img src=x onerror=alert(1)> world"} />);

    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain("<img src=x onerror=alert(1)>");
  });

  it("opens links in a new tab without leaking the opener", () => {
    const { container } = render(<AnswerBody text="[LinkedIn](https://linkedin.com/in/cemkaspi)" />);

    const link = container.querySelector("a");
    expect(link?.getAttribute("href")).toBe("https://linkedin.com/in/cemkaspi");
    expect(link?.getAttribute("target")).toBe("_blank");
    expect(link?.getAttribute("rel")).toBe("noopener noreferrer");
  });

  it("drops javascript: URLs", () => {
    const { container } = render(<AnswerBody text="[click me](javascript:alert(1))" />);

    const href = container.querySelector("a")?.getAttribute("href") ?? "";
    expect(href.toLowerCase()).not.toContain("javascript:");
  });
});

describe("word reveal", () => {
  const TEXT = "First **bold words** then\n\n- item one\n- item two";

  it("does nothing unless asked", () => {
    const { container } = render(<AnswerBody text={TEXT} />);

    expect(container.querySelector(".reveal-word")).toBeNull();
  });

  it("wraps every word in reading order, with a sequential index, and loses no text", () => {
    const plain = render(<AnswerBody text={TEXT} />).container.textContent;
    const { container } = render(<AnswerBody text={TEXT} reveal />);

    const words = [...container.querySelectorAll<HTMLElement>(".reveal-word")];
    expect(words.map((word) => word.textContent)).toEqual([
      "First",
      "bold",
      "words",
      "then",
      "item",
      "one",
      "item",
      "two",
    ]);
    expect(words.map((word) => word.style.getPropertyValue("--i"))).toEqual(["0", "1", "2", "3", "4", "5", "6", "7"]);
    expect(container.textContent).toBe(plain);
  });

  it("reveals inside formatting, so bold and lists animate too", () => {
    const { container } = render(<AnswerBody text={TEXT} reveal />);

    expect(container.querySelectorAll("strong .reveal-word").length).toBe(2);
    expect(container.querySelectorAll("li .reveal-word").length).toBe(4);
  });

  it("leaves code whole", () => {
    const { container } = render(<AnswerBody text={"Run `npm run dev` now\n\n```\nnpm install\nnpm test\n```"} reveal />);

    expect(container.querySelector("code .reveal-word")).toBeNull();
    expect(container.querySelector("pre")?.textContent).toContain("npm install\nnpm test");
  });

  it("sets the stagger step on the container", () => {
    const { container } = render(<AnswerBody text={TEXT} reveal />);

    const step = (container.firstElementChild as HTMLElement).style.getPropertyValue("--reveal-step");
    expect(step).toMatch(/^0\.\d+s$/);
  });

  it("is skipped when the visitor prefers reduced motion", () => {
    motion.reduced = true;
    const { container } = render(<AnswerBody text={TEXT} reveal />);

    expect(container.querySelector(".reveal-word")).toBeNull();
    expect(container.textContent).toContain("item two");
  });

  it("skips to the end when the answer is clicked", () => {
    const { container } = render(<AnswerBody text={TEXT} reveal />);
    expect(container.querySelector(".reveal-word")).not.toBeNull();

    fireEvent.click(container.firstElementChild as HTMLElement);

    expect(container.querySelector(".reveal-word")).toBeNull();
    expect(container.textContent).toContain("item two");
  });
});
