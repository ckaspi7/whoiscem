import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { chooseTheme, latestTheme } from "./client";
import { THEME_STORAGE_KEY } from "./registry";

const root = document.documentElement;

/** A browser whose View Transitions API holds each update until the test runs it. */
function withViewTransitions() {
  const updates: Array<() => void> = [];
  Object.defineProperty(document, "startViewTransition", {
    configurable: true,
    writable: true,
    value: vi.fn((update: () => void) => {
      updates.push(update);
    }),
  });
  return { updates, start: document.startViewTransition as unknown as ReturnType<typeof vi.fn> };
}

function reducedMotion(reduce: boolean) {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: (query: string) => ({ matches: reduce && query.includes("reduce") }),
  });
}

describe("chooseTheme", () => {
  beforeEach(() => {
    root.dataset.theme = "aurora";
    window.localStorage.clear();
    window.history.replaceState(null, "", "/");
  });

  afterEach(() => {
    Reflect.deleteProperty(document, "startViewTransition");
    Reflect.deleteProperty(window, "matchMedia");
  });

  it("switches at once where there is no View Transitions API", () => {
    chooseTheme("aurora-night", "if-present");

    expect(root.dataset.theme).toBe("aurora-night");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("aurora-night");
  });

  it("hands the change to a view transition, so the page can cross-fade, instead of making it itself", () => {
    const { updates, start } = withViewTransitions();
    reducedMotion(false);

    chooseTheme("aurora-night", "if-present");

    expect(start).toHaveBeenCalledTimes(1);
    expect(root.dataset.theme).toBe("aurora");

    updates[0]();
    expect(root.dataset.theme).toBe("aurora-night");
  });

  it("remembers the choice straight away, even while the fade has not started", () => {
    withViewTransitions();

    chooseTheme("aurora-night", "if-present");

    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("aurora-night");
    expect(latestTheme()).toBe("aurora-night");
  });

  it("does not fade for a visitor who asked for reduced motion", () => {
    const { start } = withViewTransitions();
    reducedMotion(true);

    chooseTheme("aurora-night", "if-present");

    expect(start).not.toHaveBeenCalled();
    expect(root.dataset.theme).toBe("aurora-night");
  });

  it("ends on the last choice when two are made before the first has been applied", () => {
    const { updates } = withViewTransitions();

    chooseTheme("aurora-night", "if-present");
    chooseTheme("aurora", "if-present");
    expect(latestTheme()).toBe("aurora");

    for (const update of updates) update();
    expect(root.dataset.theme).toBe("aurora");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("aurora");
  });

  it("stops reporting a choice as pending once it is on the page", () => {
    const { updates } = withViewTransitions();

    chooseTheme("aurora-night", "if-present");
    updates[0]();
    root.dataset.theme = "porcelain";

    expect(latestTheme()).toBe("porcelain");
  });
});
