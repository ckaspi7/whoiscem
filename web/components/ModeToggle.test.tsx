import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { THEME_STORAGE_KEY } from "@/themes/registry";

import { ModeToggle } from "./ModeToggle";

const root = document.documentElement;

describe("ModeToggle", () => {
  beforeEach(() => {
    root.dataset.theme = "aurora";
    window.localStorage.clear();
    window.history.replaceState(null, "", "/");
  });

  it("is a button for dark mode, pressed only on the dark half of the pair", async () => {
    render(<ModeToggle />);
    const button = screen.getByRole("button", { name: "Dark mode", hidden: true });
    expect(button).toHaveAttribute("aria-pressed", "false");

    root.dataset.theme = "aurora-night";
    await waitFor(() => expect(button).toHaveAttribute("aria-pressed", "true"));
  });

  it("switches to the other half, remembers it, and leaves a clean URL clean", () => {
    render(<ModeToggle />);
    fireEvent.click(screen.getByRole("button", { name: "Dark mode", hidden: true }));

    expect(root.dataset.theme).toBe("aurora-night");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("aurora-night");
    expect(window.location.search).toBe("");

    fireEvent.click(screen.getByRole("button", { name: "Dark mode", hidden: true }));
    expect(root.dataset.theme).toBe("aurora");
  });

  it("keeps ?theme= in step when the URL already has one, so a reload does not undo the switch", () => {
    window.history.replaceState(null, "", "/?theme=aurora");
    render(<ModeToggle />);
    fireEvent.click(screen.getByRole("button", { name: "Dark mode", hidden: true }));

    expect(window.location.search).toBe("?theme=aurora-night");
  });

  it("does nothing on a theme that has no other half", () => {
    root.dataset.theme = "porcelain";
    render(<ModeToggle />);
    fireEvent.click(screen.getByRole("button", { name: "Dark mode", hidden: true }));

    expect(root.dataset.theme).toBe("porcelain");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });
});
