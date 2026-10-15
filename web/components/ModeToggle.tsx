"use client";

import { Moon, Sun } from "@phosphor-icons/react";

import { chooseTheme, latestTheme, useActiveTheme } from "@/themes/client";
import { counterpart, schemeOf } from "@/themes/registry";

/**
 * The light/dark switch. It is always in the DOM and `hidden` unless a theme
 * reveals it (the Aurora pair does), the same way the profile rail is: nothing
 * about the layout is decided in JavaScript, so it cannot pop in after
 * hydration. Which icon shows is also CSS, from the theme on <html>. Only the
 * pressed state follows the theme through React; the click asks for the theme
 * at that moment (including one still fading in), so it is right even before a
 * re-render catches up.
 */
export function ModeToggle() {
  const active = useActiveTheme();

  return (
    <button
      type="button"
      className="mode-toggle pill hidden h-9 w-9 shrink-0 items-center justify-center transition-colors"
      aria-label="Dark mode"
      aria-pressed={schemeOf(active) === "dark"}
      onClick={() => {
        const other = counterpart(latestTheme());
        if (other) chooseTheme(other, "if-present");
      }}
    >
      <Moon className="mode-icon-moon" size={17} weight="fill" aria-hidden />
      <Sun className="mode-icon-sun" size={17} weight="fill" aria-hidden />
    </button>
  );
}
