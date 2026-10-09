import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library only registers its own unmount-after-each-test when the test
// runner exposes globals, which this config does not. Without it a component
// rendered in one test stays mounted into the next, so screen queries find
// duplicates and a still-mounted hook keeps writing to sessionStorage.
afterEach(cleanup);
