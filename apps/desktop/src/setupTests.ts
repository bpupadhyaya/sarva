import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Without Vitest's `globals: true`, Testing Library's auto-cleanup can't
// find a global `afterEach` to hook into — without this, each render()
// leaves its DOM tree mounted and the next test's queries see duplicates.
afterEach(() => {
  cleanup();
});
