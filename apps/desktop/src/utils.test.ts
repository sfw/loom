import { describe, expect, it } from "vitest";

import { isTransientRequestError } from "./utils";

describe("isTransientRequestError", () => {
  it("treats desktop request timeout messages as transient", () => {
    expect(isTransientRequestError(new Error("Request timed out after 15000ms"))).toBe(true);
    expect(isTransientRequestError(new Error("Request timed out after 20000ms"))).toBe(true);
  });

  it("does not hide ordinary application errors", () => {
    expect(isTransientRequestError(new Error("400 Bad Request"))).toBe(false);
    expect(isTransientRequestError(new Error("conversation not found"))).toBe(false);
  });
});
