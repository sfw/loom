import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ActivityBar from "./ActivityBar";

describe("ActivityBar", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders an idle scanner when no backend work is active", () => {
    render(
      <ActivityBar
        active={false}
        mode="idle"
        label="Idle"
      />,
    );

    const activityBar = screen.getByTestId("desktop-activity-bar");
    expect(activityBar).toHaveAttribute("data-active", "false");
    expect(activityBar).toHaveAttribute("data-visual-active", "false");

    const segments = activityBar.querySelectorAll("[data-segment-state]");
    expect(segments).toHaveLength(8);
    for (const segment of segments) {
      expect(segment).toHaveAttribute("data-segment-state", "idle");
    }
  });

  it("holds the active visual state briefly after work settles", () => {
    const { rerender } = render(
      <ActivityBar
        active
        mode="run"
        label="1 active run"
      />,
    );

    const activityBar = screen.getByTestId("desktop-activity-bar");
    expect(activityBar).toHaveAttribute("data-visual-active", "true");

    rerender(
      <ActivityBar
        active={false}
        mode="idle"
        label="Idle"
      />,
    );

    expect(activityBar).toHaveAttribute("data-visual-active", "true");

    act(() => {
      vi.advanceTimersByTime(299);
    });
    expect(activityBar).toHaveAttribute("data-visual-active", "true");

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(activityBar).toHaveAttribute("data-visual-active", "false");
  });
});
