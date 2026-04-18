import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppProvider, shallowEqual, useAppSelector } from "./AppContext";

const stateMocks = vi.hoisted(() => ({
  useAppState: vi.fn(),
}));

vi.mock("../hooks/useAppState", () => ({
  useAppState: stateMocks.useAppState,
}));

describe("AppContext selectors", () => {
  let appValue: any;

  beforeEach(() => {
    appValue = {
      error: "",
      selectedWorkspaceFilePath: "",
      setError: vi.fn(),
      setSelectedWorkspaceFilePath: vi.fn(),
    };
    stateMocks.useAppState.mockImplementation(() => appValue);
  });

  it("keeps unrelated selector consumers from rerendering", () => {
    const errorRenders = vi.fn();
    const fileRenders = vi.fn();

    function ErrorProbe() {
      const selection = useAppSelector((state) => ({
        error: state.error,
      }), shallowEqual);
      errorRenders(selection.error);
      return null;
    }

    function FileProbe() {
      const selection = useAppSelector((state) => ({
        selectedWorkspaceFilePath: state.selectedWorkspaceFilePath,
      }), shallowEqual);
      fileRenders(selection.selectedWorkspaceFilePath);
      return null;
    }

    const probes = (
      <>
        <ErrorProbe />
        <FileProbe />
      </>
    );

    const view = render(<AppProvider>{probes}</AppProvider>);

    expect(errorRenders).toHaveBeenCalledTimes(1);
    expect(fileRenders).toHaveBeenCalledTimes(1);

    appValue = {
      ...appValue,
      selectedWorkspaceFilePath: "notes.md",
    };
    view.rerender(<AppProvider>{probes}</AppProvider>);

    expect(errorRenders).toHaveBeenCalledTimes(1);
    expect(fileRenders).toHaveBeenCalledTimes(2);
  });
});
