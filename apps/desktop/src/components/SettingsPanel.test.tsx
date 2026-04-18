import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SettingsPanel from "./SettingsPanel";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

describe("SettingsPanel", () => {
  beforeEach(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation(() => ({
        matches: true,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      })),
    });

    mockApp = {
      runtime: {
        version: "0.3.0",
        runtime_role: "primary",
        database_path: "/tmp/loom.db",
        config_path: "/tmp/loom.toml",
        host: "127.0.0.1",
        port: 9000,
        started_at: "2026-04-17T19:00:00Z",
      },
      models: [
        {
          name: "primary",
          provider: "openai_compatible",
          model: "gpt-4.1",
          model_id: "gpt-4.1",
          tier: 1,
          roles: ["planner", "executor"],
          temperature: 0.1,
          max_tokens: 8192,
        },
      ],
      settings: {
        basic: [],
        advanced: [],
        updated_at: "",
      },
      updateModelSettings: vi.fn().mockResolvedValue(undefined),
      setError: vi.fn(),
      setNotice: vi.fn(),
    };
  });

  it("allows tuning temperature and max tokens for a model", async () => {
    const user = userEvent.setup();

    render(<SettingsPanel />);

    await user.click(screen.getByRole("button", { name: "Tune" }));
    await user.clear(screen.getByLabelText("primary temperature"));
    await user.type(screen.getByLabelText("primary temperature"), "1");
    await user.clear(screen.getByLabelText("primary max tokens"));
    await user.type(screen.getByLabelText("primary max tokens"), "4096");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(mockApp.updateModelSettings).toHaveBeenCalledWith(
      "primary",
      {
        temperature: 1,
        max_tokens: 4096,
      },
    ));
    expect(mockApp.setNotice).toHaveBeenCalledWith("Updated primary model settings.");
  });
});
