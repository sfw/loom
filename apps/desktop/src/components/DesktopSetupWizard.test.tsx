import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import DesktopSetupWizard from "./DesktopSetupWizard";

const setupStatus = {
  needs_setup: true,
  config_path: "/Users/test/.loom/loom.toml",
  providers: [
    {
      display_name: "Anthropic (Claude API)",
      provider_key: "anthropic",
      needs_api_key: true,
      default_base_url: "https://api.anthropic.com",
    },
    {
      display_name: "Ollama",
      provider_key: "ollama",
      needs_api_key: false,
      default_base_url: "http://localhost:11434",
    },
  ],
  role_presets: {
    all: ["planner", "executor", "extractor", "verifier", "compactor"],
    primary: ["planner", "executor"],
    utility: ["extractor", "verifier", "compactor"],
  },
};

describe("DesktopSetupWizard", () => {
  it("discovers a model and completes first-run setup", async () => {
    const user = userEvent.setup();
    const onDiscoverModels = vi.fn().mockResolvedValue(["claude-sonnet-4-5"]);
    const onCompleteSetup = vi.fn().mockResolvedValue(undefined);

    render(
      <DesktopSetupWizard
        status={setupStatus}
        onDiscoverModels={onDiscoverModels}
        onCompleteSetup={onCompleteSetup}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByLabelText("API Key"), "sk-test");
    await user.click(screen.getByRole("button", { name: "Discover models" }));

    await waitFor(() => expect(onDiscoverModels).toHaveBeenCalledWith(
      "anthropic",
      "https://api.anthropic.com",
      "sk-test",
    ));

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: /All roles/ }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Save and start Loom" }));

    await waitFor(() => expect(onCompleteSetup).toHaveBeenCalledWith({
      models: [
        expect.objectContaining({
          name: "primary",
          provider: "anthropic",
          model: "claude-sonnet-4-5",
          roles: ["planner", "executor", "extractor", "verifier", "compactor"],
        }),
      ],
    }));
  });

  it("offers a utility model when the primary model only covers primary roles", async () => {
    const user = userEvent.setup();

    render(
      <DesktopSetupWizard
        status={setupStatus}
        onDiscoverModels={vi.fn().mockResolvedValue(["claude-sonnet-4-5"])}
        onCompleteSetup={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByLabelText("API Key"), "sk-test");
    await user.type(screen.getByLabelText("Model Name"), "claude-sonnet-4-5");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: /Primary only/ }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getByText(/Remaining roles:/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add utility model" })).toBeInTheDocument();
  });
});
