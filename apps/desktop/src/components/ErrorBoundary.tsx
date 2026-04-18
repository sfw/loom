import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  message: string;
};

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      message: "",
    };
  }

  static getDerivedStateFromError(error: unknown): State {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : "The desktop renderer hit an unexpected error.",
    };
  }

  componentDidCatch(error: unknown, info: ErrorInfo) {
    console.error("Desktop renderer crashed", error, info);
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="flex min-h-screen items-center justify-center bg-[#09090b] px-6 text-zinc-100">
        <div className="w-full max-w-md rounded-2xl border border-zinc-800 bg-[#111114] p-6 shadow-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.22em] text-red-400">
            Renderer Error
          </p>
          <h1 className="mt-3 text-xl font-semibold text-zinc-100">
            Loom Desktop hit a rendering problem.
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-zinc-400">
            The app shell caught the crash so the whole desktop does not go blank. Reload the UI to recover.
          </p>
          {this.state.message && (
            <pre className="mt-4 overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-950/80 px-4 py-3 text-xs text-zinc-400">
              {this.state.message}
            </pre>
          )}
          <button
            type="button"
            onClick={this.handleReload}
            className="mt-5 inline-flex items-center rounded-lg bg-[#6b7a5e] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#8a9a7b]"
          >
            Reload Desktop
          </button>
        </div>
      </div>
    );
  }
}
