import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

type ActivityMode = "idle" | "thread" | "run" | "mixed";

const DOT_COUNT = 8;
const FRAME_INTERVAL_MS = 90;
const IDLE_HOLD_MS = 300;

function usePrefersReducedMotion(): boolean {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return;
    }

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => {
      setPrefersReducedMotion(mediaQuery.matches);
    };

    update();
    mediaQuery.addEventListener("change", update);
    return () => {
      mediaQuery.removeEventListener("change", update);
    };
  }, []);

  return prefersReducedMotion;
}

export default function ActivityBar({
  active,
  mode,
  label,
  testId = "desktop-activity-bar",
}: {
  active: boolean;
  mode: ActivityMode;
  label: string;
  testId?: string;
}) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [frameIndex, setFrameIndex] = useState(0);
  const [visualActive, setVisualActive] = useState(active);
  const directionRef = useRef<1 | -1>(1);
  const holdTimeoutRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (holdTimeoutRef.current !== null) {
        window.clearTimeout(holdTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (holdTimeoutRef.current !== null) {
      window.clearTimeout(holdTimeoutRef.current);
      holdTimeoutRef.current = null;
    }

    if (active) {
      setVisualActive(true);
      return;
    }

    if (!visualActive) {
      directionRef.current = 1;
      setFrameIndex(0);
      return;
    }

    holdTimeoutRef.current = window.setTimeout(() => {
      holdTimeoutRef.current = null;
      directionRef.current = 1;
      setFrameIndex(0);
      setVisualActive(false);
    }, IDLE_HOLD_MS);
  }, [active, visualActive]);

  useEffect(() => {
    if (!visualActive || prefersReducedMotion) {
      return;
    }

    const intervalId = window.setInterval(() => {
      setFrameIndex((current) => {
        let next = current + directionRef.current;
        if (next >= DOT_COUNT) {
          directionRef.current = -1;
          next = DOT_COUNT - 2;
        } else if (next < 0) {
          directionRef.current = 1;
          next = 1;
        }
        return Math.max(0, Math.min(next, DOT_COUNT - 1));
      });
    }, FRAME_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [prefersReducedMotion, visualActive]);

  const activeIndex = prefersReducedMotion ? Math.floor(DOT_COUNT / 2) : frameIndex;
  const trailColorClass = mode === "thread"
    ? "bg-[#8a9a7b]/85"
    : mode === "run"
      ? "bg-sky-400/85"
      : "bg-cyan-400/85";
  const headColorClass = mode === "thread"
    ? "bg-[#d2e0c8] shadow-[0_0_14px_rgba(190,200,180,0.5)]"
    : mode === "run"
      ? "bg-sky-300 shadow-[0_0_14px_rgba(56,189,248,0.55)]"
      : "bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.6)]";
  const shellClass = mode === "thread"
    ? "border-[#8a9a7b]/20 bg-[#8a9a7b]/[0.06]"
    : mode === "run"
      ? "border-sky-500/20 bg-sky-500/[0.06]"
      : mode === "mixed"
        ? "border-cyan-500/20 bg-cyan-500/[0.06]"
        : "border-zinc-800/80 bg-zinc-900/70";
  const glowClass = mode === "thread"
    ? "bg-[#8a9a7b]/20"
    : mode === "run"
      ? "bg-sky-500/20"
      : "bg-cyan-500/20";

  return (
    <div
      className="relative"
      title={label}
      aria-label={label}
      data-testid={testId}
      data-active={active ? "true" : "false"}
      data-visual-active={visualActive ? "true" : "false"}
      data-mode={mode}
    >
      <div
        className={cn(
          "pointer-events-none absolute inset-x-1 inset-y-0 rounded-full blur-md transition-opacity duration-200",
          glowClass,
          visualActive ? "opacity-100" : "opacity-0",
        )}
      />
      <div
        className={cn(
          "relative flex items-center gap-1 rounded-full border px-2.5 py-1",
          "backdrop-blur-sm transition-colors duration-200",
          shellClass,
        )}
      >
        {Array.from({ length: DOT_COUNT }, (_, index) => {
          const isHead = visualActive && index === activeIndex;
          const isTrail = visualActive && !isHead && Math.abs(index - activeIndex) === 1;
          const segmentState = isHead ? "head" : isTrail ? "trail" : "idle";

          return (
            <span
              // This compact scanner intentionally mirrors the TUI rhythm.
              key={index}
              data-segment-state={segmentState}
              className={cn(
                "block h-2 w-1 rounded-full transition-all duration-150",
                "origin-center",
                isHead
                  ? cn(headColorClass, prefersReducedMotion ? "" : "scale-y-110")
                  : isTrail
                    ? trailColorClass
                    : "bg-zinc-700/75",
              )}
            />
          );
        })}
      </div>
    </div>
  );
}
