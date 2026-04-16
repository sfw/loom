import type { WorkspaceArtifact, WorkspaceFileEntry } from "./api";

export type WorkspaceAttachmentOption = {
  path: string;
  isDir: boolean;
  source: "workspace" | "artifact";
  recency: number;
};

export function normalizeWorkspaceAttachmentPath(value: string): string {
  return String(value || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "").trim();
}

export function workspaceAttachmentName(path: string): string {
  const parts = normalizeWorkspaceAttachmentPath(path).split("/").filter(Boolean);
  return parts[parts.length - 1] || path;
}

export function isHiddenWorkspaceAttachmentPath(path: string): boolean {
  return normalizeWorkspaceAttachmentPath(path)
    .split("/")
    .filter(Boolean)
    .some((segment) => segment.startsWith("."));
}

function normalizeWorkspaceAttachmentQuery(value: string): string {
  return String(value || "").trim().toLowerCase();
}

export function buildWorkspaceAttachmentOptions(args: {
  workspaceEntries: WorkspaceFileEntry[];
  recentArtifacts: WorkspaceArtifact[];
}): WorkspaceAttachmentOption[] {
  const byPath = new Map<string, WorkspaceAttachmentOption>();

  for (const entry of args.workspaceEntries) {
    const cleanPath = normalizeWorkspaceAttachmentPath(entry.path);
    if (!cleanPath || isHiddenWorkspaceAttachmentPath(cleanPath)) {
      continue;
    }
    byPath.set(cleanPath, {
      path: cleanPath,
      isDir: Boolean(entry.is_dir),
      source: "workspace",
      recency: Number.MAX_SAFE_INTEGER,
    });
  }

  for (const [artifactIndex, artifact] of args.recentArtifacts.entries()) {
    const cleanPath = normalizeWorkspaceAttachmentPath(artifact.path);
    if (!cleanPath || isHiddenWorkspaceAttachmentPath(cleanPath)) {
      continue;
    }
    const existingArtifact = byPath.get(cleanPath);
    byPath.set(cleanPath, {
      path: cleanPath,
      isDir: false,
      source: "artifact",
      recency: Math.min(existingArtifact?.recency ?? Number.MAX_SAFE_INTEGER, artifactIndex),
    });
    const parts = cleanPath.split("/").filter(Boolean);
    for (let index = 1; index < parts.length; index += 1) {
      const dirPath = parts.slice(0, index).join("/");
      if (isHiddenWorkspaceAttachmentPath(dirPath)) {
        continue;
      }
      const existingDirectory = byPath.get(dirPath);
      byPath.set(dirPath, {
        path: dirPath,
        isDir: true,
        source: "artifact",
        recency: Math.min(existingDirectory?.recency ?? Number.MAX_SAFE_INTEGER, artifactIndex + 0.5),
      });
    }
  }

  return Array.from(byPath.values());
}

export function rankWorkspaceAttachmentSuggestions(args: {
  options: WorkspaceAttachmentOption[];
  query: string;
  selectedPaths?: string[];
  limit?: number;
}): WorkspaceAttachmentOption[] {
  const query = normalizeWorkspaceAttachmentQuery(args.query);
  const selectedPaths = new Set(args.selectedPaths ?? []);
  const limit = Math.max(1, args.limit ?? (query ? 24 : 18));

  const pathNameRank = (option: WorkspaceAttachmentOption) => {
    if (!query) return 3;
    const name = normalizeWorkspaceAttachmentQuery(workspaceAttachmentName(option.path));
    const path = normalizeWorkspaceAttachmentQuery(option.path);
    if (name.startsWith(query)) return 0;
    if (name.includes(query)) return 1;
    if (path.includes(query)) return 2;
    return 3;
  };

  return args.options
    .filter((option) => !selectedPaths.has(option.path))
    .filter((option) => {
      if (!query) return true;
      const haystack = `${normalizeWorkspaceAttachmentQuery(option.path)} ${normalizeWorkspaceAttachmentQuery(workspaceAttachmentName(option.path))}`;
      return haystack.includes(query);
    })
    .sort((left, right) => {
      const leftRank = pathNameRank(left);
      const rightRank = pathNameRank(right);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      if (left.source !== right.source) {
        return left.source === "artifact" ? -1 : 1;
      }
      if (left.isDir !== right.isDir) {
        return left.isDir ? -1 : 1;
      }
      if (left.recency !== right.recency) {
        return left.recency - right.recency;
      }
      return left.path.localeCompare(right.path);
    })
    .slice(0, limit);
}
