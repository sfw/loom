import { useApp } from "@/context/AppContext";
import { cn } from "@/lib/utils";
import {
  formatDate,
  approvalQuestionOptions,
  approvalQuestionType,
  approvalQuestionContext,
} from "@/utils";
import { Shield, MessageCircle, ChevronRight } from "lucide-react";
import type { ApprovalFeedItem } from "@/api";

function kindBadge(kind: string) {
  switch (kind) {
    case "task_approval":
      return { label: "Task Approval", color: "bg-amber-500/15 text-amber-400 border-amber-500/30" };
    case "task_question":
      return { label: "Question", color: "bg-blue-500/15 text-blue-400 border-blue-500/30" };
    case "conversation_approval":
      return { label: "Conversation", color: "bg-purple-500/15 text-purple-400 border-purple-500/30" };
    default:
      return { label: kind, color: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30" };
  }
}

function ApprovalCard({ item }: { item: ApprovalFeedItem }) {
  const {
    approvalReplyDrafts,
    setApprovalReplyDrafts,
    replyingApprovalId,
    handleReplyApproval,
    handleSelectApprovalContext,
  } = useApp();

  const badge = kindBadge(item.kind);
  const isQuestion = item.kind === "task_question";
  const isConversationApproval = item.kind === "conversation_approval";
  const options = approvalQuestionOptions(item);
  const questionType = approvalQuestionType(item);
  const contextNote = approvalQuestionContext(item);
  const draftText = approvalReplyDrafts[item.id] || "";
  const isReplying = replyingApprovalId === item.id;

  return (
    <div className="rounded-xl border border-zinc-800/60 bg-zinc-900/50 p-4 transition-colors hover:border-zinc-700/60">
      {/* Header row: badge + timestamp */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold",
            badge.color,
          )}
        >
          {isQuestion ? (
            <MessageCircle size={12} className="shrink-0" />
          ) : (
            <Shield size={12} className="shrink-0" />
          )}
          {badge.label}
        </span>
        <span className="text-[11px] text-zinc-500 tabular-nums shrink-0">
          {formatDate(item.created_at)}
        </span>
      </div>

      {/* Title + summary */}
      <h3 className="text-sm font-medium text-zinc-100 mb-1 leading-snug">
        {item.title || "Untitled approval"}
      </h3>
      {item.summary && (
        <p className="text-xs text-zinc-400 mb-2 leading-relaxed">{item.summary}</p>
      )}

      {/* Context note */}
      {contextNote && (
        <p className="text-xs text-zinc-500 italic mb-3">{contextNote}</p>
      )}

      {/* Metadata pills */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {item.tool_name && (
          <span className="rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-medium text-zinc-400">
            {item.tool_name}
          </span>
        )}
        {item.risk_level && (
          <span
            className={cn(
              "rounded-md px-2 py-0.5 text-[10px] font-medium",
              item.risk_level === "high"
                ? "bg-red-500/15 text-red-400"
                : item.risk_level === "medium"
                  ? "bg-amber-500/15 text-amber-400"
                  : "bg-zinc-800/80 text-zinc-400",
            )}
          >
            {item.risk_level}
          </span>
        )}
        {item.run_id && (
          <span className="rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-mono text-zinc-500">
            run:{item.run_id.slice(0, 8)}
          </span>
        )}
        {item.conversation_id && (
          <span className="rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-mono text-zinc-500">
            thread:{item.conversation_id.slice(0, 8)}
          </span>
        )}
      </div>

      {/* Action buttons */}
      {isQuestion ? (
        <div className="space-y-3">
          {/* Quick-reply option buttons */}
          {options.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {options.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  disabled={isReplying}
                  onClick={() =>
                    handleReplyApproval(item, {
                      decision: "answer",
                      response_type: "answered",
                      selected_option_ids: [option.id],
                      selected_labels: [option.label],
                      custom_response: option.label,
                    })
                  }
                  className={cn(
                    "rounded-lg border border-zinc-700/60 bg-zinc-800/60 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors",
                    "hover:border-[#8a9a7b]/40 hover:bg-[#6b7a5e]/10 hover:text-[#bec8b4]",
                    "disabled:opacity-50 disabled:cursor-not-allowed",
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          )}

          {/* Custom reply area */}
          <div className="flex gap-2">
            <textarea
              value={draftText}
              onChange={(e) =>
                setApprovalReplyDrafts((prev) => ({
                  ...prev,
                  [item.id]: e.target.value,
                }))
              }
              placeholder={
                questionType
                  ? `Reply to ${questionType}...`
                  : "Type a custom reply..."
              }
              rows={2}
              className="flex-1 rounded-lg border border-zinc-700/60 bg-zinc-800/40 px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-[#8a9a7b]/50 focus:ring-1 focus:ring-[#8a9a7b]/20 resize-none"
            />
            <button
              type="button"
              disabled={!draftText.trim() || isReplying}
              onClick={() =>
                handleReplyApproval(item, {
                  decision: "answer",
                  response_type: "answered",
                  custom_response: draftText.trim(),
                })
              }
              className={cn(
                "self-end rounded-lg bg-[#6b7a5e] px-4 py-2 text-xs font-semibold text-white transition-colors",
                "hover:bg-[#8a9a7b] disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              Send
            </button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => handleSelectApprovalContext(item)}
            className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700/60 bg-zinc-800/60 px-3 py-1.5 text-xs font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-100"
          >
            Open context
            <ChevronRight size={12} />
          </button>
          <button
            type="button"
            disabled={isReplying}
            onClick={() =>
              handleReplyApproval(item, { decision: "approve" })
            }
            className={cn(
              "rounded-lg bg-emerald-600/80 px-3 py-1.5 text-xs font-semibold text-white transition-colors",
              "hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            Approve
          </button>
          {isConversationApproval && (
            <button
              type="button"
              disabled={isReplying}
              onClick={() =>
                handleReplyApproval(item, { decision: "approve_all" })
              }
              className={cn(
                "rounded-lg border border-emerald-600/40 bg-emerald-600/10 px-3 py-1.5 text-xs font-semibold text-emerald-400 transition-colors",
                "hover:bg-emerald-600/20 disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              Always allow
            </button>
          )}
          <button
            type="button"
            disabled={isReplying}
            onClick={() =>
              handleReplyApproval(item, { decision: "deny" })
            }
            className={cn(
              "rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs font-semibold text-red-400 transition-colors",
              "hover:bg-red-500/20 disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            Deny
          </button>
        </div>
      )}
    </div>
  );
}

export default function InboxTab() {
  const { filteredApprovalItems } = useApp();

  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800/60 px-6 py-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-zinc-100">Inbox</h2>
          {filteredApprovalItems.length > 0 && (
            <span className="flex h-6 min-w-[24px] items-center justify-center rounded-full bg-[#6b7a5e] px-2 text-[11px] font-semibold tabular-nums text-white">
              {filteredApprovalItems.length}
            </span>
          )}
        </div>
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto px-6 py-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-800">
        {filteredApprovalItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-zinc-800/60">
              <Shield size={24} className="text-zinc-600" />
            </div>
            <p className="text-sm font-medium text-zinc-400 mb-1">
              No pending approvals
            </p>
            <p className="text-xs text-zinc-600 max-w-xs">
              When tasks or threads require your approval, they will
              appear here.
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            {filteredApprovalItems.map((item) => (
              <ApprovalCard key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
