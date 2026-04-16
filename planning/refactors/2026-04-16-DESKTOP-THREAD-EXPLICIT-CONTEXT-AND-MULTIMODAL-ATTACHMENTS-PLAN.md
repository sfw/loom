# Desktop Thread Explicit Context And Multimodal Attachments Plan (2026-04-16)

## Objective
Add first-class explicit context to desktop thread messages so users can:
1. attach workspace files and folders from the composer with an `@` picker
2. attach pasted images as actual message content instead of preview-only local UI state
3. send those attachments through the conversation API as structured context, not text sugar
4. preserve this context in thread replay/history without introducing avoidable schema churn

## Executive Summary
The desktop thread composer is currently halfway between text chat and a richer multimodal surface:
1. the UI already lets users paste images into the composer, but those images are not sent with the message
2. the thread transcript can already linkify file paths after the fact
3. the run launcher already has an explicit workspace context model for files and folders
4. the engine already has multimodal content blocks for tool results

The missing piece is the user-message path. We should extend the conversation message contract so a user turn can carry:
1. plain text
2. attached workspace-relative paths
3. attached local image blocks

This should be implemented as a focused patch, not a broad redesign:
1. reuse the existing desktop workspace-path suggestion patterns from the run launcher
2. reuse the existing content-block infrastructure where possible
3. keep database changes optional, and prefer storing attachment metadata in existing JSON-bearing surfaces first
4. make the API backward compatible so existing text-only clients continue to work

## Repo-Accurate Current State
1. `apps/desktop/src/components/ThreadsTab.tsx` uses a plain `<textarea>` composer with no mention or attachment model for workspace paths.
2. `apps/desktop/src/components/ThreadsTab.tsx:1286-1306` shows pasted-image previews and explicitly says `vision support pending`, which confirms the images are not actually attached to outgoing messages.
3. `apps/desktop/src/api.ts:1397-1409` sends conversation messages as `{ message, role }`.
4. `src/loom/api/schemas.py:38-40` defines `ConversationMessageRequest` as only `message` and `role`.
5. `src/loom/api/routes.py:7726-7759` accepts that text-only payload and starts the cowork turn with `user_message=message`.
6. `src/loom/api/routes.py:3599-3607` records replay `user_message` events with only `{"text": user_message}`.
7. `apps/desktop/src/components/RunsTab.tsx:718-805` already has a good local model for workspace file/folder suggestions and ranking.
8. `apps/desktop/src/components/RunsTab.tsx:759-767` already serializes explicit workspace context as:
   - `workspace_paths`
   - `workspace_files`
   - `workspace_directories`
9. `src/loom/read_scope.py:36-55` already normalizes and iterates explicit attached workspace paths.
10. `src/loom/content.py` and the model providers already support multimodal content blocks, but that path is currently used for tool results, not user-authored messages.

## User-Facing Problem
Today a desktop user can do all of the following:
1. mention a file path manually as text
2. click file paths in transcript output after the agent mentions them
3. paste an image and see a local preview
4. attach workspace context when launching a run

But they cannot do the most useful thread-level operation cleanly:
1. select files and folders inline from the thread composer as explicit context
2. send pasted images as actual multimodal message content
3. rely on those attachments surviving the round-trip as structured data

That gap creates ambiguity for both UX and execution:
1. the user thinks they attached something
2. the API treats the message as plain text
3. the engine loses the distinction between ordinary prose and user-scoped context

## Goals
1. Make workspace path attachments explicit and structured in thread chat.
2. Make pasted images real message attachments in thread chat.
3. Keep the UX lightweight and composer-native.
4. Preserve backward compatibility for existing API clients.
5. Avoid database schema changes unless we discover they are necessary for correctness.
6. Reuse existing run-launch and multimodal primitives rather than inventing parallel systems.

## Non-Goals
1. Rebuilding the entire desktop composer as a rich text editor.
2. General-purpose arbitrary file upload for every file type in this patch.
3. A full redesign of transcript rendering.
4. Reworking the model provider stack beyond what is needed for user-authored content blocks.
5. Changing run-launch explicit context behavior in this patch, except to share utilities.

## Design Principles
1. `@` is a discovery affordance, not the data model.
2. Structured attachments should be transmitted separately from message text.
3. Workspace file/folder attachments should remain workspace-relative and normalized.
4. Image attachments should use the existing content-block system.
5. Desktop replay/history should preserve enough attachment metadata to explain what the user sent.
6. Text-only fallback must remain valid for non-vision models and old clients.
7. The implementation should degrade gracefully when a workspace path disappears between selection and send.

## Proposed Contract

### 1. Conversation Message Request
Extend `ConversationMessageRequest` to allow structured optional attachments while keeping `message` as the canonical text field.

Proposed shape:
1. `message: str`
2. `role: str = "user"`
3. `workspace_paths: list[str] = []`
4. `workspace_files: list[str] = []`
5. `workspace_directories: list[str] = []`
6. `content_blocks: list[ContentBlockResponse] = []`

Notes:
1. `workspace_*` mirrors the existing run-launch contract so the semantics stay consistent.
2. `content_blocks` should initially support image blocks for desktop-authored user content.
3. Existing clients can keep sending just `message` and `role`.

### 2. Attachment Semantics
1. Workspace file/folder attachments are explicit read scope hints.
2. Pasted images are explicit multimodal user inputs.
3. The visible message text remains user-authored prose, not an encoded attachment blob.
4. The desktop may optionally insert inline visual chips or code tokens in the draft, but the authoritative data should be the structured attachment payload.

### 3. Replay/History Semantics
When a user message is appended to the replay journal, include:
1. `text`
2. normalized workspace path attachments
3. serialized user content blocks

This gives the desktop enough information to:
1. show attachment chips in history/transcript
2. reopen linked workspace files
3. indicate that images were actually attached, not merely previewed locally

## Recommended Persistence Strategy

### Preferred Path: No DB Migration
Use existing JSON-bearing payload surfaces first:
1. keep `conversation_turns.content` as plain text
2. store attachment metadata in replay journal payloads
3. if needed, store lightweight attachment metadata in `session_state.ui_state` for desktop convenience

Why this is attractive:
1. no schema migration
2. no risk to existing conversation-turn queries
3. the transcript path already leans on replay events
4. user-facing attachment rendering belongs more naturally in replay events than in the old text-only turn table

### Fallback Path: Add Attachment Metadata To Conversation Turns
Only if we discover a correctness gap in replay-only persistence should we add a DB column such as `metadata TEXT` or `content_blocks TEXT` to `conversation_turns`.

If we do that, we must follow the repository migration policy:
1. update `src/loom/state/schema.sql`
2. update `src/loom/state/schema/base.sql`
3. add a migration step under `src/loom/state/migrations/steps/`
4. register it in `src/loom/state/migrations/registry.py`
5. add migration tests and docs

Current recommendation: avoid this unless replay-only persistence proves insufficient.

## Target UX

### 1. Workspace Path Picker
In `ThreadsTab`:
1. typing `@` at a valid trigger location opens a workspace path picker popover
2. results include both files and directories
3. results are ranked similarly to run-launch suggestions:
   - exact filename prefix
   - filename contains
   - path contains
   - recent artifact-backed paths ahead of generic workspace paths
4. arrow keys navigate the picker
5. `Enter` or `Tab` inserts/selects the highlighted option
6. `Escape` closes the picker
7. selected workspace attachments are shown as removable chips near the composer

### 2. Pasted Images
1. keep the existing preview row
2. remove the misleading `vision support pending` text once wired
3. treat pasted images as pending image content blocks for the next send
4. show removable chips or thumbnails before submit

### 3. Transcript Rendering
1. user messages with workspace attachments should show lightweight chips or indicators
2. image attachments should show an indicator such as `1 image attached`
3. clicking a workspace path chip should route to the Files tab
4. the transcript should still render sensibly for text-only historical turns

## Architecture Workstreams

## Workstream 0: Shared Attachment Model

### Problem
Run launch and thread chat both need explicit workspace path attachments, but today they use separate UX paths and only run launch serializes structured workspace context.

### Plan
1. Extract the desktop-side path option model and ranking logic from `RunsTab`.
2. Create a shared helper or hook for:
   - hidden-path filtering
   - path normalization
   - artifact-vs-workspace ranking
   - filename/path matching
3. Reuse this helper in both the run launcher and the thread composer.

### Primary Files
1. `apps/desktop/src/components/RunsTab.tsx`
2. new shared helper under `apps/desktop/src/` or `apps/desktop/src/hooks/`
3. related run and thread tests

### Acceptance
1. Thread and run attachment suggestions rank the same way for the same workspace inputs.
2. Hidden files remain excluded consistently.

## Workstream 1: Conversation API Contract

### Problem
The conversation API only accepts text, so explicit context and images have nowhere structured to go.

### Plan
1. Extend `ConversationMessageRequest` with optional attachment fields.
2. Keep the endpoint backward compatible.
3. Normalize workspace-relative paths using existing read-scope utilities.
4. Validate image blocks conservatively:
   - allow only supported image block types
   - reject malformed or unsafe absolute-path references
5. Thread the attachment payload into the cowork turn entrypoint.

### Primary Files
1. `src/loom/api/schemas.py`
2. `src/loom/api/routes.py`
3. `apps/desktop/src/api.ts`
4. API tests

### Acceptance
1. Old text-only clients continue to work unchanged.
2. New clients can send workspace path attachments and image content blocks.
3. Invalid attachment payloads fail cleanly with 4xx responses.

## Workstream 2: Cowork Session User-Message Support

### Problem
Multimodal content blocks exist in the system, but the user-message path still enters the model as plain text.

### Plan
1. Add a user-message representation that can carry:
   - message text
   - attachment metadata
   - optional content blocks
2. Reuse `loom.content` serialization instead of inventing a parallel shape.
3. Extend the cowork session message-building path so user-authored image content blocks reach model providers.
4. Make workspace path attachments available to the turn as explicit read-scope hints and operator-visible metadata.
5. Ensure text-only models degrade gracefully:
   - message text remains intact
   - image blocks fall back to text indicators if vision is unavailable

### Primary Files
1. `src/loom/cowork/session.py`
2. `src/loom/content.py`
3. provider adapters under `src/loom/models/`
4. related session/provider tests

### Acceptance
1. Vision-capable models receive user-authored image content blocks.
2. Non-vision models still receive a sensible textual fallback.
3. Workspace path attachments are available as structured context during the turn.

## Workstream 3: Desktop Thread Composer UX

### Problem
The current thread composer has no attachment model beyond local pasted-image preview state.

### Plan
1. Add mention parsing around the caret in `ThreadsTab`.
2. Open a path-picker popover when `@` starts a valid mention token.
3. Maintain a separate attachment state instead of overloading the draft text.
4. Show selected workspace attachments as chips above the composer.
5. Convert pasted images into pending image blocks for the outgoing request.
6. Clear attachment state on successful send and preserve it on failure.
7. Make mention navigation coexist with existing composer key handling:
   - mention picker gets first chance at arrow keys and enter
   - prompt history remains intact when no picker is open

### Primary Files
1. `apps/desktop/src/components/ThreadsTab.tsx`
2. possibly a new `ComposerAttachmentPicker` component
3. desktop thread tests

### Acceptance
1. Typing `@` opens the path picker.
2. A selected file or folder becomes a removable attachment chip.
3. Pasted images become actual outgoing attachments.
4. Prompt history and send shortcuts still work.

## Workstream 4: Replay And Transcript Rendering

### Problem
If structured attachments are sent but not replayed, the thread history will misrepresent what the user actually sent.

### Plan
1. Extend replay `user_message` payloads to include attachment metadata.
2. Surface attachment indicators in the desktop thread transcript.
3. Reuse existing file-navigation affordances when possible.
4. Keep message fallback rendering safe when attachment metadata is absent.

### Primary Files
1. `src/loom/api/routes.py`
2. `apps/desktop/src/history.ts`
3. `apps/desktop/src/conversationTimeline.ts`
4. `apps/desktop/src/components/ThreadsTab.tsx`
5. replay/history tests

### Acceptance
1. Reloading the thread still shows attachment indicators.
2. Workspace path chips remain actionable after reload.
3. Historical text-only messages remain unchanged.

## Open Design Decisions
1. Should selecting an `@` result insert visible inline text, or only create a chip?
   - Recommendation: create a chip and optionally insert a light inline code token only if we want plain-text discoverability.
2. Should folders be expanded to child files at send time?
   - Recommendation: no. Preserve folder attachments as folder-level explicit context, matching run launch.
3. Should the transcript render image thumbnails or just indicators for user-authored images?
   - Recommendation: start with indicators, then add thumbnails if the replay payload and desktop rendering path stay simple.
4. Should we persist attachment metadata in `conversation_turns` immediately?
   - Recommendation: no. Use replay payloads first.

## Risks And Knock-Ons
1. Composer key conflicts could break prompt history or send behavior if mention handling is not carefully layered.
2. Workspace path normalization must reject invalid or escaping relative paths.
3. Desktop-only optimistic state could drift from replay if send failure paths are not handled carefully.
4. Provider behavior for user-authored multimodal blocks may differ from tool-result multimodal blocks and needs test coverage.
5. If transcript history relies on `conversation_turns` in some fallback paths, replay-only attachment metadata may need a small compatibility shim.

## Test Plan

### Desktop
1. `ThreadsTab` opens and navigates the `@` path picker.
2. selecting a file creates an attachment chip
3. selecting a folder creates an attachment chip
4. mention navigation does not break ArrowUp/ArrowDown prompt history
5. pressing Enter with picker open selects the item instead of sending prematurely
6. pasted images are included in the outgoing request payload
7. failed sends preserve attachments and previews
8. successful sends clear attachments

### API
1. text-only message request still works
2. workspace path attachments are normalized and accepted
3. invalid workspace paths are rejected
4. malformed content blocks are rejected
5. replay journal stores user attachment metadata

### Engine / Provider
1. user-authored image blocks reach OpenAI-compatible provider requests
2. user-authored image blocks degrade gracefully for non-vision models
3. workspace path attachments become explicit read-scope hints for the turn

## Recommended Delivery Order
1. shared attachment/path helper extraction
2. conversation API request extension
3. cowork user-message multimodal/context wiring
4. desktop thread composer UX
5. replay/transcript rendering
6. cleanup and polish

## Definition Of Done
1. A desktop user can type `@` in thread chat and attach a workspace file or folder.
2. A desktop user can paste an image and have it sent as an actual attachment.
3. The conversation API carries structured attachment data.
4. The cowork turn can consume that data as explicit context.
5. Thread replay shows that those attachments were part of the sent message.
6. The patch ships without a required DB migration unless testing proves one is necessary.

## Immediate Implementation Notes
1. Start from the existing run-launch explicit context model rather than inventing a new path taxonomy.
2. Start from the existing `loom.content` content-block machinery rather than inventing a new image payload format.
3. Keep the first transcript rendering pass simple: chips and indicators are enough.
4. If replay-only persistence proves sufficient, keep this patch schema-free.
