"""Tests for conversation and feedback schemas and API client methods."""

from __future__ import annotations

from loom.api.schemas import ConversationMessageRequest, FeedbackRequest


class TestConversationMessageRequest:
    def test_defaults(self):
        msg = ConversationMessageRequest(message="Hello")
        assert msg.message == "Hello"
        assert msg.role == "user"
        assert msg.workspace_paths == []
        assert msg.content_blocks == []

    def test_custom_role(self):
        msg = ConversationMessageRequest(message="Status update", role="system")
        assert msg.role == "system"

    def test_attachment_fields(self):
        msg = ConversationMessageRequest(
            workspace_paths=["src/app.tsx"],
            workspace_files=["src/app.tsx"],
            workspace_directories=["src"],
            content_blocks=[{"type": "image", "source_path": "/tmp/pasted.png"}],
        )
        assert msg.message == ""
        assert msg.workspace_paths == ["src/app.tsx"]
        assert msg.workspace_files == ["src/app.tsx"]
        assert msg.workspace_directories == ["src"]
        assert msg.content_blocks == [{"type": "image", "source_path": "/tmp/pasted.png"}]


class TestFeedbackRequest:
    def test_basic(self):
        fb = FeedbackRequest(feedback="Great work!")
        assert fb.feedback == "Great work!"
        assert fb.subtask_id is None

    def test_with_subtask(self):
        fb = FeedbackRequest(feedback="Needs improvement", subtask_id="step-1")
        assert fb.subtask_id == "step-1"
