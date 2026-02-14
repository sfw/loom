"""Tests for conversation and feedback schemas and API client methods."""

from __future__ import annotations

from loom.api.schemas import ConversationMessageRequest, FeedbackRequest


class TestConversationMessageRequest:
    def test_defaults(self):
        msg = ConversationMessageRequest(message="Hello")
        assert msg.message == "Hello"
        assert msg.role == "user"

    def test_custom_role(self):
        msg = ConversationMessageRequest(message="Status update", role="system")
        assert msg.role == "system"


class TestFeedbackRequest:
    def test_basic(self):
        fb = FeedbackRequest(feedback="Great work!")
        assert fb.feedback == "Great work!"
        assert fb.subtask_id is None

    def test_with_subtask(self):
        fb = FeedbackRequest(feedback="Needs improvement", subtask_id="step-1")
        assert fb.subtask_id == "step-1"
