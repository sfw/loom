"""Learned-pattern review and deletion flows."""

from __future__ import annotations

from loom.tui.screens import LearnedScreen
from loom.tui.widgets import ChatLog


async def show_learned_patterns(self) -> None:
    """Show the learned behavioral patterns review modal."""
    from loom.learning.manager import LearningManager

    mgr = LearningManager(self._db)
    patterns = await mgr.query_behavioral(limit=50)

    def handle_result(result: str) -> None:
        if result:
            self._delete_learned_patterns(result)

    self.push_screen(LearnedScreen(patterns), callback=handle_result)


async def delete_learned_patterns(self, deleted_ids_csv: str) -> None:
    """Delete patterns whose IDs were selected in the review screen."""
    from loom.learning.manager import LearningManager

    if not deleted_ids_csv or not self._db:
        return

    mgr = LearningManager(self._db)
    chat = self.query_one("#chat-log", ChatLog)
    count = 0

    for raw_id in deleted_ids_csv.split(","):
        raw_id = raw_id.strip()
        if not raw_id:
            continue
        try:
            pid = int(raw_id)
            if await mgr.delete_pattern(pid):
                count += 1
        except (ValueError, Exception):
            pass

    if count:
        chat.add_info(f"Deleted {count} learned pattern(s).")
