"""Tests for workspace management: changelog, snapshots, diff, revert."""

from __future__ import annotations

from pathlib import Path

from loom.tools.workspace import ChangeLog, DiffGenerator, validate_workspace


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create a workspace and data dir."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return workspace, data_dir


class TestChangeLog:
    def test_record_create(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        cl = ChangeLog("t1", workspace, data_dir)

        cl.record_before_write("new_file.txt")
        entries = cl.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "create"
        assert entries[0].before_snapshot is None

    def test_record_modify(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "existing.txt").write_text("original content")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("existing.txt")

        entries = cl.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "modify"
        assert entries[0].before_snapshot is not None

        # Verify snapshot content
        snapshot = Path(entries[0].before_snapshot)
        assert snapshot.exists()
        assert snapshot.read_text() == "original content"

    def test_record_delete(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "to_delete.txt").write_text("delete me")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_delete("to_delete.txt")

        entries = cl.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "delete"
        snapshot = Path(entries[0].before_snapshot)
        assert snapshot.read_text() == "delete me"

    def test_record_rename(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_rename("old.txt", "new.txt")

        entries = cl.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "rename"
        assert entries[0].new_path == "new.txt"

    def test_subtask_tracking(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("a.txt", subtask_id="step-1")
        cl.record_before_write("b.txt", subtask_id="step-2")
        cl.record_before_write("c.txt", subtask_id="step-1")

        step1 = [e for e in cl.get_entries() if e.subtask_id == "step-1"]
        assert len(step1) == 2

    def test_persistence(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl1 = ChangeLog("t1", workspace, data_dir)
        cl1.record_before_write("file.txt")

        # Reload from disk
        cl2 = ChangeLog("t1", workspace, data_dir)
        entries = cl2.get_entries()
        assert len(entries) == 1
        assert entries[0].path == "file.txt"

    def test_get_summary(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "mod.txt").write_text("original")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("new.txt")  # create
        cl.record_before_write("mod.txt")  # modify
        cl.record_delete("del.txt")  # delete (noop since file doesn't exist)
        cl.record_rename("old.txt", "new2.txt")

        summary = cl.get_summary()
        assert "new.txt" in summary["created"]
        assert "mod.txt" in summary["modified"]
        assert "old.txt -> new2.txt" in summary["renamed"]

    def test_auto_increment_ids(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("a.txt")
        cl.record_before_write("b.txt")
        cl.record_before_write("c.txt")

        ids = [e.id for e in cl.get_entries()]
        assert ids == [1, 2, 3]


class TestRevert:
    def test_revert_create(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("new_file.txt")
        (workspace / "new_file.txt").write_text("new content")

        assert (workspace / "new_file.txt").exists()
        cl.revert_entry(1)
        assert not (workspace / "new_file.txt").exists()

    def test_revert_modify(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "file.txt").write_text("original")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("file.txt")
        (workspace / "file.txt").write_text("modified")

        assert (workspace / "file.txt").read_text() == "modified"
        cl.revert_entry(1)
        assert (workspace / "file.txt").read_text() == "original"

    def test_revert_delete(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "file.txt").write_text("content")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_delete("file.txt")
        (workspace / "file.txt").unlink()

        assert not (workspace / "file.txt").exists()
        cl.revert_entry(1)
        assert (workspace / "file.txt").read_text() == "content"

    def test_revert_rename(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "old.txt").write_text("content")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_rename("old.txt", "new.txt")
        (workspace / "old.txt").rename(workspace / "new.txt")

        assert not (workspace / "old.txt").exists()
        assert (workspace / "new.txt").exists()
        cl.revert_entry(1)
        assert (workspace / "old.txt").exists()
        assert not (workspace / "new.txt").exists()

    def test_revert_all(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "existing.txt").write_text("original")

        cl = ChangeLog("t1", workspace, data_dir)

        # Create new file
        cl.record_before_write("new.txt")
        (workspace / "new.txt").write_text("new")

        # Modify existing
        cl.record_before_write("existing.txt")
        (workspace / "existing.txt").write_text("changed")

        count = cl.revert_all()
        assert count == 2
        assert not (workspace / "new.txt").exists()
        assert (workspace / "existing.txt").read_text() == "original"

    def test_revert_subtask(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)

        cl.record_before_write("step1.txt", subtask_id="s1")
        (workspace / "step1.txt").write_text("from step 1")

        cl.record_before_write("step2.txt", subtask_id="s2")
        (workspace / "step2.txt").write_text("from step 2")

        # Revert only step 1
        count = cl.revert_subtask("s1")
        assert count == 1
        assert not (workspace / "step1.txt").exists()
        assert (workspace / "step2.txt").exists()

    def test_revert_missing_entry_raises(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        cl = ChangeLog("t1", workspace, data_dir)

        try:
            cl.revert_entry(999)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestDiffGenerator:
    def test_diff_modified_file(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        (workspace / "file.txt").write_text("line 1\nline 2\nline 3\n")

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("file.txt")
        (workspace / "file.txt").write_text("line 1\nline 2 modified\nline 3\n")

        gen = DiffGenerator(workspace)
        diff = gen.generate(cl, "file.txt")
        assert "---" in diff
        assert "+++" in diff
        assert "-line 2" in diff
        assert "+line 2 modified" in diff

    def test_diff_new_file(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)

        cl = ChangeLog("t1", workspace, data_dir)
        cl.record_before_write("new.txt")
        (workspace / "new.txt").write_text("new content\n")

        gen = DiffGenerator(workspace)
        diff = gen.generate(cl, "new.txt")
        assert "+new content" in diff

    def test_diff_no_entries(self, tmp_path: Path):
        workspace, data_dir = _setup_workspace(tmp_path)
        cl = ChangeLog("t1", workspace, data_dir)

        gen = DiffGenerator(workspace)
        diff = gen.generate(cl, "nonexistent.txt")
        assert diff == ""


class TestValidateWorkspace:
    def test_valid_workspace(self, tmp_path: Path):
        valid, msg = validate_workspace(tmp_path)
        assert valid
        assert msg == "OK"

    def test_nonexistent_path(self):
        valid, msg = validate_workspace("/nonexistent/path/xyz")
        assert not valid
        assert "does not exist" in msg

    def test_file_not_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        valid, msg = validate_workspace(f)
        assert not valid
        assert "not a directory" in msg
