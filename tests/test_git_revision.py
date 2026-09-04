"""The .git fallback that keeps in-container reports traceable.

The app image ships no `git` binary, so `_git_revision()`'s subprocess call
returns nothing there and every report used to stamp "unknown" — which made
the tuning runbook's "no -dirty suffix" pre-launch check vacuous.
"""
import eval.run_eval as run_eval
from eval.run_eval import _git_revision_from_dotgit

SHA = "5bff87a1234567890abcdef1234567890abcdef1"


def _dotgit(tmp_path, monkeypatch):
    """Point ROOT.parent at a throwaway tree and return its .git dir."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    monkeypatch.setattr(run_eval, "ROOT", tmp_path / "eval")
    return git_dir


def test_reads_the_sha_through_a_loose_branch_ref(tmp_path, monkeypatch):
    git_dir = _dotgit(tmp_path, monkeypatch)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "refs" / "heads" / "main").write_text(f"{SHA}\n")

    assert _git_revision_from_dotgit() == "5bff87a-nogit"


def test_reads_the_sha_from_packed_refs_when_no_loose_ref_exists(
        tmp_path, monkeypatch):
    git_dir = _dotgit(tmp_path, monkeypatch)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n"
        f"{SHA} refs/heads/main\n"
        f"0000000000000000000000000000000000000000 refs/heads/other\n"
    )

    assert _git_revision_from_dotgit() == "5bff87a-nogit"


def test_reads_a_detached_head_written_as_a_bare_sha(tmp_path, monkeypatch):
    git_dir = _dotgit(tmp_path, monkeypatch)
    (git_dir / "HEAD").write_text(f"{SHA}\n")

    assert _git_revision_from_dotgit() == "5bff87a-nogit"


def test_returns_unknown_when_the_ref_resolves_nowhere(tmp_path, monkeypatch):
    git_dir = _dotgit(tmp_path, monkeypatch)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    assert _git_revision_from_dotgit() == "unknown"


def test_never_stamps_unknown_when_the_git_binary_is_missing(
        tmp_path, monkeypatch):
    """The whole point: no git binary must still yield a traceable SHA."""
    git_dir = _dotgit(tmp_path, monkeypatch)
    (git_dir / "HEAD").write_text(f"{SHA}\n")

    def no_git_binary(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_eval.subprocess, "run", no_git_binary)

    assert run_eval._git_revision() == "5bff87a-nogit"
