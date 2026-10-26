from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "pre-commit"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    shutil.copy(HOOK, hooks / "pre-commit")
    (hooks / "pre-commit").chmod(0o755)
    for k, v in (("core.hooksPath", "hooks"), ("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(tmp_path), "config", k, v], check=True)
    return tmp_path


def _commit(repo: Path, rel: str, text: str) -> int:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", "-f", rel], check=True)
    return subprocess.run(["git", "-C", str(repo), "commit", "-qm", "x"],
                          capture_output=True).returncode


@pytest.mark.parametrize("rel", ["runs/a.json", "results/x.csv", "forks/f.txt", "teacher/t.md",
                                 "x/appworld_initial_context.json", "a.tar", "log.jsonl"])
def test_blocks_paths(repo, rel):
    assert _commit(repo, rel, "data") != 0


def test_blocks_task_text(repo):
    assert _commit(repo, "notes.txt", "prompt ... [APPWORLD_CURRENT_TASK] ...") != 0


@pytest.mark.parametrize("secret", ["sk-ant-" + "A1b2C3d4" * 6,
                                    "sk-or-v1-" + "0123456789abcdef" * 3])
def test_blocks_api_keys(repo, secret):
    assert _commit(repo, "config/judges.toml", f'api_key = "{secret}"\n') != 0


def test_allows_code(repo):
    assert _commit(repo, "guardfork/x.py", "print('hi')\n") == 0
