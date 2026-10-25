from guardfork.judge import JUDGE_VERSION, judge_versions, prompt_sha256


def test_judge_prompt_matches_pinned_version():
    versions = judge_versions()
    assert JUDGE_VERSION in versions, f"add [versions.{JUDGE_VERSION}] to judge_freeze.toml"
    entry = versions[JUDGE_VERSION]
    assert prompt_sha256() == entry["sha256"], (
        "config/judge_prompt.md changed: update the draft sha256, or bump JUDGE_VERSION and "
        "add a new version if this one is frozen"
    )
    assert entry.get("status") in ("draft", "frozen")
    if entry.get("status") == "frozen":
        assert entry.get("frozen"), "frozen versions need a date"


def test_template_has_only_known_placeholders():
    import re

    from guardfork.judge import load_template

    names = set(re.findall(r"\{\{(\w+)\}\}", load_template()))
    assert names == {"TASK", "N_TURNS", "TURNS"}
