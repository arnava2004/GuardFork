from guardfork.freeze import frozen_versions, guard_sources_sha256, load_dev_set
from guardfork.guards.rules import GUARDS_VERSION


def test_guard_sources_match_frozen_version():
    versions = frozen_versions()
    assert GUARDS_VERSION in versions, f"add [versions.{GUARDS_VERSION}] to guards_freeze.toml"
    assert guard_sources_sha256() == versions[GUARDS_VERSION], (
        "guard sources changed: bump GUARDS_VERSION and add a new frozen version "
        "instead of editing a released one"
    )


def test_dev_set_excludes_tuning_tasks_and_jobs():
    dev = load_dev_set()
    assert dev.excludes("anything", "302c169_2")
    assert dev.excludes("anything", "07b42fd_3")  # whole scenario, not just the touched task
    assert dev.excludes("w0-mini-4b-mlx", None)
    assert not dev.excludes("w0-eval-9b", "82e2fac_1")
    assert len(dev.scenarios) == 15 and len(dev.tasks) == 45
    assert not (dev.scenarios & {"c901732", "60d0b5b"})  # referee-smoke runs stay in eval
