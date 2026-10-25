"""Student sampling config (`config/sampling.toml`), shared by every launcher via the proxy."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guardfork import REPO_ROOT

DEFAULT_SAMPLING_PATH = REPO_ROOT / "config" / "sampling.toml"


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    request: dict[str, Any] = field(default_factory=dict)
    seed_per_request: bool = False
    sha256: str = ""

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256[:12], **self.request,
                "seed_per_request": self.seed_per_request}


NO_SAMPLING = SamplingConfig("server-defaults")


def load_sampling(path: Path | str | None = DEFAULT_SAMPLING_PATH) -> SamplingConfig:
    if path is None or str(path) == "none":
        return NO_SAMPLING
    raw = Path(path).read_bytes()
    data = tomllib.loads(raw.decode("utf-8"))
    return SamplingConfig(
        name=data["name"],
        request=dict(data.get("request", {})),
        seed_per_request=bool(data.get("seed_per_request", False)),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
