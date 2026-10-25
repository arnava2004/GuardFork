#!/usr/bin/env bash
# One-time, idempotent setup of a Lambda Cloud GPU VM (Ubuntu, x86_64, NVIDIA) for W0+ runs.
# Re-run after any failure; finished steps are skipped or cheap (Docker layer cache).
#
#   bash ~/guardfork/scripts/vm_setup.sh
#
# guardfork itself is never cloned from a remote here: rsync it from the Mac first (command
# printed below). The repo must not be pushed publicly (AGENTS.md rule 4).
set -euo pipefail

GF_DIR="${GF_DIR:-$HOME/guardfork}"
HZ_DIR="${HZ_DIR:-$HOME/harness-zero}"
HZ_REPO="https://github.com/metaevo-ai/harness-zero.git"
HZ_COMMIT="a4a6223241479c750fdab827c3a449bb1d41b99a"
DEPS_IMAGE="ahd-appworld-deps:v2"
IMAGE_TAG="v2"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-env}"
VLLM_VERSION="${VLLM_VERSION:-}"   # empty = latest; set e.g. VLLM_VERSION=0.24.0 to pin
SMOKE_TASK="07b42fd_1"
SMOKE_JOB="vm-oracle-smoke"

export BUILDX_BUILDER=default COMPOSE_BAKE=false PYTHONHASHSEED=0
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n==> [%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
run() { printf '+ %s\n' "$*"; "$@"; }

RSYNC_CMD="rsync -az --exclude .venv --exclude runs --exclude jobs --exclude forks \
--exclude results --exclude teacher --exclude __pycache__ \
/Users/arnavarora/Desktop/Developer/guardfork/ ubuntu@<VM_IP>:~/guardfork/"

step "Preflight: x86_64 Ubuntu with an NVIDIA GPU, guardfork rsynced to $GF_DIR"
[[ "$(uname -m)" == "x86_64" ]] || { echo "need x86_64 (got $(uname -m)); GH200 is arm64" >&2; exit 1; }
grep -q '^ID=ubuntu' /etc/os-release || { echo "need Ubuntu" >&2; exit 1; }
run nvidia-smi -L
echo "Sync guardfork from the Mac (never push it publicly):"
echo "  $RSYNC_CMD"
if [[ ! -f "$GF_DIR/pyproject.toml" || ! -d "$GF_DIR/guardfork" ]]; then
  echo "guardfork not found at $GF_DIR; run the rsync above on the Mac, then re-run." >&2
  exit 1
fi

step "Base apt packages"
missing=()
for pkg in git rsync curl ca-certificates gnupg2 python3; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done
if ((${#missing[@]})); then
  run sudo apt-get update
  run sudo apt-get install -y --no-install-recommends "${missing[@]}"
else
  echo "all present"
fi

step "Docker engine + compose v2"
if ! command -v docker >/dev/null 2>&1; then
  # https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script
  curl -fsSL https://get.docker.com | sudo sh
fi
run sudo systemctl enable --now docker
if ! docker compose version >/dev/null 2>&1 && ! sudo docker compose version >/dev/null 2>&1; then
  # VERIFY: docker-compose-plugin comes from Docker's apt repo (get.docker.com path);
  # docker-compose-v2 is Ubuntu's own package when Docker came from docker.io.
  run sudo apt-get install -y docker-compose-plugin || run sudo apt-get install -y docker-compose-v2
fi
if ! docker info >/dev/null 2>&1; then
  if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
    run sudo usermod -aG docker "$USER"
  fi
  if [[ -z "${VM_SETUP_SG:-}" ]]; then
    echo "re-executing under the docker group (sg docker)"
    exec sg docker -c "VM_SETUP_SG=1 $(printf '%q ' bash "$0" "$@")"
  fi
  echo "docker still unusable without sudo; log out and back in, then re-run" >&2
  exit 1
fi
run docker compose version

step "NVIDIA Container Toolkit"
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  run sudo apt-get update
  run sudo apt-get install -y nvidia-container-toolkit
fi
if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
  run sudo nvidia-ctk runtime configure --runtime=docker
  run sudo systemctl restart docker
fi
run docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi -L

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
run uv --version

step "Harness-Zero at $HZ_COMMIT -> $HZ_DIR"
if [[ ! -d "$HZ_DIR/.git" ]]; then
  run git clone "$HZ_REPO" "$HZ_DIR"
fi
if [[ "$(git -C "$HZ_DIR" rev-parse HEAD)" != "$HZ_COMMIT" ]]; then
  git -C "$HZ_DIR" cat-file -e "$HZ_COMMIT^{commit}" 2>/dev/null || run git -C "$HZ_DIR" fetch origin
  run git -C "$HZ_DIR" checkout --detach "$HZ_COMMIT"
fi
(cd "$HZ_DIR" && run uv sync --frozen)
[[ -x "$HZ_DIR/.venv/bin/harbor" ]] || { echo "harbor missing from $HZ_DIR/.venv" >&2; exit 1; }

step "guardfork venv + commit hooks"
(cd "$GF_DIR" && run uv sync --frozen)
if [[ -d "$GF_DIR/.git" ]]; then
  run git -C "$GF_DIR" config core.hooksPath hooks
  git -C "$GF_DIR" log -1 --oneline
fi

step "AppWorld dependencies image $DEPS_IMAGE (native amd64)"
if docker image inspect "$DEPS_IMAGE" >/dev/null 2>&1; then
  echo "exists"
else
  run docker build --builder default -t "$DEPS_IMAGE" - <<'EOF'
FROM python:3.13-slim-bookworm AS dependencies
ENV DEBIAN_FRONTEND=noninteractive PIP_PROGRESS_BAR=off PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gcc libc6-dev \
    && pip install --no-cache-dir appworld==0.1.3.post1 && appworld install \
    && apt-get purge -y --auto-remove gcc libc6-dev && rm -rf /var/lib/apt/lists/*
EOF
fi

step "Official AppWorld data -> $HZ_DIR/data"
if [[ -d "$HZ_DIR/data/api_docs" && -d "$HZ_DIR/data/base_dbs" && -d "$HZ_DIR/data/tasks" \
      && -f "$HZ_DIR/data/datasets/test_normal.txt" && -f "$HZ_DIR/data/version.txt" ]]; then
  echo "already downloaded"
else
  run docker run --rm -v "$HZ_DIR":/w -w /w "$DEPS_IMAGE" appworld download data --root /w
  # The container writes as root; hand the tree back before git touches it.
  run sudo chown -R "$(id -u):$(id -g)" "$HZ_DIR/data"
  # download replaces data/; this restores the tracked Harbor task dirs.
  run git -C "$HZ_DIR" checkout -- data
fi
[[ -d "$HZ_DIR/data/appworld-harbor/tasks/$SMOKE_TASK" ]] \
  || { echo "Harbor task dir for $SMOKE_TASK missing; git -C $HZ_DIR checkout -- data" >&2; exit 1; }

step "Prepare allowlisted build context (build.py --prepare-only)"
(cd "$HZ_DIR" && run .venv/bin/python envs/appworld/build.py --prepare-only)

step "Build client/world/verifier:$IMAGE_TAG on top of $DEPS_IMAGE"
sed "s|^FROM python:3.13-slim-bookworm AS dependencies|FROM $DEPS_IMAGE AS dependencies|; \
/^RUN apt-get update/,/^RUN pip install/d" \
  "$HZ_DIR/envs/appworld/Dockerfile" > "$HZ_DIR/data/appworld-build/Dockerfile"
grep -qxF "FROM $DEPS_IMAGE AS dependencies" "$HZ_DIR/data/appworld-build/Dockerfile"
if grep -q 'pip install --no-cache-dir appworld' "$HZ_DIR/data/appworld-build/Dockerfile"; then
  echo "sed did not strip the original dependency RUN lines" >&2; exit 1
fi
for target in client world verifier; do
  run docker build --builder default --target "$target" -t "ahd-appworld-$target:$IMAGE_TAG" \
    "$HZ_DIR/data/appworld-build"
done
docker images | grep ahd-appworld

step "Harbor oracle smoke on $SMOKE_TASK (expects reward 1.0)"
check_oracle() {
  python3 - "$GF_DIR/jobs/$SMOKE_JOB" <<'PY'
import glob, json, sys
files = glob.glob(f"{sys.argv[1]}/*/result.json")
rewards = [((json.load(open(f)).get("verifier_result") or {}).get("rewards") or {}).get("reward")
           for f in files]
print(f"oracle rewards: {rewards}")
sys.exit(0 if rewards and all(r == 1.0 for r in rewards) else 1)
PY
}
if [[ -d "$GF_DIR/jobs/$SMOKE_JOB" ]] && check_oracle; then
  echo "already passed"
else
  rm -rf "${GF_DIR:?}/jobs/$SMOKE_JOB"
  (
    cd "$HZ_DIR"
    export PYTHONPATH="$HZ_DIR:$HZ_DIR/src"
    run .venv/bin/harbor run --path data/appworld-harbor/tasks --include-task-name "$SMOKE_TASK" \
      --agent oracle --env envs.appworld.harbor_environment:AppWorldDockerEnvironment \
      --n-attempts 1 --n-concurrent 1 --job-name "$SMOKE_JOB" --jobs-dir "$GF_DIR/jobs" --yes
  )
  check_oracle || { echo "oracle smoke did not score 1.0; see $GF_DIR/jobs/$SMOKE_JOB" >&2; exit 1; }
fi

step "vLLM venv -> $VLLM_VENV"
if [[ -x "$VLLM_VENV/bin/vllm" ]] && "$VLLM_VENV/bin/python" -c 'import vllm' 2>/dev/null; then
  echo "exists"
else
  run uv venv --python 3.12 "$VLLM_VENV"
  # https://recipes.vllm.ai/Qwen/Qwen3.5-9B (Qwen3.5 needs vLLM >= 0.17.0)
  run uv pip install --python "$VLLM_VENV/bin/python" -U "vllm${VLLM_VERSION:+==$VLLM_VERSION}" \
    --torch-backend=auto
fi
VLLM_VER="$("$VLLM_VENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
echo "vllm $VLLM_VER"

step "Record environment -> runs/vm/"
mkdir -p "$GF_DIR/runs/vm"
ENV_FILE="$GF_DIR/runs/vm/env_$(date -u +%Y%m%dT%H%M%SZ).txt"
{
  echo "host: $(hostname)  date_utc: $(date -u +%FT%TZ)"
  echo "guardfork_commit: $(git -C "$GF_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "harness_zero_commit: $(git -C "$HZ_DIR" rev-parse HEAD)"
  echo "vllm: $VLLM_VER"
  echo "uv: $(uv --version)"
  echo "docker: $(docker version --format '{{.Server.Version}}')"
  echo "compose: $(docker compose version --short)"
  for img in "$DEPS_IMAGE" "ahd-appworld-client:$IMAGE_TAG" "ahd-appworld-world:$IMAGE_TAG" \
             "ahd-appworld-verifier:$IMAGE_TAG"; do
    echo "image $img: $(docker image inspect --format '{{.Id}}' "$img")"
  done
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  echo "nproc: $(nproc)  mem: $(free -g | awk '/^Mem:/{print $2}') GiB"
} | tee "$ENV_FILE"

step "Done. Next: scripts/vm_serve.sh Qwen/Qwen3.5-9B (see 'Lambda GPU VM' in scripts/w0_runbook.md)"
