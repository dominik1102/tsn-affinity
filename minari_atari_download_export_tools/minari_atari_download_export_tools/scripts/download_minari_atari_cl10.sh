#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_minari_atari_cl10.sh /net/tscratch/people/plgdomin088/datasets/minari
# If the first argument is omitted, Minari uses ~/.minari/datasets.

if [[ $# -ge 1 ]]; then
  export MINARI_DATASETS_PATH="$1"
  mkdir -p "$MINARI_DATASETS_PATH"
  echo "[minari] MINARI_DATASETS_PATH=$MINARI_DATASETS_PATH"
else
  echo "[minari] using default Minari path: ~/.minari/datasets"
fi

# Helpful on Cyfronet/HPC if SSL certs are problematic.
python - <<'PY' || true
try:
    import certifi, os
    print(certifi.where())
except Exception:
    pass
PY

DATASETS=(
  atari/alien/expert-v0
  atari/atlantis/expert-v0
  atari/boxing/expert-v0
  atari/breakout/expert-v0
  atari/centipede/expert-v0
  atari/doubledunk/expert-v0
  atari/freeway/expert-v0
  atari/pong/expert-v0
  atari/spaceinvaders/expert-v0
  atari/tennis/expert-v0
)

for ds in "${DATASETS[@]}"; do
  echo "\n[minari] downloading $ds"
  minari download "$ds"
done

echo "\n[minari] local Atari datasets:"
minari list local | grep -i atari || true
