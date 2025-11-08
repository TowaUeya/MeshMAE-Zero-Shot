#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [--config configs/pretrain_target.yaml] [--dry-run]

This script wraps the official MeshMAE pretraining entrypoint so that
continuing self-supervised learning on fossil meshes is one command away.
The script expects that the MeshMAE repository is available and that the
`train_pretrain.sh` helper script remains API-compatible with the public repo.

Environment variables:
  MESHMAE_ROOT   Path to the MeshMAE repository (default: ../MeshMAE)
  PYTHON         Python executable to invoke (default: python)
  EXTRA_ARGS     Additional CLI flags appended to the MeshMAE script
USAGE
}

CONFIG="configs/pretrain_target.yaml"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

MESHMAE_ROOT="${MESHMAE_ROOT:-../MeshMAE}"
PYTHON_BIN="${PYTHON:-python}"

if [[ ! -d "$MESHMAE_ROOT" ]]; then
  echo "MeshMAE repository not found at $MESHMAE_ROOT" >&2
  exit 1
fi

CONFIG_JSON=$(python - <<'PY'
import sys
import yaml
from pathlib import Path
cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
meshmae = cfg.get("meshmae", {})
print(" ".join([
    f"--dataroot {meshmae.get('dataroot', './datasets/fossils_maps')}",
    f"--batch_size {meshmae.get('batch_size', 32)}",
    f"--epochs {meshmae.get('epochs', 200)}",
    f"--mask_ratio {meshmae.get('mask_ratio', 0.6)}",
    f"--blr {meshmae.get('base_learning_rate', 1.5e-4)}",
    f"--weight_decay {meshmae.get('weight_decay', 0.05)}",
    f"--warmup_epochs {meshmae.get('warmup_epochs', 20)}",
]))
PY
)

CHECKPOINT_ARGS=$(python - <<'PY'
import sys
import yaml
from pathlib import Path
cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
meshmae = cfg.get("meshmae", {})
resume = meshmae.get("resume_checkpoint")
save = meshmae.get("save_checkpoint")
args = []
if resume:
    args.append(f"--resume {resume}")
if save:
    args.append(f"--output_dir {Path(save).parent}")
print(" ".join(args))
PY
)

CMD=("bash" "scripts/pretrain/train_pretrain.sh")
IFS=' ' read -r -a CONFIG_ARGS <<< "$CONFIG_JSON"
IFS=' ' read -r -a CKPT_ARGS <<< "$CHECKPOINT_ARGS"
CMD+=("${CONFIG_ARGS[@]}")
CMD+=("${CKPT_ARGS[@]}")
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  CMD+=($EXTRA_ARGS)
fi

pushd "$MESHMAE_ROOT" >/dev/null
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run command: ${CMD[*]}"
else
  echo "Running: ${CMD[*]}"
  "$PYTHON_BIN" -m pip install -r requirements.txt
  "${CMD[@]}"
fi
popd >/dev/null
