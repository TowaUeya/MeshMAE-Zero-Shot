#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
使い方: run_target_pretrain.sh [--config <path>] [--dry-run]

MeshMAE の継続自己教師あり学習を YAML から実行するラッパーです。
USAGE
}

CONFIG="configs/pretrain_target.yaml"
CLI_DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      CLI_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知の引数です: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "設定ファイルが存在しません: $CONFIG" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${MESHMAE_ROOT:-}" ]]; then
  if [[ -d "$REPO_ROOT/../MeshMAE" ]]; then
    MESHMAE_ROOT="$REPO_ROOT/../MeshMAE"
  elif [[ -d "$REPO_ROOT/MeshMAE" ]]; then
    MESHMAE_ROOT="$REPO_ROOT/MeshMAE"
  else
    MESHMAE_ROOT="$REPO_ROOT/../MeshMAE"
  fi
fi

if [[ ! -d "$MESHMAE_ROOT" ]]; then
  echo "MeshMAE リポジトリが見つかりません: $MESHMAE_ROOT" >&2
  exit 1
fi

CONFIG_EXPORTS=$(python - <<'PY' "$CONFIG" "$REPO_ROOT"
import json
import shlex
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f) or {}

def resolve_path(value):
    if value in (None, ""):
        return ""
    p = Path(value)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return str(p)

data = {
    "dataroot": resolve_path(cfg.get("dataroot", "./datasets/")),
    "batch_size": int(cfg.get("batch_size", 8)),
    "epochs": int(cfg.get("epochs", 20)),
    "mask_ratio": float(cfg.get("mask_ratio", 0.75)),
    "init_checkpoint": resolve_path(cfg.get("init_checkpoint", "")),
    "resume_checkpoint": resolve_path(cfg.get("resume_checkpoint", "")),
    "save_checkpoint": resolve_path(cfg.get("save_checkpoint", "")),
    "dry_run": bool(cfg.get("dry_run", False)),
}

lines = []
for key, value in data.items():
    if isinstance(value, str):
        lines.append(f"CFG_{key.upper()}={shlex.quote(value)}")
    elif isinstance(value, bool):
        lines.append(f"CFG_{key.upper()}={'1' if value else '0'}")
    else:
        lines.append(f"CFG_{key.upper()}={value}")
print("\n".join(lines))
PY
)

eval "$CONFIG_EXPORTS"

if [[ $CLI_DRY_RUN -eq 1 ]]; then
  CFG_DRY_RUN=1
fi

DEFAULT_INIT_PATH="$(cd "$REPO_ROOT" && pwd)/checkpoints/shapenet_pretrain.pkl"

INIT_PATH="$CFG_INIT_CHECKPOINT"
RESUME_PATH="$CFG_RESUME_CHECKPOINT"
SAVE_PATH="$CFG_SAVE_CHECKPOINT"

if [[ -n "$INIT_PATH" ]]; then
  if [[ ! -f "$INIT_PATH" ]]; then
    echo "init_checkpoint で指定したファイルが存在しません: $INIT_PATH" >&2
    exit 1
  fi
else
  if [[ -f "$DEFAULT_INIT_PATH" ]]; then
    echo "init_checkpoint が未指定ですが $DEFAULT_INIT_PATH が存在します。config にパスを設定してください。" >&2
    exit 1
  fi
fi

if [[ -n "$RESUME_PATH" && ! -f "$RESUME_PATH" ]]; then
  echo "resume_checkpoint で指定したファイルが存在しません: $RESUME_PATH" >&2
  exit 1
fi

if [[ -n "$SAVE_PATH" ]]; then
  SAVE_DIR="$(dirname "$SAVE_PATH")"
  mkdir -p "$SAVE_DIR"
fi

printf '使用設定:\n  dataroot: %s\n  init: %s\n  resume: %s\n  save: %s\n' \
  "$CFG_DATAROOT" "${INIT_PATH:-<none>}" "${RESUME_PATH:-<none>}" "${SAVE_PATH:-<none>}"

export DATAROOT="$CFG_DATAROOT"
export BATCH_SIZE="$CFG_BATCH_SIZE"
export EPOCHS="$CFG_EPOCHS"
export MASK_RATIO="$CFG_MASK_RATIO"

if [[ -n "$INIT_PATH" ]]; then
  export INIT="$INIT_PATH"
else
  unset INIT || true
fi

if [[ -n "$RESUME_PATH" ]]; then
  export RESUME="$RESUME_PATH"
else
  unset RESUME || true
fi

if [[ -n "$SAVE_PATH" ]]; then
  export SAVE="$SAVE_PATH"
else
  unset SAVE || true
fi

CMD=("bash" "scripts/pretrain/train_pretrain.sh")
CMD_STR="${CMD[*]}"

OUT_DIR="$REPO_ROOT/out"
mkdir -p "$OUT_DIR"
LOG_JSON="$OUT_DIR/pretrain_log.json"

if [[ $CFG_DRY_RUN -eq 1 ]]; then
  echo "ドライラン: ${CMD_STR}"
  python - <<'PY' "$LOG_JSON" "$CMD_STR" "$CFG_DATAROOT" "$INIT_PATH" "$RESUME_PATH" "$SAVE_PATH"
import json
import sys
log_path, cmdline, dataroot, init_path, resume_path, save_path = sys.argv[1:7]
with open(log_path, "w", encoding="utf-8") as f:
    json.dump({
        "cmdline": cmdline,
        "dataroot": dataroot,
        "init": init_path,
        "resume": resume_path,
        "save": save_path,
        "missing_keys": [],
        "unexpected_keys": [],
        "dry_run": True,
    }, f, ensure_ascii=False, indent=2)
print(f"ログを作成しました: {log_path}")
PY
  exit 0
fi

OUTPUT_CAPTURE="$(mktemp)"
trap 'rm -f "$OUTPUT_CAPTURE"' EXIT

set +e
set -o pipefail
(
  cd "$MESHMAE_ROOT"
  "${CMD[@]}"
) | tee "$OUTPUT_CAPTURE"
STATUS=${PIPESTATUS[0]}
set +o pipefail
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "MeshMAE 実行が失敗しました (exit=$STATUS)" >&2
  exit $STATUS
fi

python - <<'PY' "$OUTPUT_CAPTURE" "$LOG_JSON" "$CMD_STR" "$CFG_DATAROOT" "$INIT_PATH" "$RESUME_PATH" "$SAVE_PATH"
import json
import sys
from pathlib import Path
output_path, log_path, cmdline, dataroot, init_path, resume_path, save_path = sys.argv[1:7]
missing = []
unexpected = []
last_json = None
for line in Path(output_path).read_text(encoding="utf-8").splitlines()[::-1]:
    line = line.strip()
    if not line:
        continue
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        continue
    else:
        if isinstance(parsed, dict) and "missing_keys" in parsed and "unexpected_keys" in parsed:
            last_json = parsed
            break
if last_json:
    missing = last_json.get("missing_keys", [])
    unexpected = last_json.get("unexpected_keys", [])
with open(log_path, "w", encoding="utf-8") as f:
    json.dump({
        "cmdline": cmdline,
        "dataroot": dataroot,
        "init": init_path,
        "resume": resume_path,
        "save": save_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "dry_run": False,
    }, f, ensure_ascii=False, indent=2)
print(f"ログを作成しました: {log_path}")
PY

trap - EXIT
rm -f "$OUTPUT_CAPTURE"
