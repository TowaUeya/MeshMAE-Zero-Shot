#!/usr/bin/env bash
set -e

python train_pretrain.py \
  --dataroot "${DATAROOT:-./datasets/}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --epochs "${EPOCHS:-20}" \
  --mask_ratio "${MASK_RATIO:-0.75}" \
  ${INIT:+--init "$INIT"} \
  ${RESUME:+--resume "$RESUME"} \
  ${SAVE:+--save_ckpt "$SAVE"}
