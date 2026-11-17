#!/usr/bin/env bash
# Q/K position reinjection from layer 1. One configuration per launch.
# Usage from this directory:  ./run_rocstories_train_qk_reinject.sh
# Default order: four-tensor, then baseline.
#   MODEL=four_tensor ./run_rocstories_train_qk_reinject.sh
#   MODEL=baseline_ar_attention ./run_rocstories_train_qk_reinject.sh
# Edit N_HEADS, N_LAYERS, and SEED below, then launch again.
# Reported grid, same knobs as run_rocstories_train.sh:
#   H2 L2, H4 L2, H4 L3    seeds 41, 42, 43
# Checkpoints go to output_four_tensor_qk_reinject and
# output_baseline_ar_attention_qk_reinject.
# Baseline position_scale is omitted, so c matches the mean 4-tensor Pos energy.
# All training knobs live in this file. The train scripts have no defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Ubuntu dual RTX 3080 env (same as run_rocstories_train.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- data pool (shared with the published runs; not rewritten if present) ---
DATA_DIR="rocstories_T4max_x16"
T=4
MAX_X=16

# --- model / train (edit here) ---
D_X=64
D_T=32
N_HEADS=4
N_LAYERS=3
DROPOUT=0.1
PADDING_IDX=0
EPOCHS=50
BATCH=32
LR=0.0003
OPTIMIZER=adamw
NUM_WORKERS=2
SEED=41
DETERMINISTIC=true
SAVE_CHECKPOINTS=true
RESUME_FROM_CHECKPOINT=false
STOP_AT_VAL_MINIMUM=true
VAL_MINIMUM_PATIENCE=5
VAL_MINIMUM_RATIO=1.01
CHECKPOINT_EVERY=0

# Layer 0 stays the original forward. 0 also adds Pos to layer 0's Q and K.
REINJECT_FROM_LAYER=1

# Unset / empty: four_tensor then baseline_ar_attention.
MODEL="${MODEL:-}"

RUN_TAG="rocstories_T${T}N${MAX_X}_dx${D_X}_dt${D_T}_H${N_HEADS}L${N_LAYERS}_do${DROPOUT}_b${BATCH}_s${SEED}"

case "${MODEL}" in
  "")
    MODELS=(four_tensor baseline_ar_attention)
    ;;
  four_tensor|baseline_ar_attention)
    MODELS=("${MODEL}")
    ;;
  *)
    echo "MODEL must be empty, four_tensor, or baseline_ar_attention, got: ${MODEL}" >&2
    exit 1
    ;;
esac

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${HOME}/.miniconda3/envs/torch/bin/python" ]]; then
    PYTHON="${HOME}/.miniconda3/envs/torch/bin/python"
  else
    PYTHON="python"
  fi
fi

TAG="T${T}_N${MAX_X}"
JSONL="${DATA_DIR}/psf_pool_${TAG}.jsonl"
META="${DATA_DIR}/psf_pool_${TAG}.meta.json"
VOCAB="${DATA_DIR}/psf_pool_${TAG}.vocab.json"

pool_ready() {
  [[ -f "${JSONL}" && -f "${META}" && -f "${VOCAB}" ]] || return 1
  "${PYTHON}" - "${META}" "${T}" "${MAX_X}" <<'PY'
import json, sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
want_t, want_n = int(sys.argv[2]), int(sys.argv[3])
ok = int(meta.get("T", -1)) == want_t and int(meta.get("N", -1)) == want_n
raise SystemExit(0 if ok else 1)
PY
}

run_train() {
  local model="$1"
  local train_py train_log output_dir

  unset FOUR_TENSOR_TERMINAL_LOG BASELINE_AR_TERMINAL_LOG
  case "${model}" in
    four_tensor)
      train_py="four_tensor_train_qk_reinject.py"
      train_log="${ROOT}/four_tensor_train_qk_reinject.log"
      output_dir="${DATA_DIR}/output_four_tensor_qk_reinject"
      export FOUR_TENSOR_TERMINAL_LOG="${train_log}"
      ;;
    baseline_ar_attention)
      train_py="baseline_ar_attention_train_qk_reinject.py"
      train_log="${ROOT}/baseline_ar_attention_train_qk_reinject.log"
      output_dir="${DATA_DIR}/output_baseline_ar_attention_qk_reinject"
      export BASELINE_AR_TERMINAL_LOG="${train_log}"
      ;;
    *)
      echo "internal error: unknown model ${model}" >&2
      exit 1
      ;;
  esac

  mkdir -p "${output_dir}"

  echo "[train] model=${model}  script=${train_py}  python=${PYTHON}"
  echo "[train] output_dir=${output_dir}  tag=${RUN_TAG}  log=${train_log}"
  echo "[train] reinject_from_layer=${REINJECT_FROM_LAYER} d_x=${D_X} d_t=${D_T} n_heads=${N_HEADS} n_layers=${N_LAYERS} dropout=${DROPOUT} padding_idx=${PADDING_IDX} epochs=${EPOCHS} batch/gpu=${BATCH} lr=${LR} opt=${OPTIMIZER} workers=${NUM_WORKERS} seed=${SEED} deterministic=${DETERMINISTIC}"
  "${PYTHON}" "${train_py}" \
    --reinject_from_layer "${REINJECT_FROM_LAYER}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${output_dir}" \
    --run_tag "${RUN_TAG}" \
    --d_x "${D_X}" \
    --d_t "${D_T}" \
    --n_heads "${N_HEADS}" \
    --n_layers "${N_LAYERS}" \
    --dropout "${DROPOUT}" \
    --padding_idx "${PADDING_IDX}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --lr "${LR}" \
    --optimizer "${OPTIMIZER}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --deterministic "${DETERMINISTIC}" \
    --save_checkpoints "${SAVE_CHECKPOINTS}" \
    --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" \
    --stop_at_val_minimum "${STOP_AT_VAL_MINIMUM}" \
    --val_minimum_patience "${VAL_MINIMUM_PATIENCE}" \
    --val_minimum_ratio "${VAL_MINIMUM_RATIO}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    2>&1 | tee "${train_log}"
}

if pool_ready; then
  echo "[data] reuse ${DATA_DIR} (T=${T}, N=${MAX_X} already present)"
else
  echo "[data] four_tensor_data_rocstories.py --T ${T} --max_x ${MAX_X} --out_dir ${DATA_DIR}"
  "${PYTHON}" four_tensor_data_rocstories.py --T "${T}" --max_x "${MAX_X}" --out_dir "${DATA_DIR}"
fi

echo "[run] models: ${MODELS[*]}"
for model in "${MODELS[@]}"; do
  run_train "${model}"
done
