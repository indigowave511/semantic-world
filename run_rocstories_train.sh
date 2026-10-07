#!/usr/bin/env bash
# Generate ROCStories T=4 / max_x=16 pool if missing, then train.
# Usage from this directory:  ./run_rocstories_train.sh
# Toggle: leave MODEL unset to run both, in order.
#   ./run_rocstories_train.sh
#   MODEL=four_tensor ./run_rocstories_train.sh
#   MODEL=baseline_ar_attention ./run_rocstories_train.sh
# All training knobs live in this file. The train scripts have no defaults.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Ubuntu dual RTX 3080 env (same as hook_1130_causal_mask.py launch).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- data pool ---
DATA_DIR="rocstories_T4max_x16"
T=4
MAX_X=16

# --- model / train (edit here) ---
# 3080 10GB x2 DataParallel. Heads are full (d_x,d_t), so H is expensive.
# Scale one axis at a time: H1 L1 -> H2 L1 -> H2 L2 -> H3 L2
# Then d 96/48 or N=20. 128/64 does not fit 10GB + AdamW.
D_X=64
D_T=32
N_HEADS=2
N_LAYERS=2
DROPOUT=0.1
PADDING_IDX=0
EPOCHS=50
BATCH=32
LR=0.0003
OPTIMIZER=adamw
NUM_WORKERS=2
SEED=43
DETERMINISTIC=true
SAVE_CHECKPOINTS=true
RESUME_FROM_CHECKPOINT=false
STOP_AT_VAL_MINIMUM=true
VAL_MINIMUM_PATIENCE=5
VAL_MINIMUM_RATIO=1.01
CHECKPOINT_EVERY=0

# --- which trainer ---
# Unset / empty: four_tensor then baseline_ar_attention.
# four_tensor           -> four_tensor_train.py only
# baseline_ar_attention -> baseline_ar_attention_train.py only
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
      train_py="four_tensor_train.py"
      train_log="${ROOT}/four_tensor_train.log"
      output_dir="${DATA_DIR}/output_four_tensor_preLN"
      export FOUR_TENSOR_TERMINAL_LOG="${train_log}"
      ;;
    baseline_ar_attention)
      train_py="baseline_ar_attention_train.py"
      train_log="${ROOT}/baseline_ar_attention_train.log"
      output_dir="${DATA_DIR}/output_baseline_ar_attention_preLN"
      export BASELINE_AR_TERMINAL_LOG="${train_log}"
      ;;
    *)
      echo "internal error: unknown model ${model}" >&2
      exit 1
      ;;
  esac

  echo "[train] model=${model}  script=${train_py}  python=${PYTHON}"
  echo "[train] output_dir=${output_dir}  tag=${RUN_TAG}  log=${train_log}"
  echo "[train] data_dir=${DATA_DIR} d_x=${D_X} d_t=${D_T} n_heads=${N_HEADS} n_layers=${N_LAYERS} dropout=${DROPOUT} padding_idx=${PADDING_IDX} epochs=${EPOCHS} batch/gpu=${BATCH} lr=${LR} opt=${OPTIMIZER} workers=${NUM_WORKERS} seed=${SEED} deterministic=${DETERMINISTIC}"
  "${PYTHON}" "${train_py}" \
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
