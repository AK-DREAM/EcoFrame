#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/lmms-eval:${PYTHONPATH:-}"

if [[ "${DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

DATASETS="${DATASETS:-${DATASET_NAME:-videomme}}"
# videomme longvideobench_val_v mlvu_dev
MODEL_FAMILY="${MODEL_FAMILY:-llava_ov}"
# llava_ov qwen25_vl llava_vid internvl3
FRAME_SCHEDULE="${FRAME_SCHEDULE:-[4,8,16,32]}"
CTOF_ATTN_LAYERS="${CTOF_ATTN_LAYERS:-19.21}"
LAMBDA_ATTN="${LAMBDA_ATTN:-0.5}"
LAMBDA_RELE="${LAMBDA_RELE:-1.0}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4}"
EARLY_EXIT="${EARLY_EXIT:-False}"
ENTROPY_THRESHOLD_START="${ENTROPY_THRESHOLD_START:-0.1}"
ENTROPY_THRESHOLD_STEP="${ENTROPY_THRESHOLD_STEP:-0.2}"
ENCODER_MODEL_NAME="${ENCODER_MODEL_NAME:-openai/clip-vit-large-patch14}"
# openai/clip-vit-base-patch32
# openai/clip-vit-large-patch14
# google/siglip-so400m-patch14-384
ENCODER_NAME="${ENCODER_NAME:-${ENCODER_MODEL_NAME##*/}}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-./cache/features}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12345}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs}"
LIMIT="${LIMIT:-}"
DOC_ID_FILE="${DOC_ID_FILE:-}"
MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS:-}"

case "${MODEL_FAMILY}" in
  llava_ov)
    LMMS_MODEL="${LMMS_MODEL:-llava_onevision_refine}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-lmms-lab/llava-onevision-qwen2-7b-ov}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32"
    LOG_MODEL_TAG="${LOG_MODEL_TAG:-ov_7b}"
    ;;
  qwen25_vl)
    LMMS_MODEL="${LMMS_MODEL:-qwen2_5_vl_refine}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},min_pixels=200704,max_pixels=200704,attn_implementation=flash_attention_2,interleave_visuals=False,max_num_frames=32"
    LOG_MODEL_TAG="${LOG_MODEL_TAG:-qwen25vl_7b}"
    ;;
  llava_vid)
    LMMS_MODEL="${LMMS_MODEL:-llava_vid_refine}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-lmms-lab/LLaVA-Video-7B-Qwen2}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32"
    LOG_MODEL_TAG="${LOG_MODEL_TAG:-llava_vid_7b}"
    ;;
  internvl3)
    LMMS_MODEL="${LMMS_MODEL:-internvl3_refine}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-OpenGVLab/InternVL3-8B}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},modality=video,num_frame=32,use_flash_attn=True"
    LOG_MODEL_TAG="${LOG_MODEL_TAG:-internvl3_8b}"
    ;;
  *)
    echo "Unknown MODEL_FAMILY=${MODEL_FAMILY}. Use llava_ov, qwen25_vl, llava_vid, or internvl3." >&2
    exit 1
    ;;
esac

if [[ -n "${MODEL_EXTRA_ARGS}" ]]; then
  MODEL_BASE_ARGS="${MODEL_BASE_ARGS},${MODEL_EXTRA_ARGS}"
fi

score_summary() {
  python - "$1" "$2" "$3" <<'PY'
import json
import pathlib
import sys

root, task, label = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for path in sorted(root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        items = json.loads(path.read_text()).get("results", {}).get(task, {}).items()
    except Exception:
        continue
    score = next(((k, v) for k, v in items if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
    if score:
        print(f"[{label}] {score[0]}: {score[1]:.4f}")
        print(f"[{label}] results file: {path}")
        sys.exit(0)
print(f"[{label}] score: not found")
PY
}

echo "Datasets: ${DATASETS}"
echo "Model family: ${MODEL_FAMILY}"
echo "LMMS model: ${LMMS_MODEL}"
echo "Pretrained: ${PRETRAINED_NAME}"
echo "Schedule: ${FRAME_SCHEDULE}"
echo "Early exit: ${EARLY_EXIT} (threshold: ${ENTROPY_THRESHOLD_START} + round * ${ENTROPY_THRESHOLD_STEP})"
echo "Encoder: ${ENCODER_MODEL_NAME}"

for dataset_name in ${DATASETS}; do
  image_cache_dir="${FEATURE_CACHE_ROOT}/${dataset_name}/${ENCODER_NAME}"

  RUN_DIR="${OUTPUT_ROOT}/${dataset_name}/${RUN_ID}"
  DEBUG_DIR="${RUN_DIR}/debug"
  RUN_LOG="${RUN_DIR}/run.log"
  mkdir -p "${DEBUG_DIR}"

  MODEL_ARGS="${MODEL_BASE_ARGS},frame_schedule=${FRAME_SCHEDULE},lambda_attn=${LAMBDA_ATTN},lambda_rele=${LAMBDA_RELE},candidate_multiplier=${CANDIDATE_MULTIPLIER},encoder_model_name=${ENCODER_MODEL_NAME},ctof_attn_layers=${CTOF_ATTN_LAYERS},frame_debug_dir=${DEBUG_DIR},early_exit=${EARLY_EXIT},entropy_threshold_start=${ENTROPY_THRESHOLD_START},entropy_threshold_step=${ENTROPY_THRESHOLD_STEP}"
  MODEL_ARGS="${MODEL_ARGS},image_cache_dir=${image_cache_dir}"

  CMD=(
    accelerate launch
    --num_processes="${NUM_PROCESSES}"
    --main_process_port="${MAIN_PROCESS_PORT}"
    -m lmms_eval
    --model "${LMMS_MODEL}"
    --model_args "${MODEL_ARGS}"
    --tasks "${dataset_name}"
    --batch_size 1
    --log_samples
    --log_samples_suffix "${LOG_MODEL_TAG}_${dataset_name}"
    --output_path "${RUN_DIR}"
  )
  [[ -n "${LIMIT}" ]] && CMD+=(--limit "${LIMIT}")
  [[ -n "${DOC_ID_FILE}" ]] && CMD+=(--doc_id_file "${DOC_ID_FILE}")

  echo "[$(date '+%F %T')] running ${dataset_name}"
  echo "Output dir: ${RUN_DIR}"
  echo "Image cache: ${image_cache_dir}"
  "${CMD[@]}" 2>&1 | tee "${RUN_LOG}"
  score_summary "${RUN_DIR}" "${dataset_name}" "${dataset_name}"
done

echo "[$(date '+%F %T')] all jobs finished"
