#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/lmms-eval:${PYTHONPATH:-}"

if [[ "${DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

DATASETS="${DATASETS:-videomme}"
# videomme longvideobench_val_v mlvu_dev activitynetqa
FRAME_COUNTS="${FRAME_COUNTS:-4 8 16 32}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen25_vl}"
# qwen25_vl llava_ov
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12345}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs}"
MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS:-}"

case "${MODEL_FAMILY}" in
  qwen25_vl)
    LMMS_MODEL="${LMMS_MODEL:-qwen2_5_vl_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},min_pixels=200704,max_pixels=200704,attn_implementation=flash_attention_2,interleave_visuals=False"
    FRAME_ARG_NAME="max_num_frames"
    ;;
  llava_ov)
    LMMS_MODEL="${LMMS_MODEL:-llava_onevision_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-lmms-lab/llava-onevision-qwen2-7b-ov}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},conv_template=qwen_1_5,model_name=llava_qwen"
    FRAME_ARG_NAME="max_frames_num"
    ;;
  *)
    echo "Unknown MODEL_FAMILY=${MODEL_FAMILY}. Use qwen25_vl or llava_ov." >&2
    exit 1
    ;;
esac

echo "Datasets: ${DATASETS}"
echo "Frame counts: ${FRAME_COUNTS}"
echo "Model family: ${MODEL_FAMILY}"
echo "LMMS model: ${LMMS_MODEL}"
echo "Pretrained: ${PRETRAINED_NAME}"

for dataset_name in ${DATASETS}; do
  for num_frames in ${FRAME_COUNTS}; do
    output_path="${OUTPUT_ROOT}/${dataset_name}/uniform_${MODEL_FAMILY}/${num_frames}_frames"
    model_args="${MODEL_BASE_ARGS},${FRAME_ARG_NAME}=${num_frames}"
    if [[ -n "${MODEL_EXTRA_ARGS}" ]]; then
      model_args="${model_args},${MODEL_EXTRA_ARGS}"
    fi
    mkdir -p "${output_path}"

    echo "[$(date '+%F %T')] start ${dataset_name}/${MODEL_FAMILY}/${num_frames}_frames"
    nohup setsid accelerate launch \
      --num_processes="${NUM_PROCESSES}" \
      --main_process_port="${MAIN_PROCESS_PORT}" \
      -m lmms_eval \
      --model "${LMMS_MODEL}" \
      --model_args "${model_args}" \
      --tasks "${dataset_name}" \
      --batch_size 1 \
      --log_samples \
      --output_path "${output_path}" \
      >"${output_path}/run.log" 2>&1 &
    wait "$!"
    echo "[$(date '+%F %T')] done  ${dataset_name}/${MODEL_FAMILY}/${num_frames}_frames"
  done
done

echo "[$(date '+%F %T')] all uniform jobs finished"
