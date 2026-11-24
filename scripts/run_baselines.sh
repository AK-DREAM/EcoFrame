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
# videomme longvideobench_val_v mlvu_dev
METHODS="${METHODS:-aks}"
# base aks bolt focus
MODEL_FAMILY="${MODEL_FAMILY:-qwen25_vl}"
# llava_ov qwen25_vl llava_vid internvl3
ENCODER_NAME="${ENCODER_NAME:-clip-vit-large-patch14}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12345}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logs}"
MODEL_EXTRA_ARGS="${MODEL_EXTRA_ARGS:-}"

case "${MODEL_FAMILY}" in
  llava_ov)
    LMMS_MODEL="${LMMS_MODEL:-llava_onevision_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-lmms-lab/llava-onevision-qwen2-7b-ov}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32"
    ;;
  qwen25_vl)
    LMMS_MODEL="${LMMS_MODEL:-qwen2_5_vl_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},min_pixels=200704,max_pixels=200704,attn_implementation=flash_attention_2,interleave_visuals=False,max_num_frames=32"
    ;;
  llava_vid)
    LMMS_MODEL="${LMMS_MODEL:-llava_vid_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-lmms-lab/LLaVA-Video-7B-Qwen2}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32"
    ;;
  internvl3)
    LMMS_MODEL="${LMMS_MODEL:-internvl3_select}"
    PRETRAINED_NAME="${PRETRAINED_NAME:-${MODEL_NAME:-OpenGVLab/InternVL3-8B}}"
    MODEL_BASE_ARGS="pretrained=${PRETRAINED_NAME},modality=video,num_frame=32,use_flash_attn=True"
    ;;
  *)
    echo "Unknown MODEL_FAMILY=${MODEL_FAMILY}. Use llava_ov, qwen25_vl, llava_vid, or internvl3." >&2
    exit 1
    ;;
esac

if [[ -n "${MODEL_EXTRA_ARGS}" ]]; then
  MODEL_BASE_ARGS="${MODEL_BASE_ARGS},${MODEL_EXTRA_ARGS}"
fi

for dataset_name in ${DATASETS}; do
  for method_name in ${METHODS}; do
    output_path="${OUTPUT_ROOT}/${dataset_name}/baselines/${method_name}/${MODEL_FAMILY}"
    selected_frames_path="baselines/selected_frames_${method_name}/${dataset_name}/${ENCODER_NAME}/selected_frames.json"
    model_args="${MODEL_BASE_ARGS}"
    [[ "${method_name}" != "base" ]] && model_args="${model_args},selected_frames_path=${selected_frames_path}"
    mkdir -p "${output_path}"

    echo "[$(date '+%F %T')] start ${dataset_name}/${method_name}/${MODEL_FAMILY}"
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
    echo "[$(date '+%F %T')] done  ${dataset_name}/${method_name}/${MODEL_FAMILY}"
  done
done

echo "[$(date '+%F %T')] all baseline jobs finished"
