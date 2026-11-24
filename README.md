# EcoFrame

Official code for **When and Where to Look: Adaptive Visual Evidence Scheduling for Efficient Long Video Understanding**.

[Paper](https://arxiv.org/abs/2608.03918) | [Installation](#-installation) | [Quick Start](#-quick-start) | [Citation](#-citation)

## 📰 News

- **2026-11-24:** Code released.
- **2026-08-04:** Our paper is available on [arXiv](https://arxiv.org/abs/2608.03918).

## 📦 Installation

```bash
conda env create -f envs/ecoframe.yml
conda activate ecoframe
python -m pip install --no-deps --no-build-isolation -e .
python -m pip install --no-deps --no-build-isolation -e ./lmms-eval
bash scripts/install_flash_attn.sh
```

## 🚀 Quick Start

### 1. Set Paths

```bash
export HF_HOME=/path/to/huggingface
export FEATURE_CACHE_ROOT=/path/to/feature_cache
```

### 2. Cache Features

Datasets are downloaded and extracted automatically from Hugging Face.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/cache_video_embeddings.py \
    --dataset_name videomme \
    --cache_root "$FEATURE_CACHE_ROOT" \
    --encoder_model_name openai/clip-vit-large-patch14
```

Caches: `{cache_root}/{dataset_name}/{encoder_name}/`. `encoder_name` defaults to the encoder ID without its organization prefix.

### 3. Run Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
MODEL_FAMILY=llava_ov DATASETS="videomme" EARLY_EXIT=True \
ENTROPY_THRESHOLD_START=0.1 ENTROPY_THRESHOLD_STEP=0.2 \
bash scripts/run.sh
```

Supported model families: `llava_ov`, `qwen25_vl`, and `internvl3`.

Recommended early-exit thresholds:

| Dataset | `ENTROPY_THRESHOLD_START` | `ENTROPY_THRESHOLD_STEP` |
| --- | --- | --- |
| VideoMME | `0.1` | `0.2` |
| LongVideoBench / MLVU | `0.2` | `0.3` |

Outputs: `logs/{dataset}/{RUN_ID}/`.

## ⚙️ Evaluation Options

| Variable | Default | Meaning |
| --- | --- | --- |
| `FRAME_SCHEDULE` | `[4,8,16,32]` | Frame budget per round |
| `EARLY_EXIT` | `False` | Enable online early exit; `False` runs all rounds with per-round logs (recommended) |
| `ENTROPY_THRESHOLD_START` / `ENTROPY_THRESHOLD_STEP` | `0.1` / `0.2` | Initial early-exit threshold and increment per round |
| `ENCODER_MODEL_NAME` | `openai/clip-vit-large-patch14` | Feature encoder model |
| `LIMIT` | Unset | Evaluation sample limit |

`MODEL_EXTRA_ARGS` accepts additional comma-separated model arguments.

## 🗂️ Repository Structure

```text
core/
  runner.py             Coarse-to-fine inference scheduler
  adapters/             Model interfaces and attention patches
  lmms_eval_wrappers/   Evaluation wrappers
llava/                  Bundled LLaVA-NeXT implementation
lmms-eval/              Bundled lmms-eval evaluation framework
scripts/                Preprocessing and evaluation entry points
envs/                   Environment configuration
```

## 🙏 Acknowledgements

This project builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2026ecoframe,
  title={When and Where to Look: Adaptive Visual Evidence Scheduling for Efficient Long Video Understanding},
  author={Li, Ke and Chen, Jiayu and Li, Maoliang and Zheng, Zihao and Zou, Hailong and Zhang, Hengyi and Liu, Xuanzhe and Chen, Xiang},
  journal={arXiv preprint arXiv:2608.03918},
  year={2026},
  url={https://arxiv.org/abs/2608.03918}
}
```
