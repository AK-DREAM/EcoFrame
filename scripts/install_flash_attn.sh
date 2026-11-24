#!/usr/bin/env bash
set -euo pipefail

python -m pip install --force-reinstall --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
