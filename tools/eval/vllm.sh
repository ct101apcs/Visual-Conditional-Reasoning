#!/usr/bin/env bash
# Run a script with the vLLM venv. Triton compiles a small C helper at start-up and needs Python.h;
# the system lacks python3-dev, so the matching 3.10 headers from miniconda are used instead.
export C_INCLUDE_PATH=/home/stackops/miniconda3/include/python3.10${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export HF_HUB_OFFLINE=1          # Qwen3-VL-8B-Instruct is in the local HF cache
exec /home/stackops/.venvs/anchor-vllm/bin/python "$@"
