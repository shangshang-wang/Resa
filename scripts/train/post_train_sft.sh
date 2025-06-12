#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=3,6 # Set the GPUs you want to use
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
#MODEL_NAME="Qwen2.5-Math-1.5B"
#MODEL_NAME="Qwen2.5-1.5B"

DATASET_NAME="curated_still"
#DATASET_NAME="curated_deepscaler" # for trace-based ablation

PY_SCRIPT="./resa/post_train_hf/sft.py"
PY_CONFIG="./recipes/${MODEL_NAME}/sft/model_${DATASET_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${MODEL_NAME} with dataset ${DATASET_NAME}"
echo ""

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file "${ACCELERATE_DS_CONFIG}" \
    --main_process_port=29500 \
    --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}"

echo "END TIME: $(date)"
echo "DONE"
