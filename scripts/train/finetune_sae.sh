#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

export OMP_NUM_THREADS=36

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_DATASET_NAME="still"
PT_TYPE="grpo"
SOURCE_MODEL_CKPT_LIST=("checkpoint-XXXX")

TRIGGER_DATASET_NAME="still" # deepscaler, open_rs3, openthoughts3

START_LAYER=1
END_LAYER=26
STEP=4

PY_SCRIPT="./scripts/train/finetune.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/train_sae.yaml"

for ((i=START_LAYER; i<=END_LAYER; i+=STEP)); do
    HOOKPOINT_ARRAY=()
    for ((j=0; j<STEP; j++)); do
      LAYER=$((i + j))
      if (( LAYER > END_LAYER )); then
        break
      fi
      HOOKPOINT_ARRAY+=("model.layers.${LAYER}")
    done

    echo ""
    echo "Fine-tuning SAE on model ${BASE_MODEL_NAME} (post-trained with ${PT_DATASET_NAME} via ${PT_TYPE}) with trigger dataset ${TRIGGER_DATASET_NAME}"
    echo "Override sae_hookpoints=${HOOKPOINT_ARRAY[*]}"
    echo ""

    ARGS=(--config "${PY_CONFIG}" --source_model_post_train_dataset_name "${PT_DATASET_NAME}" --trigger_dataset_name "${TRIGGER_DATASET_NAME}" --sae_hookpoints)
    for hook in "${HOOKPOINT_ARRAY[@]}"; do
        ARGS+=( "$hook")
    done
    ARGS+=( --source_model_checkpoints)
    for ckpt in "${SOURCE_MODEL_CKPT_LIST[@]}"; do
        ARGS+=( "${ckpt}")
    done

    torchrun --nproc_per_node="${GPU_COUNT}" --master-port=29500 "${PY_SCRIPT}" "${ARGS[@]}"
done

echo "END TIME: $(date)"
echo "DONE"