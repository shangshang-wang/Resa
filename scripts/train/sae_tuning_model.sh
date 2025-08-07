#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}, make sure it is 2 GPUs."
echo ""

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_DATASET_NAME="still" # deepscaler, openthoughts3
PT_TYPE="grpo" # grpo, sft, base (meaning using the base model without any post-training)
SOURCE_MODEL_CKPT="checkpoint-XXXX"

SAE_NAME="sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
SAE_HOOKPOINT_LIST=(
    "model.layers.1" "model.layers.2" "model.layers.3" "model.layers.4"
    "model.layers.5" "model.layers.6" "model.layers.7" "model.layers.8"
    "model.layers.9" "model.layers.10" "model.layers.11" "model.layers.12"
    "model.layers.13" "model.layers.14" "model.layers.15" "model.layers.16"
    "model.layers.17" "model.layers.18" "model.layers.19" "model.layers.20"
    "model.layers.21" "model.layers.22" "model.layers.23" "model.layers.24"
    "model.layers.25" "model.layers.26"
)
TRIGGER_DATASET_NAME="still"
SAE_TYPE="finetuned" # trained_from_scratch

TARGET_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B" # Qwen2.5-Math-1.5B, Qwen2.5-1.5B
ELICITATION_DATASET_NAME="still" # deepscaler, open_rs1, open_r1, 2thought, openthoughts3


PY_SCRIPT="./scripts/train/sae_tuning.py"
PY_CONFIG="./recipes/${BASE_MODEL_NAME}/${PT_TYPE}/sae_tuning.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

for SAE_HOOKPOINT in "${SAE_HOOKPOINT_LIST[@]}"; do
    echo ""
    echo "Tuning ${TARGET_MODEL_NAME} with ${SAE_TYPE} SAE (trained on ${TRIGGER_DATASET_NAME}) at ${SAE_HOOKPOINT} from ${BASE_MODEL_NAME} (${SOURCE_MODEL_CKPT} post-trained with ${PT_DATASET_NAME} via ${PT_TYPE}) on ${ELICITATION_DATASET_NAME}"
    echo ""

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29500 \
        --num_processes="${GPU_COUNT}" \
            "${PY_SCRIPT}" --config "${PY_CONFIG}" \
                --base_model_name "${BASE_MODEL_NAME}" \
                --source_model_post_train_dataset_name "${PT_DATASET_NAME}" \
                --source_model_post_train_type "${PT_TYPE}" \
                --source_model_checkpoint "${SOURCE_MODEL_CKPT}" \
                --sae_name "${SAE_NAME}" \
                --sae_hookpoint "${SAE_HOOKPOINT}" \
                --trigger_dataset_name "${TRIGGER_DATASET_NAME}" \
                --sae_type "${SAE_TYPE}" \
                --target_model_name "${TARGET_MODEL_NAME}" \
                --elicitation_dataset_name "${ELICITATION_DATASET_NAME}"
done

echo "END TIME: $(date)"
echo "DONE"