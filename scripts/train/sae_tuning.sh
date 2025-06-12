#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=3,4 # Set the GPUs you want to use
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
#MODEL_NAME="Qwen2.5-7B-Instruct"
#MODEL_CKPT="checkpoint-2000" # best RL-trained checkpoint
#MODEL_CKPT="checkpoint-0"
MODEL_CKPT="checkpoint-1000"
#MODEL_CKPT="checkpoint-1"
#MODEL_CKPT="checkpoint-10"
#MODEL_CKPT="checkpoint-50"
#MODEL_CKPT="checkpoint-100"
#MODEL_CKPT="checkpoint-500"
#MODEL_CKPT="checkpoint-3000"

#PT_DATASET_NAME="curated_still"
PT_DATASET_NAME="curated_deepscaler"
#PT_DATASET_NAME="curated_openthoughts3"

STUDENT_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
#STUDENT_MODEL_NAME="Qwen2.5-7B-Instruct"
DISTILL_TYPE="sft_r1_distill"
#STUDENT_MODEL_NAME="Qwen2.5-Math-1.5B"
#DISTILL_TYPE="sft_qwen_math"
#STUDENT_MODEL_NAME="Qwen2.5-1.5B"
#DISTILL_TYPE="sft_qwen"

#DISTILL_DATASET_NAME="curated_still"
DISTILL_DATASET_NAME="curated_deepscaler"
#DISTILL_DATASET_NAME="curated_open_rs1"
#DISTILL_DATASET_NAME="curated_open_r1"
#DISTILL_DATASET_NAME="curated_2thought"
#DISTILL_DATASET_NAME="curated_openthoughts3"

#SAE_TYPE="finetuned"
SAE_TYPE="reason_pretrained"

SAE_HOOKPOINT_LIST=("model.layers.12")
#SAE_HOOKPOINT_LIST=(
#    "model.layers.0" "model.layers.1" "model.layers.2" "model.layers.3"
#    "model.layers.4" "model.layers.5" "model.layers.6" "model.layers.7"
#    "model.layers.8" "model.layers.9" "model.layers.10" "model.layers.11"
#    "model.layers.12" "model.layers.13" "model.layers.14" "model.layers.15"
#    "model.layers.16" "model.layers.17" "model.layers.18" "model.layers.19"
#    "model.layers.20" "model.layers.21" "model.layers.22" "model.layers.23"
#    "model.layers.24" "model.layers.25" "model.layers.26" "model.layers.27"
#)

#SAE_TYPE="pretrained"
#SAE_HOOKPOINT_LIST=("layers.12.mlp")

PY_SCRIPT="./scripts/train/run_sae_based_distill.py"
PY_CONFIG="./recipes/${MODEL_NAME}/grpo/distill_${PT_DATASET_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

for SAE_HOOKPOINT in "${SAE_HOOKPOINT_LIST[@]}"; do
    if [ "${SAE_TYPE}" == "finetuned" ]; then
        echo ""
        echo "Distill ${STUDENT_MODEL_NAME} with finetuned SAE ${SAE_HOOKPOINT} from ${MODEL_NAME} (${MODEL_CKPT} post-trained with ${PT_DATASET_NAME}) on ${DISTILL_DATASET_NAME}"
        echo ""
    elif [ "${SAE_TYPE}" == "reason_pretrained" ]; then
        echo ""
        echo "Distill ${STUDENT_MODEL_NAME} with reason-pretrained SAE ${SAE_HOOKPOINT} from ${MODEL_NAME} (${MODEL_CKPT} post-trained with ${PT_DATASET_NAME}) on ${DISTILL_DATASET_NAME}"
        echo ""
    else
        echo ""
        echo "Distill ${STUDENT_MODEL_NAME} with pretrained SAE ${SAE_HOOKPOINT} from ${MODEL_NAME} on ${DISTILL_DATASET_NAME}"
        echo ""
    fi

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "${ACCELERATE_DS_CONFIG}" \
        --main_process_port=29501 \
        --num_processes="${GPU_COUNT}" \
            "${PY_SCRIPT}" --config "${PY_CONFIG}" \
                --host_model_checkpoint "${MODEL_CKPT}" \
                --student_model_name "${STUDENT_MODEL_NAME}" \
                --distill_dataset_name "${DISTILL_DATASET_NAME}" \
                --distill_type "${DISTILL_TYPE}" \
                --sae_hookpoint "${SAE_HOOKPOINT}" \
                --sae_type "${SAE_TYPE}"
done

echo "END TIME: $(date)"
echo "DONE"
