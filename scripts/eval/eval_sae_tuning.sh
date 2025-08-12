#!/bin/bash


MAMBA_ENV="resa_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}, make sure using 2 GPUs."
echo ""

BASE_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
PT_DATASET_NAME="still" # deepscaler
SOURCE_MODEL_CKPT="checkpoint-XXXX"

SAE_TYPE="finetuned"
TRIGGER_DATASET_NAME="still"
SAE_HOOKPOINT="model.layers.12"

TARGET_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B" # Qwen2.5-Math-1.5B, Qwen2.5-1.5B
ELICITATION_DATASET_NAME="still" # deepscaler, open_rs1, open_r1, 2thought
TARGET_MODEL_CKPT_LIST=("checkpoint-XXXX")

for TARGET_MODEL_CKPT in "${TARGET_MODEL_CKPT_LIST[@]}"; do
    echo ""
    echo "Eval ${TARGET_MODEL_NAME} with ${SAE_TYPE} SAE (trained on ${TRIGGER_DATASET_NAME}) at ${SAE_HOOKPOINT} from ${BASE_MODEL_NAME} (${SOURCE_MODEL_CKPT} post-trained with ${PT_DATASET_NAME} via ${PT_TYPE}) on ${ELICITATION_DATASET_NAME}"
    echo ""

    python ./resa/sae/merge_sae_tuned_models.py \
        --base_model_name "${BASE_MODEL_NAME}" \
        --source_model_adapter_type "${PT_TYPE}_${PT_DATASET_NAME}" \
        --source_model_checkpoint "${SOURCE_MODEL_CKPT}" \
        --sae_type "${SAE_TYPE}" \
        --trigger_dataset_name "${TRIGGER_DATASET_NAME}" \
        --sae_hookpoint "${SAE_HOOKPOINT}" \
        --target_model_name "${TARGET_MODEL_NAME}" \
        --elicitation_dataset_name "${ELICITATION_DATASET_NAME}" \
        --target_model_ckpt "${TARGET_MODEL_CKPT}"

    MODEL_PATH="${CKPT_DIR}/models/${TARGET_MODEL_NAME}/sae_tuning_${ELICITATION_DATASET_NAME}/${BASE_MODEL_NAME}_${PT_TYPE}_${PT_DATASET_NAME}_${SOURCE_MODEL_CKPT}/${SAE_TYPE}_${TRIGGER_DATASET_NAME}_${SAE_HOOKPOINT}/${TARGET_MODEL_CKPT}-merged"

    if [ "${TARGET_MODEL_NAME}" == "Qwen/Qwen2.5-Math-1.5B" ]; then
        MAX_MODEL_LENGTH=4096
        MAX_NEW_TOKENS=4096
    elif [ "${TARGET_MODEL_NAME}" == "Qwen/Qwen2.5-1.5B" ]; then
        MAX_MODEL_LENGTH=32768 # 131072
        MAX_NEW_TOKENS=32768 # 131072
    else
        MAX_MODEL_LENGTH=32768
        MAX_NEW_TOKENS=32768
    fi

    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,data_parallel_size=${GPU_COUNT},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0.6,top_p:0.95}"

    tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond") # extra task: olympiadbench

    for TASK in "${tasks[@]}"; do
        echo "Evaluating task: ${TASK}"
        lighteval vllm "${MODEL_ARGS}" "custom|${TASK}|0|0" \
            --custom-tasks ./scripts/eval/run_eval_custom_tasks.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}/${TARGET_MODEL_NAME}"
    done
done

echo "END TIME: $(date)"
echo "DONE"