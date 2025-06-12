#!/bin/bash


MAMBA_ENV="resa_eval"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1 # make sure all evaluation run on 2 GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: $GPU_COUNT, make sure using 2 GPUs."
echo ""

TEACHER_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="grpo"
TEACHER_MODEL_CKPT="checkpoint-2000"
#TEACHER_MODEL_CKPT="checkpoint-1000"

#TEACHER_MODEL_CKPT="checkpoint-0"
#TEACHER_MODEL_CKPT="checkpoint-1"
#TEACHER_MODEL_CKPT="checkpoint-10"
#TEACHER_MODEL_CKPT="checkpoint-50"
#TEACHER_MODEL_CKPT="checkpoint-100"
#TEACHER_MODEL_CKPT="checkpoint-500"
#TEACHER_MODEL_CKPT="checkpoint-3000"

PT_DATASET_NAME="curated_still"
#PT_DATASET_NAME="curated_deepscaler"

DISTILL_DATASET_NAME="curated_still"
#DISTILL_DATASET_NAME="curated_deepscaler"
#DISTILL_DATASET_NAME="curated_open_rs1"
#DISTILL_DATASET_NAME="curated_open_r1"
#DISTILL_DATASET_NAME="curated_2thought"

DISTILL_SAE_HOOKPOINT="model.layers.12"
STUDENT_MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
DISTILL_TYPE="sft_r1_distill"
#STUDENT_MODEL_NAME="Qwen2.5-Math-1.5B"
#DISTILL_TYPE="sft_qwen_math"
#STUDENT_MODEL_NAME="Qwen2.5-1.5B"
#DISTILL_TYPE="sft_qwen"

#DISTILL_SAE_HOOKPOINT="layers.12.mlp"
#DISTILL_TYPE="base"

STUDENT_MODEL_CKPT_LIST=("checkpoint-XXXX")

# loop over all the checkpoints in the list
for STUDENT_MODEL_CKPT in "${STUDENT_MODEL_CKPT_LIST[@]}"; do
    if [ "${DISTILL_TYPE}" == "base" ]; then
        echo ""
        echo "Eval student model ${STUDENT_MODEL_NAME} (${STUDENT_MODEL_CKPT} on ${DISTILL_DATASET_NAME}) distilled from pretrained SAE"
        echo ""
    else
        echo ""
        echo "Eval student model ${STUDENT_MODEL_NAME} (${STUDENT_MODEL_CKPT} on ${DISTILL_DATASET_NAME}) distilled from SAE at teacher model ${TEACHER_MODEL_NAME} (${TEACHER_MODEL_CKPT}) post-trained on ${PT_DATASET_NAME}"
        echo ""
    fi

    python ./scripts/eval/run_post_train_merge_distill.py \
      --teacher_model_name "${TEACHER_MODEL_NAME}" \
      --teacher_model_ckpt "${TEACHER_MODEL_CKPT}" \
      --teacher_model_adapter_name "${PT_TYPE}_${PT_DATASET_NAME}" \
      --student_model_name "${STUDENT_MODEL_NAME}" \
      --student_model_ckpt "${STUDENT_MODEL_CKPT}" \
      --distill_dataset_name "${DISTILL_DATASET_NAME}" \
      --distill_sae_hookpoint "${DISTILL_SAE_HOOKPOINT}" \
      --distill_type "${DISTILL_TYPE}"

    # Set model arguments (ensure that MODEL_PATH, GPU_COUNT, OUTPUT_DIR, and MODEL are defined)
    if [ "${DISTILL_TYPE}" == "base" ]; then
        MODEL_PATH="${CKPT_DIR}/models/${STUDENT_MODEL_NAME}/base/distill/${DISTILL_DATASET_NAME}/${DISTILL_SAE_HOOKPOINT}/${STUDENT_MODEL_CKPT}-merged"
    else
        MODEL_PATH="${CKPT_DIR}/models/${TEACHER_MODEL_NAME}/${PT_TYPE}_${PT_DATASET_NAME}/${TEACHER_MODEL_CKPT}/distill/${DISTILL_DATASET_NAME}/${DISTILL_SAE_HOOKPOINT}/${DISTILL_TYPE}/${STUDENT_MODEL_CKPT}-merged"
    fi

    if [ "${STUDENT_MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=4096,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
    elif [ "${STUDENT_MODEL_NAME}" == "Qwen2.5-1.5B" ]; then
        # MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=131072,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:131072,temperature:0.6,top_p:0.95}"
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    else
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    fi

    # Define an array of tasks to evaluate
    tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond")

    # Loop over each task and evaluate
    for TASK in "${tasks[@]}"; do
        echo "Evaluating task: $TASK"
        lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
            --custom-tasks ./scripts/eval/run_post_train_eval.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}/${STUDENT_MODEL_NAME}"
    done
done

echo "END TIME: $(date)"
echo "DONE"
