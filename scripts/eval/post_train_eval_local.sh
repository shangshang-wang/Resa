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

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
PT_TYPE="sft"

#DATASET_NAME="curated_still"
DATASET_NAME="curated_deepscaler"

CKPT_LIST=("checkpoint-XXXX")

# loop over all the checkpoints in the list
for CKPT in "${CKPT_LIST[@]}"; do
    echo "Running model post train merging base and adapter for checkpoint: $DATASET_NAME (${CKPT})"
    python ./scripts/eval/run_post_train_merge.py \
      --model_name "${MODEL_NAME}" \
      --adapter_type "${PT_TYPE}_${DATASET_NAME}" \
      --ckpt "${CKPT}" \

    MODEL_PATH="${CKPT_DIR}/models/${MODEL_NAME}/${PT_TYPE}_${DATASET_NAME}/${CKPT}-merged"

    # Set model arguments (ensure that MODEL_PATH, GPU_COUNT, OUTPUT_DIR, and MODEL are defined)
    if [ "${MODEL_NAME}" == "Qwen2.5-Math-1.5B" ]; then
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=4096,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}"
    elif [ "${MODEL_NAME}" == "Qwen2.5-1.5B" ]; then
#        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=131072,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:131072,temperature:0.6,top_p:0.95}"
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    else
        MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$GPU_COUNT,max_model_length=32768,gpu_memory_utilization=0.7,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    fi

    # Define an array of tasks to evaluate
    tasks=("aime24" "aime25" "amc23" "math_500" "minerva" "gpqa:diamond")

    # Loop over each task and evaluate
    for TASK in "${tasks[@]}"; do
        echo "Evaluating task: $TASK on model $MODEL_NAME post-trained with $DATASET_NAME (${CKPT})"
        lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
            --custom-tasks ./scripts/eval/run_post_train_eval.py \
            --use-chat-template \
            --output-dir "${OUTPUT_DIR}/${TASK}/${DATASET_NAME}"
    done
done

echo "END TIME: $(date)"
echo "DONE"
