#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "GPU_COUNT: ${GPU_COUNT}"
echo ""

export OMP_NUM_THREADS=36

BASE_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-1.5B
SAE_DATASET_NAME="EleutherAI/SmolLM2-135M-10B" # default dataset used by EleutherAI

export WANDB_PROJECT="Resa_train_sae"

START_LAYER=1
END_LAYER=26
STEP=1

for ((i=START_LAYER; i<=END_LAYER; i+=STEP)); do
    SAE_HOOKPOINT_LIST=()
    for ((j=0; j<STEP; j++)); do
      LAYER=$((i + j))
      if (( LAYER > END_LAYER )); then
        break
      fi
      SAE_HOOKPOINT_LIST+=("layers.${LAYER}")
    done

    echo ""
    echo "Train SAE from scratch on model ${BASE_MODEL_NAME} at ${SAE_HOOKPOINT_LIST[*]} using dataset ${SAE_DATASET_NAME}"
    echo ""

    ARGS=(--expansion_factor 64 --num_latents 65536 --batch_size 32 --run_name "${BASE_MODEL_NAME}_${i}" --hookpoints)
    for hook in "${SAE_HOOKPOINT_LIST[@]}"; do
        ARGS+=( "$hook")
    done

    torchrun --nproc_per_node "${GPU_COUNT}" -m sparsify "${BASE_MODEL_NAME}" "${SAE_DATASET_NAME}" "${ARGS[@]}"
done

echo "END TIME: $(date)"
echo "DONE"