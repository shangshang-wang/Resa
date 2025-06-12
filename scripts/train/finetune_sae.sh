#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

MODEL_NAME="DeepSeek-R1-Distill-Qwen-1.5B"
POST_TRAIN_TYPE="grpo"

DATASET_NAME="curated_still"
#DATASET_NAME="curated_deepscaler"
#DATASET_NAME="curated_open_rs3"

START_LAYER=19
END_LAYER=19
STEP=4

PY_SCRIPT="./scripts/train/run_finetune_sae.py"
PY_CONFIG="./recipes/${MODEL_NAME}/${POST_TRAIN_TYPE}/sae_${DATASET_NAME}.yaml"

for ((i=START_LAYER; i<=END_LAYER; i+=STEP)); do
    # Build sae_hookpoints override string
    HOOKPOINT_ARRAY=()
    for ((j=0; j<STEP; j++)); do
      LAYER=$((i + j))
      if (( LAYER > END_LAYER )); then
        break
      fi
      HOOKPOINT_ARRAY+=("model.layers.${LAYER}")
    done

    echo ""
    echo "Running ${PY_SCRIPT} on model ${MODEL_NAME} post-trained with dataset ${DATASET_NAME}"
    echo "Override sae_hookpoints=${HOOKPOINT_ARRAY[*]}"
    echo ""

    ARGS=(--config "${PY_CONFIG}" --sae_hookpoints)
    for hook in "${HOOKPOINT_ARRAY[@]}"; do
        ARGS+=( "$hook")
    done

    torchrun --nproc_per_node="${GPU_COUNT}" --master-port=29500 "${PY_SCRIPT}" "${ARGS[@]}"
done

echo "END TIME: $(date)"
echo "DONE"
