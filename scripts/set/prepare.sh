#!/bin/bash


MAMBA_ENV="resa"
eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

PY_SCRIPT="./scripts/set/run_download_model_sae.py"

echo ""
echo "Running script: ${PY_SCRIPT}"
echo ""

python "${PY_SCRIPT}"

echo "END TIME: $(date)"
echo "DONE"
