#!/bin/bash


echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

python ./scripts/set/run_download_base_model_sae.py
python ./scripts/set/run_download_tina_ckpts.py

echo "END TIME: $(date)"
echo "DONE"
