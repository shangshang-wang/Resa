from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading Tina checkpoints ...")
    snapshot_download(repo_id="Tina-Yi/R1-Distill-Qwen-1.5B-STILL",
                      local_dir=f"{CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_still")

    snapshot_download(repo_id="Tina-Yi/R1-Distill-Qwen-1.5B-DeepScaleR",
                      local_dir=f"{CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_deepscaler")