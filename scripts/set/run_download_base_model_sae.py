from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ...")
    snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      local_dir=f"{CKPT_DIR}/models/DeepSeek-R1-Distill-Qwen-1.5B/base")

    print("Downloading Qwen/Qwen2.5-Math-1.5B ...")
    snapshot_download(repo_id="Qwen/Qwen2.5-Math-1.5B",
                      local_dir=f"{CKPT_DIR}/models/Qwen2.5-Math-1.5B/base")
    
    print("Downloading Qwen/Qwen2.5-1.5B ...")
    snapshot_download(repo_id="Qwen/Qwen2.5-1.5B",
                      local_dir=f"{CKPT_DIR}/models/Qwen2.5-1.5B/base")
    
    print("Downloading EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k ...")
    snapshot_download(repo_id="EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",
                        local_dir=f"{CKPT_DIR}/saes/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k/base")