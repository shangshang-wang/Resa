from huggingface_hub import snapshot_download
import os


if __name__ == "__main__":
    CKPT_DIR = os.environ['CKPT_DIR']

    print("Downloading OctoThinker/OctoThinker-3B-Long-Base ...")
    snapshot_download(repo_id="OctoThinker/OctoThinker-3B-Long-Base",
                      local_dir=f"{CKPT_DIR}/models/OctoThinker-3B-Long-Base/base")

    print("Downloading OctoThinker/OctoThinker-3B-Hybrid-Base ...")
    snapshot_download(repo_id="OctoThinker/OctoThinker-3B-Hybrid-Base",
                      local_dir=f"{CKPT_DIR}/models/OctoThinker-3B-Hybrid-Base/base")

    print("Downloading OctoThinker/OctoThinker-3B-Short-Base ...")
    snapshot_download(repo_id="OctoThinker/OctoThinker-3B-Short-Base",
                      local_dir=f"{CKPT_DIR}/models/OctoThinker-3B-Short-Base/base")