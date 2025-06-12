from contextlib import nullcontext, redirect_stdout
import datasets
from datasets import load_dataset
from datetime import datetime
import gc
import logging
import os
from pathlib import Path
import re
from safetensors.torch import load_model
import shutil
from sparsify import Sae, SaeConfig, Trainer, TrainConfig
from sparsify.data import chunk_and_tokenize
import sys
import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import TrlParser
import wandb

from resa.config import SAEConfig
from resa.sae.preprocess import make_conv_for_sae_sft, make_conv_for_sae_grpo
from resa.utils.constant import RL_POST_TRAIN_DATASET_MAP, SFT_POST_TRAIN_DATASET_MAP


def rename_layer(layer_name):
    return re.sub(r"model\.layers\.(\d+)", r"layers.\1.mlp", layer_name)

def load_llm_rank(model_name_or_path, rank):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map={"": f"cuda:{rank}"},
        torch_dtype=torch.bfloat16)
    return model


if __name__ == "__main__":
    parser = TrlParser((SAEConfig))
    (sae_args,) = parser.parse_args_and_config()
    set_seed(sae_args.seed)

    os.environ["WANDB_PROJECT"] = "Resa_sae_finetune"

    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    ckpt_dir = os.environ['CKPT_DIR']
    inspect_model_prefix = f"{ckpt_dir}/models/{sae_args.host_model_name}"

    ################
    # Set up logging
    ################

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])
    log_level = "INFO"
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"SAE finetune parameters {sae_args}")

    ################
    # Load datasets
    ################

    tokenizer = AutoTokenizer.from_pretrained(f"{inspect_model_prefix}/base")

    if sae_args.host_model_post_train_dataset_name in RL_POST_TRAIN_DATASET_MAP.keys():
        assert sae_args.host_model_post_train_type == "grpo"
        host_model_post_train_dataset_name = RL_POST_TRAIN_DATASET_MAP[sae_args.host_model_post_train_dataset_name]
        if "thoughts" in sae_args.host_model_post_train_dataset_name:
            sae_finetune_dataset = load_dataset(host_model_post_train_dataset_name, "OpenThoughts-114k-math-default", split="train")
        else:
            sae_finetune_dataset = load_dataset(host_model_post_train_dataset_name, split="train")
        sae_finetune_dataset = sae_finetune_dataset.map(
            make_conv_for_sae_grpo,
            batched=True)
    elif sae_args.host_model_post_train_dataset_name in SFT_POST_TRAIN_DATASET_MAP.keys():
        assert sae_args.host_model_post_train_type == "sft"
        host_model_post_train_dataset_name = SFT_POST_TRAIN_DATASET_MAP[sae_args.host_model_post_train_dataset_name]
        sae_finetune_dataset = load_dataset(host_model_post_train_dataset_name, split="train")
        sae_finetune_dataset = sae_finetune_dataset.map(
            make_conv_for_sae_sft,
            fn_kwargs={
                "dataset_name_or_path": host_model_post_train_dataset_name,
                "model_name_or_path": sae_args.host_model_name,
                "tokenizer": tokenizer},
            batched=True)
    else:
        raise logger.error(f"Dataset {sae_args.host_model_post_train_dataset_name} not found in RL_POST_TRAIN_DATASET_MAP or SFT_POST_TRAIN_DATASET_MAP.")

    # looking for the "text" column
    tokenized_train_dataset = chunk_and_tokenize(sae_finetune_dataset, tokenizer)
    tokenized_train_dataset = tokenized_train_dataset.with_format("torch")

    ######################
    # Preaparation for DDP
    ######################

    postfix = f"{sae_args.host_model_post_train_type}_{sae_args.host_model_post_train_dataset_name}"
    inspect_model_dir = f"{inspect_model_prefix}/{postfix}"

    if sae_args.host_model_checkpoints == []:
        logger.error("Please provide the target model checkpoints")
    else:
        inspect_model_ckpts = sae_args.host_model_checkpoints

    # the following will be used when you want to fine-tune over all available checkpoints
    # inspect_model_ckpts.extend([
    #     d for d in os.listdir(inspect_model_dir)
    #     if os.path.isdir(os.path.join(inspect_model_dir, d))])
    # inspect_model_ckpts = inspect_model_ckpts[::-1]

    inspect_model_list = [ os.path.join(inspect_model_dir, ckpt) for ckpt in inspect_model_ckpts ]

    ##################
    # Fine-tuning SAEs
    ##################

    # check https://github.com/EleutherAI/sparsify/blob/5de51b2250806da1a94c21395b43dc4bdb7754b5/sparsify/config.py
    sae_trainer_cfg = TrainConfig(
        SaeConfig(expansion_factor=sae_args.sae_expansion_factor,
                  normalize_decoder=True,
                  num_latents=sae_args.sae_num_latents,
                  k=32,
                  multi_topk=False,
                  skip_connection=False,
                  transcode=False),
        batch_size=16,
        grad_acc_steps=1,
        micro_acc_steps=1,
        lr=None,  # auto set based on number of latents
        # lr_warmup_steps=100,
        dead_feature_threshold=10_000_000,
        hookpoints=sae_args.sae_hookpoints,
        init_seeds=[sae_args.seed],
        # save_every=100,
        distribute_modules=False,  # must be False for multi-gpu DDP
        log_to_wandb=True,
        wandb_log_frequency=1
    )

    # currently no resume from ckpt is supported, but the training itself is around 1-2 hours
    for inspect_model, ckpt in zip(inspect_model_list, inspect_model_ckpts):
        current_time = datetime.now()
        formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        run_name = f"{sae_args.sae_name}_{sae_args.host_model_post_train_type}_{sae_args.host_model_post_train_dataset_name}_{ckpt}_{formatted_datetime}"
        sae_trainer_cfg.run_name = run_name

        sae_trainer_cfg.output_dir = f"{ckpt_dir}/saes/{sae_args.sae_name}/{postfix}/{ckpt}"

        if ddp:
            torch.cuda.set_device(int(local_rank))
            if not dist.is_initialized():
                dist.init_process_group("nccl", device_id=torch.device(rank))
            if rank == 0:
                logger.info(f"Using DDP across {dist.get_world_size()} GPUs.")

        with nullcontext() if rank == 0 else redirect_stdout(None):
            if not ddp or rank == 0:
                llm = load_llm_rank(inspect_model, rank)
            if ddp:
                dist.barrier()
                if rank != 0:
                    llm = load_llm_rank(inspect_model, rank)
                tokenized_dataset = tokenized_train_dataset.shard(dist.get_world_size(), rank)

            trainer = Trainer(sae_trainer_cfg, tokenized_dataset, llm)
            for name, sae in trainer.saes.items():
                name = rename_layer(name)
                sae_name_or_path = f"{ckpt_dir}/saes/{sae_args.sae_name}/base/{name}/sae.safetensors"
                load_model(sae, sae_name_or_path,
                           device=str(llm.device))

            logger.info("Start training SAEs at hookpoints: %s", sae_trainer_cfg.hookpoints)
            trainer.fit()

            # --- CLEANUP START ---
            if wandb.run is not None:
                wandb.finish()

            del trainer
            del llm
            del tokenized_dataset

            torch.cuda.empty_cache()
            gc.collect()

            if ddp:
                dist.barrier()
