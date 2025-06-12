import datasets
from datasets import Dataset, load_dataset
from datetime import datetime
import logging
import os
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import sys
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import ModelConfig, SFTConfig, SFTTrainer, TrlParser, DataCollatorForCompletionOnlyLM

from resa.config import ModelPTConfig
from resa.post_train_hf.callback import FixedPromptEvaluationCallback, GradientClippingLoggerCallback, PushToHubRevisionCallback
from resa.post_train_hf.preprocess import make_conv_for_sft
from resa.utils.constant import SFT_POST_TRAIN_DATASET_MAP
from resa.utils.prompt import OPEN_R1_SYSTEM_PROMPT

import numpy as np

def check_sequence_lengths(
    dataset,
    tokenizer,
    text_column,
    num_proc=1
):
    """
    Tokenizes a dataset and analyzes the distribution of sequence lengths.

    Args:
        dataset: The Hugging Face Dataset object.
        tokenizer: The Hugging Face Tokenizer object.
        text_column: The name of the column containing the text data.
        num_proc: Number of processes to use for mapping.
    """
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Column '{text_column}' not found in dataset columns: {dataset.column_names}. "
            "Please set TEXT_COLUMN_NAME correctly."
        )

    # Function to tokenize and get length
    def get_token_length(example):
        # Tokenize without padding or truncation to get the true length
        tokenized_output = tokenizer(example[text_column], truncation=False, padding=False)
        return {"length": len(tokenized_output['input_ids'])}

    # Use map to efficiently calculate lengths for all examples
    # batched=True significantly speeds up tokenization
    try:
        lengths_dataset = dataset.map(
            get_token_length,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names # Keep only the new 'length' column
        )
        lengths = lengths_dataset["length"]
    except Exception as e:
         # Fallback to non-batched if batched fails (e.g., complex data)
        lengths_dataset = dataset.map(
            get_token_length,
            batched=False, # Try without batching
            num_proc=num_proc,
            remove_columns=dataset.column_names
        )
        lengths = lengths_dataset["length"]


    # Convert to NumPy array for efficient statistics
    lengths_np = np.array(lengths)

    if len(lengths_np) == 0:
        return

    # --- Calculate and Print Statistics ---
    print("\n--- Sequence Length Statistics ---")
    print(f"Number of examples: {len(lengths_np)}")
    print(f"Min length:         {np.min(lengths_np)}")
    print(f"Max length:         {np.max(lengths_np)}")
    print(f"Mean length:        {np.mean(lengths_np):.2f}")
    print(f"Median length:      {np.median(lengths_np)}")
    print(f"Standard Deviation: {np.std(lengths_np):.2f}")
    print("--- Percentiles ---")
    print(f"  50th (Median):  {np.percentile(lengths_np, 50)}")
    print(f"  75th:           {np.percentile(lengths_np, 75)}")
    print(f"  90th:           {np.percentile(lengths_np, 90)}")
    print(f"  95th:           {np.percentile(lengths_np, 95)}")
    print(f"  99th:           {np.percentile(lengths_np, 99)}")
    print(f"  99.9th:         {np.percentile(lengths_np, 99.9)}") # Useful for very long tail



def main():
    parser = TrlParser((ModelPTConfig, SFTConfig, ModelConfig))
    pt_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    os.environ["WANDB_PROJECT"] = "Tina_model_posttrain"

    ################
    # Set up logging
    ################

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Post training parameters {pt_args}")
    logger.info(f"Training parameters {training_args}")

    #####################
    # Set up output paths
    #####################

    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    model_name_or_path = model_args.model_name_or_path
    ckpt_dir = os.environ["CKPT_DIR"]
    ckpt_prefix = f"{ckpt_dir}/models/{model_name_or_path}"
    if model_args.use_peft:
        ckpt_postfix = f"{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    else:
        ckpt_postfix = f"full_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"

    model_args.model_name_or_path = f"{ckpt_prefix}/base"
    training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}"
    # training_args.hub_model_id = f"{training_args.hub_model_id}_{ckpt_postfix}"
    training_args.run_name = f"{model_name_or_path}_{ckpt_postfix}_{formatted_datetime}"

    # auto push to this hub
    training_args.hub_model_id = f"{training_args.hub_model_id}/{model_name_or_path}"

    #######################################################################
    # Load and preprocess dataset (tokenization is handled by SFT Trainer)
    #######################################################################

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model_post_train_dataset_name = SFT_POST_TRAIN_DATASET_MAP[pt_args.model_post_train_dataset_name]

    if pt_args.model_post_train_dataset_config is not None:
        train_dataset = load_dataset(model_post_train_dataset_name, split="train", name=pt_args.model_post_train_dataset_config)
    else:
        train_dataset = load_dataset(model_post_train_dataset_name, split="train")

    if 'question' not in train_dataset.column_names and 'problem' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column('problem', 'question')

    train_dataset = train_dataset.map(
        make_conv_for_sft,
        fn_kwargs={
            "dataset_name_or_path": model_post_train_dataset_name,
            "tokenizer": tokenizer,
            "trace_free": pt_args.trace_free,
        },
        batched=True)

    ################
    # Load the model
    ################

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True)

    if model_args.use_peft:
        logger.info(f"\n Using PEFT with {model_args.lora_r} rank, {model_args.lora_alpha} alpha, {model_args.lora_dropout} dropout.")
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            inference_mode=False,
            bias="none",
            task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, peft_config)

    ############################
    # Initialize the SFT Trainer
    ############################

    ## for Qwen models
    # instruction_template = "<｜User｜>"
    # response_template = "<｜Assistant｜>"
    # tokenizer.pad_token = "<|fim_pad|>"
    # data_collator = DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    #     mlm=False)

    callbacks = [
        FixedPromptEvaluationCallback(system_prompt=OPEN_R1_SYSTEM_PROMPT, eval_steps=training_args.save_steps),
        # PushToHubRevisionCallback(dataset_name=pt_args.model_post_train_dataset_name, use_peft=model_args.use_peft)
    ]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        # data_collator=data_collator,
        callbacks=callbacks)

    #########################
    # Training and Evaluation
    #########################

    logger.info(f"\nStarting training for {training_args.num_train_epochs} epochs.")

    # Check for last checkpoint
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        ckpt = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
        else:
            logger.info("\nNo checkpoint detected, starting training from scratch.")

    train_result = trainer.train(resume_from_checkpoint=ckpt)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    trainer.push_to_hub(commit_message=f"Add checkpoint {training_args.max_steps} post-trained on {pt_args.model_post_train_dataset_name}")

    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()