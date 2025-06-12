import argparse
import datasets
from datasets import load_dataset
from datetime import datetime
import json
import logging
import os
from peft import get_peft_model, LoraConfig, TaskType
from sparsify import Sae
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import TrlParser
from typing import Generator, List, Tuple, Optional, Any, Dict
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accelerate_set_seed
import torch.nn.functional as F # Added for F.mse_loss

from resa.config import DistillConfig
from resa.sae.preprocess import make_conv_for_sae_grpo
from resa.utils.constant import RL_POST_TRAIN_DATASET_MAP
from sparsify.data import chunk_and_tokenize


class TokenizedTextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        new_tensor = self.tokenized_data[idx]["input_ids"].clone().detach().to(torch.long)
        return {"input_ids": new_tensor}


class GlobalSAE:
    use_sae = True

def get_sae_hook(sae_module):
    def sae_reconstruction_hook(module, input, output):
        if not GlobalSAE.use_sae:
            return output

        original_shape = output[0].shape
        output_tensor = output[0]

        # Ensure SAE runs on the same device as the activation
        current_device = output_tensor.device
        sae_module.to(current_device) # Move SAE if necessary

        flat_output = output_tensor.reshape(-1, original_shape[-1])
        # Run SAE in eval mode and without gradients for the hook forward pass
        with torch.no_grad():
             sae_output_dict = sae_module(flat_output) # Assumes sae_module is callable
             reconstructed_output = sae_output_dict.sae_out

        reconstructed_output = reconstructed_output.reshape(original_shape)

        return (reconstructed_output,) + output[1:]

    return sae_reconstruction_hook


# Includes multi-epoch training loop
def train_model(args, accelerator, peft_model, sae_module, train_dataloader, tokenizer, run_name, output_dir):
    global_step = 0

    # Optimizer needs to be defined before preparing with accelerator
    optimizer = optim.AdamW(peft_model.parameters(), lr=args.learning_rate)

    peft_model, optimizer, train_dataloader = accelerator.prepare(peft_model, optimizer, train_dataloader)

    # Register main SAE hook *after* model preparation
    unwrapped_peft_model = accelerator.unwrap_model(peft_model)
    model_layers = unwrapped_peft_model.base_model.model.model.layers # Adjust path if model structure differs

    if args.sae_type == "finetuned":
        sae_layer_index = int(args.sae_hookpoint.split(".")[-1])
    elif args.sae_type == "reason_pretrained":
        sae_layer_index = int(args.sae_hookpoint.split(".")[-1])
    elif args.sae_type == "pretrained":
        sae_layer_index = int(args.sae_hookpoint.split(".")[1])
    else:
        raise ValueError(f"Invalid SAE type: {args.sae_type}. Expected 'finetuned', 'pretrained', 'reason_pretrained'.")

    sae_hook_handle = model_layers[sae_layer_index].register_forward_hook(get_sae_hook(sae_module))
    accelerator.print(f"Registered main SAE hook on layer {sae_layer_index}")

    # --- Hook for capturing activations for SAE loss calculation ---
    # This list will store the activation from the target layer during the base model pass
    captured_activations_for_sae_loss = []
    def capture_activation_hook(module, input, output):
        # Detach and clone to prevent holding onto the graph unnecessarily for logging
        # output[0] is the main hidden state tensor
        captured_activations_for_sae_loss.append(output[0].detach().clone())

    # Get the specific layer from the base model to attach the capture hook
    # Ensure the path to layers is correct for your base model structure
    base_model_target_layer = unwrapped_peft_model.base_model.model.model.layers[sae_layer_index]
    # --- End Hook Setup ---

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    accelerator.print(f"***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataloader.dataset)}")
    accelerator.print(f"  Num Epochs = {args.num_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed) = {args.batch_size * accelerator.num_processes}")
    accelerator.print(f"  Total optimization steps = {max_train_steps}")
    accelerator.print(f"  Steps per epoch = {num_update_steps_per_epoch}")
    accelerator.print(f"  Initial learning rate = {args.learning_rate}")
    accelerator.print(f"  Saving checkpoint every {args.save_steps} steps")

    # Use tqdm only on the main process, set total to max_train_steps
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_main_process, desc="Training Progress")

    try:
        # --- Epoch Loop ---
        for epoch in range(args.num_epochs):
            peft_model.train()
            total_epoch_loss = 0
            sae_module.eval() # Ensure SAE is in eval mode for reconstruction loss calculation

            # --- Inner Step Loop (over dataloader) ---
            for step, batch in enumerate(train_dataloader):
                inputs = batch["input_ids"]
                grad_norm = None # Initialize grad_norm for logging
                avg_sae_recon_loss = 0.0 # Initialize sae loss for logging

                with accelerator.accumulate(peft_model):
                    optimizer.zero_grad()

                    # --- Base model pass (teacher) ---
                    # Temporarily add hook to capture activations for SAE loss calculation
                    captured_activations_for_sae_loss.clear() # Clear previous step's capture
                    capture_hook_handle = base_model_target_layer.register_forward_hook(capture_activation_hook)

                    GlobalSAE.use_sae = False # Ensure SAE hook is off for base pass
                    with torch.no_grad():
                        # Use the unwrapped model for the base pass
                        # Ensure adapter is disabled correctly if peft_model is used directly
                        # Using unwrapped_peft_model is clearer here
                        with unwrapped_peft_model.disable_adapter():
                            base_outputs = unwrapped_peft_model(inputs)
                            base_logits = base_outputs.logits
                            # Keep base_probs on device for KLDiv
                            base_probs = torch.nn.functional.softmax(base_logits, dim=-1).to(base_logits.dtype)

                    # Remove the temporary capture hook immediately after use
                    capture_hook_handle.remove()

                    # --- Calculate SAE Reconstruction Loss (using captured activation) ---
                    if captured_activations_for_sae_loss:
                        original_activation = captured_activations_for_sae_loss[0]
                        original_shape = original_activation.shape
                        flat_original_activation = original_activation.reshape(-1, original_shape[-1])

                        # Ensure SAE is on the correct device (might be redundant if moved outside loop, but safe)
                        sae_module.to(flat_original_activation.device)

                        with torch.no_grad(): # Ensure no gradients for this calculation
                            sae_output_dict = sae_module(flat_original_activation)
                            reconstructed_activation_flat = sae_output_dict.sae_out

                        # Calculate MSE Loss
                        sae_recon_loss = F.mse_loss(reconstructed_activation_flat, flat_original_activation)
                        avg_sae_recon_loss = accelerator.gather(sae_recon_loss).mean().item() # Gather and average across devices

                    else:
                        # This shouldn't happen if the hook works correctly, but handle it just in case
                        accelerator.print("Warning: No activations captured for SAE loss calculation this step.", main_process_only=True)
                        avg_sae_recon_loss = 0.0 # Or perhaps float('nan')

                    # --- PEFT model pass (student with main SAE hook active) ---
                    GlobalSAE.use_sae = True # Turn main SAE hook on
                    # The prepared peft_model already has the main SAE hook registered via unwrapped_peft_model
                    peft_outputs = peft_model(inputs)
                    peft_logits = peft_outputs.logits
                    peft_log_probs = torch.nn.functional.log_softmax(peft_logits, dim=-1)

                    # --- Loss Calculation (KL Divergence) ---
                    loss = torch.nn.functional.kl_div(
                        peft_log_probs,
                        base_probs,
                        reduction='batchmean',
                        log_target=False
                    )

                    # --- Backpropagation ---
                    accelerator.backward(loss)

                    # --- Gradient Clipping & Capture Norm ---
                    if accelerator.sync_gradients:
                        # Capture the norm *before* clipping (clip_grad_norm_ returns this)
                        grad_norm = accelerator.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                        if grad_norm is not None: # grad_norm is tensor, convert for logging
                           grad_norm = grad_norm.item()

                    # --- Optimizer Step ---
                    optimizer.step()

                    # --- Logging ---
                    avg_loss = accelerator.gather(loss).mean().item() # Gather and average KL loss
                    total_epoch_loss += avg_loss

                    progress_bar.update(1)
                    progress_bar.set_postfix({"kl_loss": avg_loss, "epoch": epoch})
                    global_step += 1

                    # Log metrics only on the main process
                    if accelerator.is_main_process:
                        log_dict = {
                            "step_kl_loss": avg_loss,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "epoch": epoch,
                            "sae_reconstruction_loss": avg_sae_recon_loss, # Log SAE loss
                        }
                        if grad_norm is not None:
                            log_dict["grad_norm"] = grad_norm # Log grad norm if available

                        accelerator.log(log_dict, step=global_step) # Log with global_step

                    # --- Checkpointing ---
                    if global_step % args.save_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            unwrapped_peft_model_to_save = accelerator.unwrap_model(peft_model)
                            unwrapped_peft_model_to_save.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)
                            accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


                    # Minimal cleanup within loop
                    del inputs, base_outputs, base_logits, base_probs, peft_outputs, peft_logits, peft_log_probs, loss
                    if 'original_activation' in locals():
                        del original_activation
                    if 'reconstructed_activation_flat' in locals():
                        del reconstructed_activation_flat
                    if 'flat_original_activation' in locals():
                        del flat_original_activation
                    if 'sae_output_dict' in locals():
                        del sae_output_dict


            # --- End of Epoch Logging ---
            avg_epoch_loss = total_epoch_loss / num_update_steps_per_epoch
            accelerator.print(f"Epoch {epoch} finished. Average KL Loss: {avg_epoch_loss:.4f}")
            if accelerator.is_main_process:
                accelerator.log({"epoch_kl_loss": avg_epoch_loss}, step=global_step)

    finally:
        progress_bar.close()
        # Remove main SAE hook cleanly
        if 'sae_hook_handle' in locals() and sae_hook_handle:
            sae_hook_handle.remove()
            accelerator.print("Removed main SAE hook.")
        # Ensure capture hook is removed if loop exited unexpectedly
        if 'capture_hook_handle' in locals() and capture_hook_handle:
             try:
                 capture_hook_handle.remove()
                 accelerator.print("Ensured removal of temporary activation capture hook.")
             except Exception as e:
                 accelerator.print(f"Could not remove capture hook: {e}", main_process_only=True)


        # --- Final Model Saving ---
        # ... (final saving code remains the same)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            final_model_to_save = accelerator.unwrap_model(peft_model)
            final_model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            accelerator.print(f"Final model saved to {output_dir}")


def main():
    parser = TrlParser((DistillConfig))
    (args,) = parser.parse_args_and_config()

    accelerator = Accelerator(log_with="wandb")
    accelerate_set_seed(args.seed)

    os.environ["WANDB_PROJECT"] = "Resa_model_distill"

    ################
    # Set up logging
    ################

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO, # Root logger level
        handlers=[logging.StreamHandler()], # Basic handler
    )
    logger = logging.getLogger(__name__)
    # Control verbosity for other libraries
    if accelerator.is_main_process: # Only set verbosity on main process
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(logging.ERROR) # Reduce logging on non-main processes

    logger.info(f"Process {accelerator.process_index} starting...")
    logger.info(f"Using {accelerator.num_processes} processes.")
    logger.info(f"Distillation parameters {args}")

    ##############
    # Set up paths
    ##############

    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    # Ensure CKPT_DIR is set
    ckpt_dir = os.environ.get("CKPT_DIR", "./checkpoints") # Provide a default
    if not os.path.exists(ckpt_dir) and accelerator.is_main_process:
        os.makedirs(ckpt_dir)
    ckpt_postfix = f"{args.host_model_post_train_type}_{args.host_model_post_train_dataset_name}"

    logger.info(f"Using student model {args.student_model_name}")
    student_model_path = os.path.join(ckpt_dir,
                                      "models",
                                      args.student_model_name,
                                      "base")
    if args.sae_type == "finetuned":
        accelerator.print(f"Using finetuned SAE ({args.sae_hookpoint}) from {args.sae_name}/{ckpt_postfix}/{str(args.host_model_checkpoint)}")
        sae_path = os.path.join(ckpt_dir,
                                "saes",
                                args.sae_name,
                                ckpt_postfix,
                                str(args.host_model_checkpoint),
                                args.sae_hookpoint)

        output_dir = os.path.join(ckpt_dir,
                                  "models",
                                  args.base_model_name,
                                  ckpt_postfix,
                                  str(args.host_model_checkpoint),
                                  "distill",
                                  args.distill_dataset_name,
                                  args.sae_hookpoint,
                                  args.distill_type)

        run_name = f"{args.base_model_name}_{ckpt_postfix}_{args.distill_dataset_name}_{args.sae_hookpoint}_{args.student_model_name}_{formatted_datetime}"

    elif args.sae_type == "reason_pretrained":
        accelerator.print(f"Using reason_pretrained SAE ({args.sae_hookpoint}) from {args.sae_name}/{ckpt_postfix}/{str(args.host_model_checkpoint)}/pretrain")
        sae_path = os.path.join(ckpt_dir,
                                "saes",
                                args.sae_name,
                                ckpt_postfix,
                                str(args.host_model_checkpoint),
                                "pretrain",
                                args.sae_hookpoint)

        output_dir = os.path.join(ckpt_dir,
                                  "models",
                                  args.base_model_name,
                                  ckpt_postfix,
                                  str(args.host_model_checkpoint),
                                  "distill",
                                  args.distill_dataset_name,
                                  "pretrain",
                                  args.sae_hookpoint,
                                  args.distill_type)

        run_name = f"reason_pretrained_{args.base_model_name}_{ckpt_postfix}_{args.distill_dataset_name}_{args.sae_hookpoint}_{args.student_model_name}_{formatted_datetime}"

    elif args.sae_type == "pretrained":
        accelerator.print(f"Using pretrained SAE ({args.sae_hookpoint}) from {args.sae_name}")
        sae_path = os.path.join(ckpt_dir,
                                "saes",
                                args.sae_name,
                                "base",
                                args.sae_hookpoint)

        output_dir = os.path.join(ckpt_dir,
                                  "models",
                                  args.student_model_name,
                                  "base",
                                  "distill",
                                  args.distill_dataset_name,
                                  args.sae_hookpoint)

        run_name = f"{args.base_model_name}_base_{args.distill_dataset_name}_{args.sae_hookpoint}_{args.student_model_name}_{formatted_datetime}"
    else:
        raise ValueError(f"Invalid SAE type: {args.sae_type}. Expected 'finetuned', 'pretrained' or 'reason_pretrained'.")

    # Check existence on main process
    if not os.path.exists(student_model_path):
         raise FileNotFoundError(f"Base model path not found: {student_model_path}")
    if not os.path.exists(sae_path):
         raise FileNotFoundError(f"SAE path not found: {sae_path}")

    # Initialize trackers (like WandB) on the main process after paths are set
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=os.environ["WANDB_PROJECT"],
            config=vars(args),
            init_kwargs={"wandb": {"name": run_name}}
        )

    #############################
    # Load and preprocess dataset
    #############################

    accelerator.print(f"Loading and preprocessing dataset {args.distill_dataset_name} ...")
    if args.distill_dataset_name in RL_POST_TRAIN_DATASET_MAP.keys():
        assert args.host_model_post_train_type == "grpo"
        distill_dataset_name = RL_POST_TRAIN_DATASET_MAP[args.distill_dataset_name]
        dataset_split = "train"
        raw_dataset = load_dataset(distill_dataset_name, split=dataset_split)

        if "2thought" in args.distill_dataset_name:
            raw_dataset = raw_dataset.rename_column('messages', 'problem')
            raw_dataset = raw_dataset.rename_column('verification_info', 'answer')

            def extract_problem(example):
                problem = example['problem'][0]["content"]
                return {"problem": problem}

            def extract_answer(example):
                answer = json.loads(example['answer'])
                answer = answer["answer"]["value"]
                return {"answer": f"${answer}$"}

            # Apply the transformation to the entire dataset
            raw_dataset = raw_dataset.map(extract_problem)
            raw_dataset = raw_dataset.map(extract_answer)
        elif "thoughts3" in args.distill_dataset_name:
            raw_dataset = raw_dataset.filter(lambda example: example["domain"] == "science")

        processed_dataset = raw_dataset.map(
            make_conv_for_sae_grpo,
            fn_kwargs={
                "dataset_name_or_path": distill_dataset_name if "thoughts3" in args.distill_dataset_name else None},
            batched=True,
        )

    else:
        accelerator.print(f"Dataset {args.distill_dataset_name} not found in RL_POST_TRAIN_DATASET_MAP.", main_process_only=True)
        raise ValueError(f"Dataset {args.distill_dataset_name} not found.")

    accelerator.print(f"Load and tokenize dataset: {processed_dataset}")
    tokenizer = AutoTokenizer.from_pretrained(student_model_path)
    tokenized_dataset = chunk_and_tokenize(processed_dataset, tokenizer) # Add max_length if needed
    train_dataset = TokenizedTextDataset(tokenized_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size)
    accelerator.print(f"Created DataLoader with batch size {args.batch_size}")

    ####################
    # Load model and SAE
    ####################

    accelerator.print(f"Loading Model from {student_model_path} and SAE from {sae_path}")
    model = AutoModelForCausalLM.from_pretrained(
        student_model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        attn_implementation='flash_attention_2', # or 'eager' based on model
    )
    model.requires_grad_(False) # Freeze base model

    lora_config = LoraConfig(r=args.lora_r,
                             lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout,
                             target_modules=args.lora_target_modules,
                             bias="none",
                             task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    sae_module = Sae.load_from_disk(sae_path) # Load on CPU first
    sae_module = sae_module.to(dtype=torch.bfloat16)
    sae_module = sae_module.to(accelerator.device) # Move SAE to the process's device
    sae_module.eval()
    sae_module.requires_grad_(False)
    accelerator.print(f"SAE module placed on device: {sae_module.device}")

    ####################
    # Main training func
    ####################

    accelerator.print("Starting training...")
    train_model(args, accelerator, peft_model, sae_module, train_dataloader, tokenizer, run_name, output_dir)

    ##########
    # Clean up
    ##########

    accelerator.print("Training finished.")
    accelerator.end_training()
    accelerator.print("Script finished.")


if __name__ == "__main__":
    main()