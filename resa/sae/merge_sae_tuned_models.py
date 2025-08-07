import argparse
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--source_model_adapter_type", type=str, default="grpo_still")
    parser.add_argument("--source_model_checkpoint", type=str, default="checkpoint-2000")

    parser.add_argument("--sae_type", type=str, default="finetuned")
    parser.add_argument("--trigger_dataset_name", type=str, default="still")
    parser.add_argument("--sae_hookpoint", type=str, default="model.layers.12")

    parser.add_argument("--target_model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--elicitation_dataset_name", type=str, default="still")
    parser.add_argument("--target_model_ckpt", type=str, default="checkpoint-500")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    ckpt_dir = os.environ["CKPT_DIR"]
    target_model_prefix = os.path.join(
        ckpt_dir,
        "models",
        args.target_model_name, # parent dir path
        f"sae_tuning_{args.elicitation_dataset_name}",  # target model setup
        f"{args.base_model_name}_{args.source_model_adapter_type}_{args.source_model_checkpoint}",  # source model setup
        f"{args.sae_type}_{args.trigger_dataset_name}_{args.sae_hookpoint}", # SAE training setup
    )
    target_model_path = f"{ckpt_dir}/models/{args.target_model_name}/base"

    print("Base model is loaded from: ", target_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto")

    target_model_adapter_path = f"{target_model_prefix}/{args.target_model_ckpt}"
    print("Target model adapter is loaded from: ", target_model_adapter_path)
    model = PeftModel.from_pretrained(base_model, target_model_adapter_path)

    merged_target_model_path = f"{target_model_prefix}/{args.target_model_ckpt}-merged"
    print("Merged target model will be saved to: ", merged_target_model_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_target_model_path)
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    tokenizer.save_pretrained(merged_target_model_path)