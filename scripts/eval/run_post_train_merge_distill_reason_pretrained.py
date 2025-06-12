import argparse
import os
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_name", type=str, default="DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--teacher_model_adapter_name", type=str, default="grpo_curated_still")
    parser.add_argument("--teacher_model_ckpt", type=str, default="checkpoint-2000")
    parser.add_argument("--student_model_name", type=str, default="Qwen2.5-Math-1.5B")
    parser.add_argument("--student_model_ckpt", type=str, default="checkpoint-500")
    parser.add_argument("--distill_dataset_name", type=str, default="curated_still")
    parser.add_argument("--distill_sae_hookpoint", type=str, default="model.layers.12")
    parser.add_argument("--distill_type", type=str, default="sft_r1_distill") # sft_r1_distill, sft_qwen_math, sft_qwen, base

    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    ckpt_dir = os.environ["CKPT_DIR"]

    if args.distill_type != "base":
        prefix = f"{ckpt_dir}/models/{args.teacher_model_name}/{args.teacher_model_adapter_name}/{args.teacher_model_ckpt}/distill/{args.distill_dataset_name}/pretrain/{args.distill_sae_hookpoint}/{args.distill_type}"
    else:
        prefix = f"{ckpt_dir}/models/{args.student_model_name}/base/distill/{args.distill_dataset_name}/pretrain/{args.distill_sae_hookpoint}"

    student_model_name = f"{ckpt_dir}/models/{args.student_model_name}/base"
    print("Student base model is loaded from: ", student_model_name)
    student_base_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto") # Automatically distributes across available GPUs

    student_model_adapter_name = f"{prefix}/{args.student_model_ckpt}"
    print("Student model adapter is loaded from: ", student_model_adapter_name)
    model = PeftModel.from_pretrained(student_base_model, student_model_adapter_name)
    model = model.merge_and_unload()

    student_merged_model_name = f"{prefix}/{args.student_model_ckpt}-merged"
    print("Student merged model will be saved to: ", student_merged_model_name)
    model.save_pretrained(student_merged_model_name)
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer.save_pretrained(student_merged_model_name)
