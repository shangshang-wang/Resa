from resa.utils.prompt import SAE_FINE_TUNE_SYSTEM_PROMPT


def make_conv_for_sae_grpo(example, dataset_name_or_path=None):
    if isinstance(example, str):
        example = [example]

    if not dataset_name_or_path:
        if "problem" not in example:
            if "question" in example:
                example["problem"] = example["question"]
            elif "prompt" in example:
                example["problem"] = example["prompt"]
            else:
                raise ValueError("No problem/question/prompt found in the example.")

        # Only use the final answer but in the thinking format
        combined_text = [
            f"{SAE_FINE_TUNE_SYSTEM_PROMPT}User:\n {problem}\n\nAssistant:\n <think> {answer} </think>\n<answer> Answer: {answer}</answer>"
            for problem, answer in zip(example["problem"], example["answer"])
        ]
    else:
        example["problem"] = [conversation[0]["value"] for conversation in example["conversations"]]
        combined_text = [
            f"{SAE_FINE_TUNE_SYSTEM_PROMPT}User:\n {problem}\n\nAssistant:\n <think> Think </think>\n<answer> Answer </answer>"
            for problem in zip(example["problem"])
        ]

    return {
        "text": combined_text
    }

def make_conv_for_sae_sft(example, dataset_name_or_path, model_name_or_path, tokenizer):
    if isinstance(example, str):
        example = [example]

    if dataset_name_or_path == "simplescaling/s1K-claude-3-7-sonnet":
        trajectory_name = "claude_thinking_trajectory"
        attemp_name = "claude_attempt"
    elif dataset_name_or_path == "simplescaling/s1K-1.1":
        trajectory_name = "deepseek_thinking_trajectory"
        attemp_name = "deepseek_attempt"
    elif dataset_name_or_path == "simplescaling/s1K":
        trajectory_name = "thinking_trajectories"
        attemp_name = "attempt"
    else:
        raise ValueError(f"Unknown dataset for sft post-training: {dataset_name_or_path}")

    if "Qwen" in model_name_or_path:
        conv = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<|im_start|>think\n{trajectory}<|im_start|>answer\nAnswer: {attempt}"},
            ]
            for question, trajectory, attempt in zip(example["question"], example[trajectory_name], example[attemp_name])
        ]
    elif "Llama" in model_name_or_path:
        conv = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<|start_header_id|>think\n{trajectory}<|start_header_id|>answer\nAnswer: {attempt}"},
            ]
            for question, trajectory, attempt in zip(example["question"], example[trajectory_name], example[attemp_name])
        ]
    else:
        raise ValueError(f"Unknown model_name_or_path: {model_name_or_path}")
    return {
        "text": tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False)
    }
