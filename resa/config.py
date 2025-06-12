from dataclasses import dataclass, field
from typing import Literal


# check ./recipes/MODEL_NAME/PT_METHOD/model_PT_DATASET.yaml
@dataclass
class ModelPTConfig:
    # //*******Model post-training configs*******//
    model_post_train_type: Literal["grpo", "sft"] = field(default="sft")
    model_post_train_dataset_name: str = field(default="r1_rationale_s1")
    model_post_train_dataset_config: str | None = field(default=None)
    trace_free: bool = field(default=True)

    rl_post_train_reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    rl_post_train_reward_weights: list[str] = field(default_factory=lambda: [2.0, 1.0])
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)


# check ./recipes/MODEL_NAME/PT_METHOD/sae_PT_DATASET.yaml
@dataclass
class SAEConfig:
    # //*******SAE configs*******//
    seed: int = field(default=42)

    sae_name: str = field(default="sae-Llama-3.2-1B-131k")
    sae_expansion_factor: int = field(default=32)
    sae_num_latents: int = field(default=131072)
    sae_hookpoints: list[str] = field(default_factory=lambda: ["model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"])
    sae_hookpoint_thresholds: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1])
    sae_inspect_dataset_name: str = field(default="qwen_rationale_limo")
    sae_inspect_dataset_target_column: str = field(default="question")
    sae_observe_type: Literal["problem", "completion"] = field(default="completion")
    sae_explainer_name: str = field(default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

    host_model_name: str = field(default="Llama-3.2-1B-Instruct")
    host_model_post_train_dataset_name: str = field(default="r1_rationale_s1")
    host_model_post_train_type: Literal["grpo", "sft", "base"]  = field(default="sft")
    host_model_checkpoints: list[str] = field(default_factory=lambda: ["checkpoint-500", "checkpoint-1000", "checkpoint-1500", "checkpoint-2000", "checkpoint-2500"])


# check ./recipes/MODEL_NAME/PT_METHOD/distill_PT_DATASET.yaml
@dataclass
class DistillConfig:
    # //*******Distill configs*******//
    seed: int = field(default=42)

    sae_name: str = field(default="sae-DeepSeek-R1-Distill-Qwen-1.5B-65k")
    sae_hookpoint: str = field(default="model.layers.12")
    sae_type: str = field(default="finetuned") # finetuned, pretrained, reason_pretrained

    base_model_name: str = field(default="DeepSeek-R1-Distill-Qwen-1.5B")
    distill_type: str = field(default="sft_r1_distill") # sft_r1_distill, sft_qwen_math, sft_qwen
    distill_dataset_name: str = field(default="curated_still")
    student_model_name: str = field(default="DeepSeek-R1-Distill-Qwen-1.5B") # Qwen2.5-Math-1.5B, Qwen2.5-1.5B
    host_model_post_train_dataset_name: str = field(default="curated_still")
    host_model_post_train_type: Literal["grpo"]  = field(default="grpo")
    host_model_checkpoint: str = field(default="checkpoint-500")

    lora_r: int = field(default=32)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"])
    logging_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    num_epochs: int = field(default=5)
    batch_size: int = field(default=1)
    save_steps: int = field(default=50)
