from datasets import load_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # must be set before torch initializes CUDA



dataset_name = 'AI-MO/NuminaMath-TIR'
train_dataset = load_dataset(dataset_name, split='train[:5%]')
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant  "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process is enclosed strictly within  and  tags. "
    "After closing , the assistant MUST provide the final answer in plain text."
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)

model_id, output_dir = "meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct-GRPO"             # ✅ ~9.5GB VRAM

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

torch.cuda.set_device(0)

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    torch_dtype=torch.float16,         # use torch_dtype (not dtype="float32")
    quantization_config=bnb,
    device_map={"": 0},                # <-- critical: pin model to GPU0
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
#     dtype="float32",                              # Change to bfloat16 if GPU has support
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
#         bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
#         bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
#         bnb_4bit_quant_type="nf4",                # Type of quantization. "nf4" is recommended for recent LLMs
#     ),
# )
# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different LLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
)
from trl.rewards import think_format_reward

                         # reasoning_accuracy_reward)

from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    # Training schedule / optimization
    learning_rate=2e-5,  # Learning rate for the optimizer
    # num_train_epochs=1,
    max_steps=500,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead

    # Parameters that control GRPO training (you can adapt them)
    per_device_train_batch_size=8,
    max_completion_length=256,  # default: 256               # Max completion length produced during training
    num_generations=8,
    # default: 8                         # Number of generations produced during trainig for comparison

    # Optimizations
    optim="paged_adamw_8bit",  # Optimizer
    use_liger_kernel=True,  # Enable Liger kernel optimizations for faster training

    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=10,  # Log training metrics every N steps
    report_to="trackio",  # Experiment tracking tool
    # trackio_space_id=output_dir,  # HF Space where the experiment tracking will be saved
    log_completions=False,  # Return model completions during training

    # Hub integration
    push_to_hub=False,  # Automatically push the trained model to the Hugging Face Hub
    # The model will be saved under your Hub account in the repository named `output_dir`
    # vLLM params
    # use_vllm=False,                                        # Activate vLLM training for faster training
    # vllm_mode='colocate',
    # vllm_gpu_memory_utilization=0.1,
    # vllm_enable_sleep_mode=True
)


from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    # reward_funcs=[think_format_reward, reasoning_accuracy_reward],
    reward_funcs=[think_format_reward],
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

#
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
#
# print("RANK:", os.environ.get("RANK"),
#       "LOCAL_RANK:", os.environ.get("LOCAL_RANK"),
#       "cuda.current_device:", torch.cuda.current_device())
#
# # policy model device
# print("policy first param device:", next(trainer.model.parameters()).device)
# print("trainer device:", trainer.accelerator.device)
# print("policy device:", next(trainer.model.parameters()).device)