import os, re, json, dill
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import PeftModel
from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr, make_fast_cost_from_w
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost

from copy import deepcopy
import torch
import torch.nn as nn
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl import create_reference_model


# --- your helpers: build_user_msg, parse_w, load_instances, make_fast_cost_from_w, etc. ---

SYSTEM_MSG = """Respond in the following format:
<answer>{\"w\": [w1, w2, w3, w4, w5, w6, w7]}</answer>
Where w1-w7 are non-negative numbers.
The <answer> section MUST contain valid JSON only.
"""

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    semantics = (
        f"w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]"
        f"All {w_dim} values must be >= 0."
    )
    return (
        "INSTANCE SUMMARY:"
        f"{prompt_sys}"
        "SCORE PREFERENCES:"
        f"{prompt_score}"
        f"{semantics}"
        "Choose w to maximize the final score."
        "OUTPUT FORMAT (STRICT):"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
    )

def build_query_text(row, tokenizer):
    # Build the same user message you used in GRPO
    msg_user = build_user_msg(row["prompt_sys"], row["prompt_score"], w_dim=7)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": msg_user},
    ]
    # Use chat template so model sees the exact formatting it expects
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def preprocess(row, tokenizer):
    q = build_query_text(row, tokenizer)
    enc = tokenizer(
        q,
        truncation=True,
        max_length=1024,  # tune
        padding=False,
        return_tensors=None,
    )
    return {
        "query": q,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "trial_ids": row["trial_id"],
        "params": row["params"],
    }

def strict_extract_json(text: str):
    text = (text or "").strip()
    # if you really want ONLY JSON, enforce it:
    if not (text.startswith("{") and text.endswith("}")):
        raise ValueError("Not JSON-only")
    obj = json.loads(text)
    if set(obj.keys()) != {"w"}:
        raise ValueError("Wrong keys")
    return obj

def compute_reward_single(text, trial_id, params, instances):
    # reward in roughly [-1, +1] or [0, 1]
    # 1) format gate
    try:
        obj = strict_extract_json(text)
        w = np.asarray(obj["w"], dtype=float).reshape(-1)
        if w.shape != (7,) or not np.all(np.isfinite(w)) or np.any(w < 0):
            return -1.0
    except Exception:
        return -1.0

    # 2) bounds
    lo = 1e-3 * np.ones(7)
    hi = 1e3  * np.ones(7)
    if np.any(w < lo) or np.any(w > hi):
        return -0.5

    # 3) expensive MPC score
    inst = instances[trial_id]
    rho_goal = float(inst["meta"]["rho"])
    u_max = inst["meta"]["u_max"]
    x0 = inst["init_goal"]["x0"]
    xg = inst["init_goal"]["x_end"]

    def phi_target(x):
        return np.linalg.norm(x0 - xg) - rho_goal

    cfg_score = CompositeScoreConfig(
        dt=float(inst["meta"]["dt"]),
        D=float(inst["meta"]["rho"]),
        beta_margin=float(params["beta_margin"]),
        beta_time=float(params["beta_time"]),
        beta_u=float(params["beta_u"]),
        beta_du=float(params["beta_du"]),
        U_ref=u_max,
        dU_ref=0.25 * u_max,
    )
    scorer = CompositeTrajectoryScorer(phi_target, cfg_score)

    cfg_ilqr = ILQRMPCConfig(
        H=int(params["H"]),
        max_iters=int(params["max_iters"]),
        u_min=-u_max,
        u_max=u_max,
        shift_fill="zero",
    )

    def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
        dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
        return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)

    cost_obj = make_fast_cost_from_w(inst, w, QuadCostConfig, FastQuadraticCost)
    mpc = mpc_builder(inst, cost_obj, cfg_ilqr)

    X, U = run_closed_loop_fast_ilqr_mpc(
        mpc=mpc,
        step_fn_true=inst["step"],
        x0=x0,
        T_outer=int(params["T_outer"]),
    )
    Y = float(scorer.score(X, U)["score"])   # assume in [0,1]

    # PPO likes moderate scale
    return float(Y)  # or 2*Y-1 if you want [-1,1]

# LOAD instances directly for reward
def load_instances(path: str) -> dict:
    with open(path, "rb") as f:
        instances = dill.load(f)
    print(f"Loaded {len(instances)} instances")
    return instances

#FUNCTION TO PRINT the number of trainable paraemters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params=0
    all_model_params=0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model\n parameters:{trainable_model_params}\n all model parameters {all_model_params}\n percentage of trainable model: {trainable_model_params/all_model_params*100}"

# ----------------------------
# Dummy reward model (unused; we pass rewards manually)
# ----------------------------
class DummyRewardModel(nn.Module):
    def forward(self, *args, **kwargs):
        return torch.zeros(1)

def main():
    instances = load_instances("data_10000_dt05_N20_grpo/instances.dill")

    base_id = "meta-llama/Llama-3.2-3B-Instruct"
    adapter_path = "./sft-pre-ppo-10000"

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 1) Build PPO model WITH value head from the BASE STRING (this must be a string/path)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype="auto",
    )

    # 2) Attach LoRA to the underlying pretrained_model (no merge)
    ppo_model.pretrained_model = PeftModel.from_pretrained(
        ppo_model.pretrained_model,
        adapter_path,
        is_trainable=True,
    )

    # # 3) Freeze base weights, keep LoRA trainable
    # for name, p in ppo_model.pretrained_model.named_parameters():
    #     p.requires_grad_(("lora_" in name))
    #
    # # 4) Keep v_head trainable
    # for p in ppo_model.v_head.parameters():
    #     p.requires_grad_(True)
    #
    # # 5) Patch generation_config onto the wrapper (TRL 0.23.0)
    # gc = getattr(ppo_model.pretrained_model, "generation_config", None)
    # if gc is None:
    #     gc = GenerationConfig.from_model_config(ppo_model.pretrained_model.config)
    # ppo_model.generation_config = gc

    # 6) Create reference model for KL
    ref_model = create_reference_model(ppo_model)
    # ref_model.generation_config = ppo_model.generation_config

    # base_id = "meta-llama/Llama-3.2-3B-Instruct"
    # adapter_path = "./sft-pre-ppo-10000"
    # base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto", device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(base_id)
    # peft_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    # # peft_model.print_trainable_parameters()
    # print(print_number_of_trainable_model_parameters(peft_model))
    # # merged_model = peft_model.merge_and_unload()
    # ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model, torch_dtype=torch.bfloat16, is_trainable=True)
    # # ppo_model = AutoModelForCausalLMWithValueHead(merged_model, torch_dtype=torch.bfloat16, is_trainable=True)
    #
    # print(f"PPO model prameters to be updated(ValueHead+769 params):\n{print_number_of_trainable_model_parameters(ppo_model)}")
    # print(ppo_model.v_head)
    #
    # # Reference Model:a model that is not fine tuned with all, not even with PEFT.
    # ref_model = create_reference_model(ppo_model)
    # print(f"Refernce model parameters to be updated:\n{print_number_of_trainable_model_parameters(ref_model)}\n")

    ds = load_dataset("json", data_files="data_10000_dt05_N20_grpo/dataset_all.jsonl", split="train")
    ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds["train"]
    eval_ds = ds["test"]
    train_ds = train_ds.map(
        lambda row: preprocess(row, tokenizer=tokenizer),
        remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(
        lambda row: preprocess(row, tokenizer=tokenizer),
        remove_columns=eval_ds.column_names,
    )

    print(train_ds[0])


    reward_model = DummyRewardModel()
    #

    config = PPOConfig(
        learning_rate=1e-5,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        num_ppo_epochs=4,
        logging_steps=10,
        report_to="trackio",
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    trainer = PPOTrainer(
        args=config,
        model=ppo_model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=ppo_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    gen_kwargs = dict(
        max_new_tokens=64,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # generate responses
    generation_config = GenerationConfig(max_new_tokens=64,
                                         tok_k=0.0,
                                         top_p=1.0,
                                         do_sample=True)

    for step, batch in enumerate(trainer.dataloader):
        # batch["input_ids"] is already padded by collator in PPOTrainer
        queries = batch["input_ids"].to(trainer.accelerator.device)
        attn = batch["attention_mask"].to(trainer.accelerator.device)

        responses = trainer.generate(
            input_ids=queries,
            attention_mask=attn,
            **generation_config,

        )

        # decode only the newly generated tokens
        gen_only = responses[:, queries.shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        # compute rewards (your expensive MPC scorer)
        rewards = []
        for txt, trial_id, params in zip(texts, batch["trial_ids"], batch["params"]):
            r = compute_reward_single(txt.strip(), trial_id, params, instances)  # you implement this
            rewards.append(r)

        rewards = torch.tensor(rewards, device=queries.device, dtype=torch.float32)

        # PPO update step
        stats = trainer.step(queries, responses, rewards)

        if step % 10 == 0:
            print("step", step, "reward_mean", rewards.mean().item())





    #
    #
    # ppo_config = PPOConfig(
    #     learning_rate=2e-5,
    #     batch_size=8,
    #     mini_batch_size=2,
    #     gradient_accumulation_steps=4,
    #     num_ppo_epochs=4,
    #     kl_coef=0.1,          # tune
    #     cliprange=0.2,
    #     cliprange_value=0.2,
    #     vf_coef=1.0,
    #     logging_steps=10,
    #     # target_kl=0.1,        # tune
    #     # log_with="trackio",
    #     report_to="trackio",
    # )

    # trainer = PPOTrainer(
    #     args=ppo_config,
    #     model=policy,
    #     ref_model=ref_policy,
    #     reward_model=reward_model,
    #     value_model=value_model,
    #     train_dataset=train_ds,
    #     processing_class=tokenizer,
    # )

    #
    # # trainer = PPOTrainer(
    # #     args=ppo_config,
    # #     model=fine_tuned_model,
    # #     processing_class=tokenizer,
    # # )
    #
    # trainer = PPOTrainer(
    #     args=ppo_config,
    #     model=policy,
    #     ref_model=ref_policy,
    #     reward_model=reward_model,
    #     value_model=value_model,
    #     train_dataset=train_ds,
    #     processing_class=tokenizer,
    # )



    # max_ppo_steps = 10
    # for step, batch in tqdm(enumerate(trainer.dataloader)):
    #     # break when we teach max_steps
    #     if step >= max_ppo_steps:
    #         break
    #
    #     prompt_tensors = batch["input_ids"]


    # for step, batch in enumerate(trainer.dataloader):
    #     # batch["input_ids"] is already padded by collator in PPOTrainer
    #     queries = batch["input_ids"].to(trainer.accelerator.device)
    #     attn = batch["attention_mask"].to(trainer.accelerator.device)
    #
    #     # generate responses
    #     responses = trainer.generate(
    #         input_ids=queries,
    #         attention_mask=attn,
    #         **gen_kwargs,
    #     )
    #
    #     # decode only the newly generated tokens
    #     gen_only = responses[:, queries.shape[1]:]
    #     texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    #
    #     # compute rewards (your expensive MPC scorer)
    #     rewards = []
    #     for txt, trial_id, params in zip(texts, batch["trial_ids"], batch["params"]):
    #         r = compute_reward_single(txt.strip(), trial_id, params, instances)  # you implement this
    #         rewards.append(r)
    #
    #     rewards = torch.tensor(rewards, device=queries.device, dtype=torch.float32)
    #
    #     # PPO update step
    #     stats = trainer.step(queries, responses, rewards)
    #
    #     if step % 10 == 0:
    #         print("step", step, "reward_mean", rewards.mean().item())





if __name__ == "__main__":
    main()