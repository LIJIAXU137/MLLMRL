# AI Coding Agent Instructions for VPPO-RL

These instructions make AI agents immediately productive in this repo by summarizing the architecture, workflows, conventions, and integration points.

## Big Picture
- **Goal:** Train LVLMs with VPPO (Visually-Perceptive Policy Optimization) focusing updates on visually dependent tokens.
- **Runtime:** Single-driver Ray trainer orchestrates FSDP workers (Actor, Critic, Ref) and a vLLM rollout engine.
- **Core flow:** Data → vLLM rollouts → rewards → advantage/returns → PPO updates (actor/critic) → checkpoints/logging.

## Key Entry Points
- **CLI:** `python -m verl.trainer.main config=examples/configs/config.yaml ...` merges defaults + CLI overrides via OmegaConf.
- **Examples:** Training scripts at [examples/configs/train_vppo_7b.sh](examples/configs/train_vppo_7b.sh) and [examples/configs/train_vppo_8b.sh](examples/configs/train_vppo_8b.sh).
- **Config model:** Dataclasses in [verl/trainer/config.py](verl/trainer/config.py); keys mirror CLI namespaces: `data.*`, `worker.*`, `algorithm.*`, `trainer.*`.

## Architecture & Roles
- **Trainer:** [verl/trainer/main.py](verl/trainer/main.py) bootstraps Ray, logs env vars, and runs `Runner.run()`.
- **Ray Trainer:** [verl/trainer/ray_trainer.py](verl/trainer/ray_trainer.py) manages roles and resources; enforces batch divisibility and algorithm constraints.
- **Workers (FSDP):** [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py) initializes Actor/Critic/Ref with HuggingFace models, FSDP sharding, optional CPU/offload, Ulysses seq-parallel.
- **Rollout:** vLLM SPMD in [verl/workers/rollout/vllm_rollout_spmd.py](verl/workers/rollout/vllm_rollout_spmd.py); uses `tensor_parallel_size`, sleep-mode to offload, and `SamplingParams` for generation.
- **Algorithms:** PPO utilities in [verl/trainer/core_algos.py](verl/trainer/core_algos.py) incl. GRPO/GAE/RLOO, KL controllers, policy/value loss.
- **Data:** HF datasets via [verl/utils/dataset.py](verl/utils/dataset.py); multimodal preprocessing and prompt formatting with Jinja templates.

## Critical Workflows
- **Environment Setup (Agent Mode):**
  Before executing any code or training scripts, ensure the environment is activated:
  ```bash
  conda activate vppo
  ```
- **Install (editable):**
  ```bash
  pip install -e .
  ```
- **Train (7B example, override important knobs):**
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m verl.trainer.main \
    config=examples/configs/config.yaml \
    worker.actor.model.model_path=/path/to/Qwen2.5-VL-7B-Instruct \
    data.train_files=chamber111/VPPO_ViRL39K_train \
    data.val_files=chamber111/VPPO_MMK12_validation \
    worker.rollout.n=8 trainer.total_epochs=2 \
    algorithm.use_vppo_on_perception=true \
    algorithm.top_p_perception_tokens=0.4 \
    algorithm.use_advantage_shaping=true \
    algorithm.use_entropy_penalty=true \
    algorithm.entropy_penalty_coef=0.06
  ```
- **Validation-only:** set `trainer.val_only=true` or use `trainer.val_before_train=true` with `trainer.val_freq`.
- **Checkpoints:** Auto-saved under `checkpoints/{project}/{experiment}`; resume via `trainer.load_checkpoint_path` or `trainer.find_last_checkpoint=true`.
- **Quality:** `make style` (ruff fix + format), `make quality` (lint only), `make build` (sdist/wheel).

## Project-Specific Conventions
- **OmegaConf CLI merge:** Any `config=...` file loads first; CLI keys override deeply (e.g., `data.max_prompt_length=4096`).
- **Batch divisibility:**
  - `data.rollout_batch_size % worker.actor.global_batch_size == 0` (and critic if used).
  - `(data.rollout_batch_size * worker.rollout.n) % worker.actor.micro_batch_size_per_device_for_experience == 0`.
- **Algorithm constraints:** GRPO/RLOO require `worker.rollout.n > 1`.
- **Token focus:** VPPO toggles live under `algorithm.*`; are propagated to `worker.actor.*` in `PPOConfig.post_init()`.
- **Multimodal formatting:** Use Jinja in [examples/format_prompt](examples/format_prompt) and batch reward fns in [examples/reward_function](examples/reward_function); e.g., `examples/reward_function/math.py:compute_score_wo_format`.

## Integration Points
- **Transformers & vLLM:** HuggingFace models with `attn_implementation="flash_attention_2"`; vLLM external executor with TP; ref policy optional via `algorithm.disable_kl`.
- **Qwen VL specifics:** Position IDs for Qwen2/Qwen3 VL computed in [verl/utils/dataset.py](verl/utils/dataset.py) using model-specific rope index.
- **Tokenizer/Processor:** `override_chat_template` supported in [verl/utils/tokenizer.py](verl/utils/tokenizer.py); pad token auto-fallback to EOS.
- **Ray runtime env:** Main sets `TOKENIZERS_PARALLELISM`, `NCCL_DEBUG`, `VLLM_LOGGING_LEVEL`, etc. in [verl/trainer/main.py](verl/trainer/main.py).

## Practical Examples
- **Switch to GRPO Pass@k:** `algorithm.adv_estimator=grpo_passk` and keep `worker.rollout.n>=2`.
- **Increase max lengths:** `data.max_prompt_length=4096 data.max_response_length=2048` (ensure vLLM `max_num_batched_tokens` ≥ sum).
- **Disable KL loss but keep penalty:** `algorithm.disable_kl=true algorithm.use_kl_loss=false algorithm.kl_penalty=low_var_kl`.

## Common Pitfalls
- Mismatch between `tensor_parallel_size` and world size in rollout.
- Overlong prompts without proper filtering; use `data.filter_overlong_prompts=true`.
- Using GRPO/RLOO with `rollout.n=1` will raise.

If any section is unclear or missing details you need, tell me which scenario you’re targeting (model, dataset, hardware), and I’ll refine these instructions. 