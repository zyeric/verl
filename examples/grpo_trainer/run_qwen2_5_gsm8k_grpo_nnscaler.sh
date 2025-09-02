set -x

# HF_MODEL_PATH=../Qwen2.5-3B-Instruct
HF_MODEL_PATH=../Qwen2.5-1.5B-Instruct

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
# useful for debugging
# export VERL_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_nnscaler_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=../gsm8k/train.parquet \
    data.val_files=../gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.nnscaler.static_seq_len=8192 \
    actor_rollout_ref.actor.nnscaler.plan_ngpus=4 \
    actor_rollout_ref.actor.nnscaler.runtime_ngpus=4 \
    actor_rollout_ref.actor.nnscaler.param_offload=True \
    actor_rollout_ref.actor.nnscaler.recompute_modules=Qwen2DecoderLayer \
    actor_rollout_ref.actor.nnscaler.pc_path=./examples/nnscaler/seq_parallel.yaml \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_nnscaler' \
    trainer.experiment_name='qwen2_5-1.5b_nnscaler_0902' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@