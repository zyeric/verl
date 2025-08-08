# Run Command

```
bash examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_nnscaler.sh
```

# Model Arch

## Huggingface Models

Currently, only Qwen2.5 dense models are tested, you can change `HF_MODEL_PATH` in the script to other qwen models. If 
you want to add other huggingface models, please patch the model in `hf_patch` at `./verl/utils/nnscaler_utils.py`.

## Customized Arch

You need to change the tracing logic in `_build_model_optimizer` at `./verl/workers/nnscaler_workers.py`.

# Distributed Plan

For simplicity, we use partition constraint to enforce actor and reference model to employ sequence parallelism. We
will implement parameter offload for them in the future to save the memory.

# nnScaler options

- plan_ngpus: the number of devices composing a data parallel unit
- runtime_ngpus: the number of devices in total
- static_seq_len: like the fsdp trainer, we pack the generated responses to improve the device utilization. This is achieved
by registered flash attention interface in nnscaler, you may check the `flash_attention_forward_ring` for more details. In 
addition, if you want to trace your own arch, we recommend to use `flash_attention_forward_ring` in your attention class.

# Integrate other inference engine

You can follow the implementation at `./verl/workers/sharding_manager/nnscaler_vllm.py`. Current implementation only 
supports data parallelism and sequence parallelism. We will support the sharding process in general in the future.
