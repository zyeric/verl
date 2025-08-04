from typing import Optional, Tuple

import torch

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
    has_apex = True
except ImportError:
    has_apex = False


def qwen2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
    past_key_value = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    from nnscaler.graph.parser.external.tf_ring import flash_attention_forward_ring

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attn_output = flash_attention_forward_ring(
        # self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def rmsnorm_fwd(self, hidden_states):
    if has_apex:
        return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def hf_patch():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm
    Qwen2Attention.forward = qwen2_attn_forward
    # TODO(yizhu1): disable for now seems apex in this docker has some issues
    # Qwen2RMSNorm.forward = rmsnorm_fwd
