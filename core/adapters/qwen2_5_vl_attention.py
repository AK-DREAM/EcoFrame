import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


class Qwen25VLTracker:
    def __init__(self):
        self.attn_info = {}
        self.phase = "detect"

    def reset(self):
        self.attn_info.clear()
        self.phase = "detect"


def apply_qwen2_5_vl_attention_patch():
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = Qwen2_5_VLFlashAttention2_forward


def _track_frame_weights(self, query_states, key_states):
    tracker = getattr(self, "tracker", None)
    if tracker is None or tracker.phase != "detect":
        return

    if tracker.attn_info["attn_layer"] is None or tracker.attn_info["layer_count"] in tracker.attn_info["attn_layer"]:
        video_start = tracker.attn_info["video_start"]
        video_length = tracker.attn_info["video_length"]
        frames_num = tracker.attn_info["frames_num"]
        llm_grid_t = tracker.attn_info["llm_grid_t"]
        tokens_per_temporal = tracker.attn_info["tokens_per_temporal"]

        last_query = query_states[:, :, -1:, :].float()
        video_key = key_states[:, :, video_start : video_start + video_length, :].float()
        attn_scores = torch.matmul(last_query, video_key.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.mean(1).squeeze(0).squeeze(0)
        temporal_weights = attn_weights.view(llm_grid_t, tokens_per_temporal).mean(1)
        frame_bins = torch.arange(frames_num, device=temporal_weights.device) * llm_grid_t // frames_num
        tracker.attn_info["frame_weights"].append(temporal_weights[frame_bins])

    tracker.attn_info["layer_count"] += 1


def _project_qkv(self, hidden_states):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    return bsz, q_len, query_states, key_states, value_states


def Qwen2_5_VLFlashAttention2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    bsz, q_len, query_states, key_states, value_states = _project_qkv(self, hidden_states)

    _track_frame_weights(self, query_states, repeat_kv(key_states, self.num_key_value_groups))

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        modeling_qwen2_5_vl.logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = modeling_qwen2_5_vl._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
