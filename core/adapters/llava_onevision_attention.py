import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class LlavaOneVisionTracker:
    def __init__(self):
        self.attn_info = {}
        self.phase = "detect"

    def reset(self):
        self.attn_info.clear()
        self.phase = "detect"


def apply_llava_onevision_attention_patch():
    from transformers.models.qwen2 import modeling_qwen2

    modeling_qwen2.Qwen2Attention.forward = Qwen2Attention_forward


def _track_frame_weights(self, query_states, key_states):
    tracker = getattr(self, "tracker", None)
    if tracker is None or tracker.phase != "detect":
        return

    if tracker.attn_info["attn_layer"] is None or tracker.attn_info["layer_count"] in tracker.attn_info["attn_layer"]:
        visual_start = tracker.attn_info["visual_start"]
        visual_length = tracker.attn_info["visual_length"]
        frames_num = tracker.attn_info["frames_num"]

        last_query = query_states[:, :, -1:, :].float()
        visual_key = key_states[:, :, visual_start : visual_start + visual_length - 1, :].float()
        attn_scores = torch.matmul(last_query, visual_key.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights_for_frames = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights_for_frames = attn_weights_for_frames.mean(1).squeeze(0).squeeze(0)
        tracker.attn_info["frame_weights"].append(attn_weights_for_frames.view(frames_num, -1).mean(1))
    tracker.attn_info["layer_count"] += 1


def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    from transformers.models.qwen2 import modeling_qwen2

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    _track_frame_weights(self, query_states, repeat_kv(key_states, self.num_key_value_groups))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface = modeling_qwen2.eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation != "sdpa" or not kwargs.get("output_attentions", False):
            attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
