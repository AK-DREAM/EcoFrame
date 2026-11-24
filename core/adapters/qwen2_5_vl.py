import torch

from core.adapters.qwen2_5_vl_attention import Qwen25VLTracker
from core.utils import _sample_frames, _tic, _toc


class Qwen25VLCToFAdapter:
    def __init__(
        self,
        model,
        processor,
        tokenizer,
        device,
        video_path,
        context,
        system_prompt,
        gen_kwargs,
        use_cache,
        num_options,
        max_pixels,
        min_pixels,
        padding_side="right",
        open_ended=False,
        entropy_top_k=20,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.video_path = video_path
        self.context = context
        self.system_prompt = system_prompt
        self.num_options = num_options
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.padding_side = padding_side
        self.open_ended = open_ended
        self.entropy_top_k = int(entropy_top_k)

        self.gen_kwargs = dict(gen_kwargs)
        if use_cache is not None:
            self.gen_kwargs["use_cache"] = use_cache

    def _build_inputs(self, frames):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": self.video_path,
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels,
                    },
                    {"type": "text", "text": self.context},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            videos=[frames],
            padding=True,
            padding_side=self.padding_side,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def _set_tracker(self, inputs, frames_num, attn_layers):
        video_positions = torch.where(inputs.input_ids[0] == self.model.config.video_token_id)[0]
        video_grid = inputs.video_grid_thw[0]
        spatial_merge_size = self.model.config.vision_config.spatial_merge_size
        llm_grid_t = int(video_grid[0].item())
        llm_grid_h = int(video_grid[1].item()) // spatial_merge_size
        llm_grid_w = int(video_grid[2].item()) // spatial_merge_size

        self.model.tracker = Qwen25VLTracker()
        self.model.tracker.reset()
        self.model.tracker.attn_info.update(
            {
                "video_start": int(video_positions[0].item()),
                "video_length": int(video_positions.numel()),
                "frames_num": int(frames_num),
                "llm_grid_t": llm_grid_t,
                "tokens_per_temporal": int(llm_grid_h * llm_grid_w),
                "attn_layer": attn_layers,
                "frame_weights": [],
                "layer_count": 0,
            }
        )
        for layer in self.model.model.layers:
            layer.self_attn.tracker = self.model.tracker

    def _compute_sequence_entropy(self, score_tensors):
        step_entropies = []
        for scores in score_tensors:
            log_probs = torch.log_softmax(scores.float(), dim=-1)
            topk_log_probs = log_probs.topk(min(self.entropy_top_k, log_probs.shape[-1]), dim=-1).values
            topk_probs = topk_log_probs.exp()
            step_entropies.append(-(topk_probs * topk_log_probs).sum(dim=-1))
        return float(torch.stack(step_entropies, dim=0).mean(0)[0].detach().cpu().item())

    def run_coarse(self, vr, frame_indices, attn_layers):
        frames = _sample_frames(vr, self.video_path, frame_indices)
        inputs = self._build_inputs(frames)
        self._set_tracker(inputs, len(frame_indices), attn_layers)

        vlm_start = _tic(self.device)
        coarse_outputs = self.model(**inputs, use_cache=False, return_dict=True)
        self.model.tracker.phase = "apply"

        frame_weights = torch.stack(self.model.tracker.attn_info["frame_weights"], dim=0).mean(0)
        frame_weights = frame_weights / frame_weights.sum().clamp_min(1e-12)
        frame_weights = frame_weights * frame_weights.numel()
        frame_weights = frame_weights.detach().float().cpu().tolist()

        if self.open_ended:
            vlm_start = _tic(self.device)
            coarse_gen_kwargs = dict(self.gen_kwargs)
            coarse_gen_kwargs["return_dict_in_generate"] = True
            coarse_gen_kwargs["output_scores"] = True
            coarse_generation = self.model.generate(
                **inputs,
                **coarse_gen_kwargs,
            )
            sequences = coarse_generation.sequences if hasattr(coarse_generation, "sequences") else coarse_generation
            generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
            coarse_answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            coarse_option_entropy = self._compute_sequence_entropy(coarse_generation.scores)
        else:
            options = [chr(ord("A") + i) for i in range(int(self.num_options))]
            option_ids = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
            option_logits = coarse_outputs.logits[:, -1, option_ids].float()
            option_probs = torch.softmax(option_logits, dim=-1)
            coarse_answer = options[int(option_probs[0].argmax().item())]
            coarse_option_entropy = float((-(option_probs * torch.log(option_probs.clamp_min(1e-12))).sum(dim=-1))[0].item())

        return {
            "frame_significance": [float(x) for x in frame_weights],
            "coarse_answer": coarse_answer,
            "coarse_option_entropy": coarse_option_entropy,
        }, {"inference_ms": _toc(vlm_start, self.device)}

    def run_final(self, vr, frame_indices):
        frames = _sample_frames(vr, self.video_path, frame_indices)
        inputs = self._build_inputs(frames)
        final_start = _tic(self.device)
        outputs = self.model.generate(
            **inputs,
            **self.gen_kwargs,
        )
        return {"sequences": outputs, "input_ids": inputs.input_ids}, _toc(final_start, self.device)
