import torch

from llava.constants import IMAGE_TOKEN_INDEX

from core.utils import _sample_frames, _tic, _toc


class LlavaOneVisionCToFAdapter:
    tracker_cls = None

    def __init__(
        self,
        model,
        image_processor,
        tokenizer,
        device,
        inputs,
        attention_mask,
        pad_token_id,
        gen_kwargs,
        use_cache,
        video_path,
        num_options,
        position_ids=None,
        image_sizes=None,
        modalities=None,
        open_ended=False,
        entropy_top_k=20,
    ):
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.device = device
        self.inputs = inputs
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.image_sizes = image_sizes
        self.modalities = modalities if modalities is not None else gen_kwargs.get("modalities", ["image"])
        self.video_path = video_path
        self.num_options = num_options
        self.open_ended = open_ended
        self.entropy_top_k = int(entropy_top_k)

        self.gen_kwargs = dict(gen_kwargs)
        self.gen_kwargs.pop("modalities", None)
        if pad_token_id is not None:
            self.gen_kwargs["pad_token_id"] = pad_token_id
        if use_cache is not None:
            self.gen_kwargs["use_cache"] = use_cache

        self.visual_start = torch.where(inputs[0] == IMAGE_TOKEN_INDEX)[0].item()
        self.visual_end_offset = -(inputs.shape[1] - self.visual_start)

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
        coarse_images = [self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)]
        tracker_cls = self.tracker_cls
        if tracker_cls is None:
            from core.adapters.llava_onevision_attention import LlavaOneVisionTracker

            tracker_cls = LlavaOneVisionTracker
        self.model.tracker = tracker_cls()
        self.model.tracker.reset()
        for layer in self.model.model.layers:
            layer.self_attn.tracker = self.model.tracker

        vlm_start = _tic(self.device)
        (_, coarse_position_ids, coarse_attention_mask, _, coarse_inputs_embeds, _) = self.model.prepare_inputs_labels_for_multimodal(
            self.inputs,
            self.position_ids,
            self.attention_mask,
            None,
            None,
            coarse_images,
            self.modalities,
            image_sizes=self.image_sizes,
        )
        visual_length = (coarse_inputs_embeds.shape[1] + self.visual_end_offset) - self.visual_start + 1
        self.model.tracker.attn_info.update(
            {
                "visual_start": self.visual_start,
                "visual_length": visual_length,
                "frames_num": coarse_images[0].shape[0],
                "attn_layer": attn_layers,
                "frame_weights": [],
                "layer_count": 0,
            }
        )

        coarse_outputs = self.model.model(
            position_ids=coarse_position_ids,
            attention_mask=coarse_attention_mask,
            inputs_embeds=coarse_inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
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
            coarse_generation = super(type(self.model), self.model).generate(
                position_ids=coarse_position_ids,
                attention_mask=coarse_attention_mask,
                inputs_embeds=coarse_inputs_embeds,
                **coarse_gen_kwargs,
            )
            sequences = coarse_generation.sequences if hasattr(coarse_generation, "sequences") else coarse_generation
            coarse_answer = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            coarse_option_entropy = self._compute_sequence_entropy(coarse_generation.scores)
        else:
            options = [chr(ord("A") + i) for i in range(int(self.num_options))]
            option_ids = [self.tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
            coarse_logits = self.model.lm_head(coarse_outputs.last_hidden_state[:, -1, :])
            option_logits = coarse_logits[:, option_ids].float()
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
        final_images = [self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self.device)]
        final_start = _tic(self.device)
        (_, position_ids, attention_mask, _, inputs_embeds, _) = self.model.prepare_inputs_labels_for_multimodal(
            self.inputs,
            self.position_ids,
            self.attention_mask,
            None,
            None,
            final_images,
            self.modalities,
            image_sizes=self.image_sizes,
        )
        outputs = super(type(self.model), self.model).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **self.gen_kwargs,
        )
        return outputs, _toc(final_start, self.device)
