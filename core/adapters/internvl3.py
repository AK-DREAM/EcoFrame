from PIL import Image
import torch

from core.adapters.internvl3_attention import InternVL3Tracker
from core.utils import _sample_frames, _tic, _toc
from lmms_eval.models.simple.internvl3 import build_transform, dynamic_preprocess


class InternVL3CToFAdapter:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        video_path,
        context,
        gen_kwargs,
        use_cache,
        num_options,
        max_num,
        total_max_num,
        input_size=448,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.video_path = video_path
        self.context = context
        self.num_options = num_options
        self.input_size = int(input_size)

        self.gen_kwargs = dict(gen_kwargs)
        if use_cache is not None:
            self.gen_kwargs["use_cache"] = use_cache

    def _preprocess_frames(self, frames):
        transform = build_transform(input_size=self.input_size)
        pixel_values_list = []
        num_patches_list = []
        for frame in frames:
            image = Image.fromarray(frame).convert("RGB")
            tiles = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=False, max_num=1)
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).to(self.device)
        return pixel_values, num_patches_list

    def _build_inputs(self, frames):
        pixel_values, num_patches_list = self._preprocess_frames(frames)
        question = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]) + self.context

        get_conv_template = type(self.model).chat.__globals__["get_conv_template"]
        template = get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        stop_str = template.sep.strip()
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id
        for num_patches in num_patches_list:
            image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches + "</img>"
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=self.device)

        img_positions = torch.where(input_ids[0] == img_context_token_id)[0]
        frame_token_positions = []
        offset = 0
        for num_patches in num_patches_list:
            count = self.model.num_image_token * num_patches
            frame_token_positions.append(img_positions[offset : offset + count])
            offset += count

        eos_token_id = self.tokenizer.convert_tokens_to_ids(stop_str)
        return pixel_values, input_ids, attention_mask, image_flags, frame_token_positions, stop_str, eos_token_id

    def _set_tracker(self, frame_token_positions, attn_layers):
        self.model.tracker = InternVL3Tracker()
        self.model.tracker.reset()
        self.model.tracker.attn_info.update(
            {
                "frame_token_positions": frame_token_positions,
                "attn_layer": attn_layers,
                "frame_weights": [],
                "layer_count": 0,
            }
        )
        for layer in self.model.language_model.model.layers:
            layer.self_attn.tracker = self.model.tracker

    def run_coarse(self, vr, frame_indices, attn_layers):
        frames = _sample_frames(vr, self.video_path, frame_indices)
        pixel_values, input_ids, attention_mask, image_flags, frame_token_positions, _, _ = self._build_inputs(frames)
        self._set_tracker(frame_token_positions, attn_layers)

        vlm_start = _tic(self.device)
        coarse_outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            use_cache=False,
            return_dict=True,
        )
        self.model.tracker.phase = "apply"

        frame_weights = torch.stack(self.model.tracker.attn_info["frame_weights"], dim=0).mean(0)
        frame_weights = frame_weights / frame_weights.sum().clamp_min(1e-12)
        frame_weights = frame_weights * frame_weights.numel()
        frame_weights = frame_weights.detach().float().cpu().tolist()

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
        pixel_values, input_ids, attention_mask, _, _, stop_str, eos_token_id = self._build_inputs(frames)
        final_gen_kwargs = dict(self.gen_kwargs)
        final_gen_kwargs["eos_token_id"] = eos_token_id

        final_start = _tic(self.device)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **final_gen_kwargs,
        )
        return {"sequences": outputs, "stop_str": stop_str}, _toc(final_start, self.device)
