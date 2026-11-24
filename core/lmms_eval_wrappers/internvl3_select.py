import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from core.utils import _sample_frames
from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import DEFAULT_GEN_KWARGS, InternVL3, build_transform, dynamic_preprocess


SELECTED_FRAMES_PATH = ""


@register_model("internvl3_select")
class InternVL3Select(InternVL3):
    def __init__(
        self,
        *args,
        selected_frames_path: str = SELECTED_FRAMES_PATH,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert self.modality == "video", "internvl3_select only supports modality=video"
        self.selected_frames_path = selected_frames_path
        self.selected_frames_by_doc_id = self._load_selected_frames()

    def _load_selected_frames(self):
        if not self.selected_frames_path:
            return None
        selected_frames = json.loads(Path(self.selected_frames_path).read_text())
        return [[int(frame_idx) for frame_idx in frames] for frames in selected_frames]

    def load_video_select(self, video_path, doc_id):
        video_path = video_path if isinstance(video_path, str) else video_path[0]
        vr = VideoReader(video_path, ctx=cpu(0))
        if self.selected_frames_by_doc_id is None:
            frame_idx = np.linspace(0, len(vr) - 1, self.num_frame, dtype=int)
        else:
            frame_idx = self.selected_frames_by_doc_id[int(doc_id)]
        return _sample_frames(vr, video_path, frame_idx)

    def _preprocess_frames(self, frames):
        input_size = getattr(self._config, "force_image_size", None) or getattr(self._config.vision_config, "image_size", 448)
        transform = build_transform(input_size=input_size)
        pixel_values_list = []
        num_patches_list = []

        for frame in frames:
            image = Image.fromarray(frame).convert("RGB")
            tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=1)
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list).to(torch.bfloat16).to(self._device)
        return pixel_values, num_patches_list

    def _build_inputs(self, frames, context):
        pixel_values, num_patches_list = self._preprocess_frames(frames)
        question = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]) + context

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
        input_ids = model_inputs["input_ids"].to(self._device)
        attention_mask = model_inputs["attention_mask"].to(self._device)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(stop_str)
        return pixel_values, input_ids, attention_mask, stop_str, eos_token_id

    def _build_gen_kwargs(self, gen_kwargs, eos_token_id):
        current_gen_kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
        current_gen_kwargs.pop("until", None)
        current_gen_kwargs = {k: v for k, v in current_gen_kwargs.items() if k in DEFAULT_GEN_KWARGS}
        current_gen_kwargs["eos_token_id"] = eos_token_id
        return current_gen_kwargs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            for context, gen_kwargs, doc_to_visual, doc_id, task, split in zip(
                contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits
            ):
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)[0]
                frames = self.load_video_select(video_path, doc_id)

                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]

                pixel_values, input_ids, attention_mask, stop_str, eos_token_id = self._build_inputs(frames, context)
                current_gen_kwargs = self._build_gen_kwargs(gen_kwargs, eos_token_id)

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_start = time.perf_counter()
                    outputs = self.model.generate(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **current_gen_kwargs,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_latency_ms = (time.perf_counter() - generate_start) * 1000

                ans = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                ans = ans.split(stop_str)[0].strip()
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]

                res.append(GenerationResult(text=ans, metadata={"generate_latency_ms": generate_latency_ms}))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
