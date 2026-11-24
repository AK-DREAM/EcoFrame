import copy
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from core.utils import _sample_frames
from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_vid import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    KeywordsStoppingCriteria,
    LlavaVid,
    SeparatorStyle,
    conv_templates,
    tokenizer_image_token,
)


SELECTED_FRAMES_PATH = ""


@register_model("llava_vid_select")
class LlavaVidSelect(LlavaVid):
    def __init__(
        self,
        *args,
        selected_frames_path: str = SELECTED_FRAMES_PATH,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
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
            frame_idx = np.linspace(0, len(vr) - 1, self.max_frames_num, dtype=int)
        else:
            frame_idx = self.selected_frames_by_doc_id[int(doc_id)]
        return _sample_frames(vr, video_path, frame_idx)

    def _build_prompt(self, context):
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + context
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + context
        conv = copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template else conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv, conv.get_prompt()

    def _tokenize(self, prompts):
        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if "llama_3" in self.conv_template:
            pad_token_id = 0
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(self.device)
        return input_ids, input_ids.ne(pad_token_id).to(self.device), pad_token_id

    def _preprocess_frames(self, frames):
        return self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)

    def _build_gen_kwargs(self, gen_kwargs, conv, input_ids):
        current_gen_kwargs = dict(gen_kwargs)
        current_gen_kwargs.pop("until", None)
        current_gen_kwargs.setdefault("max_new_tokens", 1024)
        current_gen_kwargs.setdefault("temperature", 0)
        current_gen_kwargs.setdefault("top_p", None)
        current_gen_kwargs.setdefault("num_beams", 1)
        temperature = current_gen_kwargs.get("temperature") or 0
        current_gen_kwargs["do_sample"] = True if temperature > 0 else False
        current_gen_kwargs["modalities"] = "video"
        current_gen_kwargs["stopping_criteria"] = [
            KeywordsStoppingCriteria([conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2], self.tokenizer, input_ids)
        ]
        return current_gen_kwargs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
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
                videos = [self._preprocess_frames(frames)]

                conv, prompt = self._build_prompt(context)
                input_ids, attention_masks, pad_token_id = self._tokenize([prompt])
                current_gen_kwargs = self._build_gen_kwargs(gen_kwargs, conv, input_ids)

                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_start = time.perf_counter()
                    cont = self.model.generate(
                        inputs=input_ids,
                        images=videos,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_id,
                        use_cache=self.use_cache,
                        **current_gen_kwargs,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_latency_ms = (time.perf_counter() - generate_start) * 1000

                sequences = cont.sequences if hasattr(cont, "sequences") else cont
                outputs = [text.strip() for text in self.tokenizer.batch_decode(sequences, skip_special_tokens=True)]
                res.extend(GenerationResult(text=text, metadata={"generate_latency_ms": generate_latency_ms}) for text in outputs)
                self.cache_hook.add_partial("generate_until", (context, current_gen_kwargs), outputs)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
