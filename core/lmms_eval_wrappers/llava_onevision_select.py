import copy
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from core.utils import _sample_frames, ctof_num_options
from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_onevision import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    KeywordsStoppingCriteria,
    Llava_OneVision,
    SeparatorStyle,
    conv_templates,
    eval_logger,
    tokenizer_image_token,
)


SELECTED_FRAMES_PATH = ""


@register_model("llava_onevision_select")
class Llava_OneVision_Select(Llava_OneVision):
    def __init__(
        self,
        *args,
        selected_frames_path: str = SELECTED_FRAMES_PATH,
        open_ended: bool = False,
        entropy_top_k: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.selected_frames_path = selected_frames_path
        self.selected_frames_by_doc_id = self._load_selected_frames()
        self.open_ended = open_ended
        self.entropy_top_k = int(entropy_top_k)

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
        question = f"{DEFAULT_IMAGE_TOKEN}\n{context}" if DEFAULT_IMAGE_TOKEN not in context else context
        conv = copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template else conv_templates[self.conv_template].copy()

        if utils.is_json(question):
            question = json.loads(question)
            for idx, item in enumerate(question):
                conv.append_message(conv.roles[idx % 2], item["value"])
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
        return conv, conv.get_prompt()

    def _tokenize(self, prompts):
        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(self.device)
        return input_ids, input_ids.ne(pad_token_id).to(self.device), pad_token_id

    def _compute_option_entropy(self, first_token_scores, num_options):
        option_ids = [
            self.tokenizer.encode(chr(ord("A") + i), add_special_tokens=False)[0]
            for i in range(int(num_options))
        ]
        option_logits = first_token_scores[:, option_ids].float()
        option_probs = torch.softmax(option_logits, dim=-1)
        option_entropy = -(option_probs * torch.log(option_probs.clamp_min(1e-12))).sum(dim=-1)
        return [
            {"option_entropy": float(entropy.detach().cpu().item())}
            for entropy in option_entropy
        ]

    def _compute_sequence_entropy(self, score_tensors):
        step_entropies = []
        for scores in score_tensors:
            log_probs = torch.log_softmax(scores.float(), dim=-1)
            topk_log_probs = log_probs.topk(min(self.entropy_top_k, log_probs.shape[-1]), dim=-1).values
            topk_probs = topk_log_probs.exp()
            step_entropies.append(-(topk_probs * topk_log_probs).sum(dim=-1))
        sequence_entropy = torch.stack(step_entropies, dim=0).mean(0)
        return [
            {"option_entropy": float(entropy.detach().cpu().item())}
            for entropy in sequence_entropy
        ]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            for context, gen_kwargs, doc_to_visual, doc_id, task, split in zip(
                contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits
            ):
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)[0]

                if origin_image_aspect_ratio is not None and self._config.image_aspect_ratio != origin_image_aspect_ratio:
                    self._config.image_aspect_ratio = origin_image_aspect_ratio
                    eval_logger.info(f"Resetting image aspect ratio to {origin_image_aspect_ratio}")

                frames = self.load_video_select(video_path, doc_id)
                image_tensor = [self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().to(self._device)]

                conv, prompt = self._build_prompt(context)
                input_ids, attention_masks, pad_token_id = self._tokenize([prompt])

                current_gen_kwargs = dict(gen_kwargs)
                current_gen_kwargs.pop("until", None)
                current_gen_kwargs.setdefault("max_new_tokens", 1024)
                current_gen_kwargs["return_dict_in_generate"] = True
                current_gen_kwargs["output_scores"] = True
                current_gen_kwargs["modalities"] = ["video"]
                current_gen_kwargs["stopping_criteria"] = [
                    KeywordsStoppingCriteria([conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2], self.tokenizer, input_ids)
                ]
                current_gen_kwargs.pop("image_aspect_ratio", None)
                if not current_gen_kwargs.get("do_sample", False):
                    current_gen_kwargs.pop("temperature", None)
                    current_gen_kwargs.pop("top_p", None)
                    current_gen_kwargs.pop("top_k", None)

                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_start = time.perf_counter()
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_id,
                        images=image_tensor,
                        use_cache=self.use_cache,
                        **current_gen_kwargs,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_latency_ms = (time.perf_counter() - generate_start) * 1000

                sequences = cont.sequences if hasattr(cont, "sequences") else cont
                score_tensors = cont.scores if hasattr(cont, "sequences") else None
                text_outputs = [text.strip() for text in self.tokenizer.batch_decode(sequences, skip_special_tokens=True)]
                if score_tensors and self.open_ended:
                    confidence_metadata = self._compute_sequence_entropy(score_tensors)
                elif score_tensors:
                    confidence_metadata = self._compute_option_entropy(score_tensors[0], ctof_num_options(task))
                else:
                    confidence_metadata = [None] * len(text_outputs)

                res.extend(
                    GenerationResult(text=text, metadata={**(metadata or {}), "generate_latency_ms": generate_latency_ms})
                    for text, metadata in zip(text_outputs, confidence_metadata)
                )
                self.cache_hook.add_partial("generate_until", (context, current_gen_kwargs), text_outputs)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
