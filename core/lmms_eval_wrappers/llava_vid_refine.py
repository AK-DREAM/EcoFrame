import copy
import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from core.adapters.llava_video import LlavaVideoCToFAdapter
from core.runner import CToFRunner
from core.utils import ctof_num_options
from lmms_eval import utils
from lmms_eval.api.instance import Instance
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


@register_model("llava_vid_refine")
class LlavaVidRefine(LlavaVid):
    def __init__(
        self,
        pretrained: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        truncation: Optional[bool] = True,
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = "llava_qwen",
        attn_implementation=("sdpa" if torch.__version__ >= "2.1.2" else "eager"),
        device_map="cuda:0",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 32,
        video_fps: int = 1,
        mm_resampler_type: str = "spatial_pool",
        mm_spatial_pool_stride: int = 2,
        mm_spatial_pool_out_channels: int = 1024,
        mm_spatial_pool_mode: str = "average",
        mm_resampler_location: str = "before",
        mm_newline_position: str = "grid",
        overwrite: bool = True,
        video_decode_backend: str = "decord",
        delay_load: bool = False,
        tie_weights: bool = True,
        force_sample: bool = False,
        add_time_instruction: bool = False,
        add_faster_video: bool = False,
        faster_token_stride: int = 10,
        frame_schedule: Optional[List[int]] = [4,8,16,32],
        lambda_attn: Optional[float] = 0.5,
        lambda_rele: Optional[float] = 1.0,
        candidate_multiplier: Optional[int] = 4,
        encoder_model_name: Optional[str] = "openai/clip-vit-large-patch14",
        image_cache_dir: Optional[str] = None,
        frame_debug_dir: Optional[str] = None,
        ctof_attn_layers: Optional[Union[int, float, str]] = 19.21,
        early_exit: bool = False,
        entropy_threshold_start: float = 0.1,
        entropy_threshold_step: float = 0.2,
        **kwargs,
    ) -> None:
        assert video_decode_backend == "decord", "llava_vid_refine only supports video_decode_backend=decord"

        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            torch_dtype=torch_dtype,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            max_frames_num=max_frames_num,
            video_fps=video_fps,
            mm_resampler_type=mm_resampler_type,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_out_channels=mm_spatial_pool_out_channels,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
            mm_resampler_location=mm_resampler_location,
            mm_newline_position=mm_newline_position,
            overwrite=overwrite,
            video_decode_backend=video_decode_backend,
            delay_load=delay_load,
            tie_weights=tie_weights,
            force_sample=force_sample,
            add_time_instruction=add_time_instruction,
            add_faster_video=add_faster_video,
            faster_token_stride=faster_token_stride,
            **kwargs,
        )

        self.frame_schedule = self._normalize_frame_schedule(frame_schedule)
        self.lambda_attn = float(lambda_attn)
        self.lambda_rele = float(lambda_rele)
        self.candidate_multiplier = int(candidate_multiplier)
        self.encoder_model_name = encoder_model_name
        self.image_cache_dir = image_cache_dir
        self.frame_debug_dir = frame_debug_dir
        self.ctof_attn_layers = ctof_attn_layers
        self.ctof_runner = CToFRunner(
            frame_schedule=self.frame_schedule,
            lambda_attn=self.lambda_attn,
            lambda_rele=self.lambda_rele,
            candidate_multiplier=self.candidate_multiplier,
            encoder_model_name=self.encoder_model_name,
            image_cache_dir=self.image_cache_dir,
            frame_debug_dir=self.frame_debug_dir,
            attn_layers=self.ctof_attn_layers,
            early_exit=early_exit,
            entropy_threshold_start=entropy_threshold_start,
            entropy_threshold_step=entropy_threshold_step,
        )

        from core.adapters.llava_vid_attention import apply_llava_vid_attention_patch

        apply_llava_vid_attention_patch()

    def _normalize_frame_schedule(self, frame_schedule):
        if frame_schedule is None or frame_schedule == "":
            return None
        if isinstance(frame_schedule, str):
            frame_schedule = json.loads(frame_schedule)
        return [int(x) for x in frame_schedule]

    def _build_prompt(self, context):
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + context
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + context
        conv = copy.deepcopy(conv_templates[self.conv_template]) if "llama_3" in self.conv_template else conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv, conv.get_prompt()

    def _build_query_text(self, doc, context):
        if isinstance(doc, dict) and "question" in doc:
            options = doc.get("options", [])
            options = "\n".join(options) if isinstance(options, list) else str(options)
            return f"{doc['question']}\n{options}".strip()
        return context.replace(DEFAULT_IMAGE_TOKEN, "").strip()

    def _tokenize(self, prompts):
        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if "llama_3" in self.conv_template:
            pad_token_id = 0
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(self.device)
        return input_ids, input_ids.ne(pad_token_id).to(self.device), pad_token_id

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
                conv, prompt = self._build_prompt(context)
                query_text = self._build_query_text(doc, context)
                video_id = os.path.splitext(os.path.basename(str(video_path)))[0]
                input_ids, attention_masks, pad_token_id = self._tokenize([prompt])
                current_gen_kwargs = self._build_gen_kwargs(gen_kwargs, conv, input_ids)

                self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

                adapter = LlavaVideoCToFAdapter(
                    model=self.model,
                    image_processor=self._image_processor,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    inputs=input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_id,
                    gen_kwargs=current_gen_kwargs,
                    use_cache=self.use_cache,
                    video_path=video_path,
                    num_options=ctof_num_options(task),
                )

                with torch.inference_mode():
                    cont = self.ctof_runner.run(
                        adapter,
                        video_path,
                        video_id,
                        query_text,
                        {
                            "task": task,
                            "split": split,
                            "doc_id": int(doc_id) if isinstance(doc_id, (int, np.integer)) else doc_id,
                        },
                    )

                if isinstance(cont, str):
                    text_outputs = [cont]
                else:
                    sequences = cont.sequences if hasattr(cont, "sequences") else cont
                    text_outputs = [text.strip() for text in self.tokenizer.batch_decode(sequences, skip_special_tokens=True)]
                res.extend(text_outputs)
                self.cache_hook.add_partial("generate_until", (context, current_gen_kwargs), text_outputs)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
