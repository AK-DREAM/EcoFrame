import copy
import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from core.adapters.llava_onevision import LlavaOneVisionCToFAdapter
from core.runner import CToFRunner
from core.utils import ctof_num_options
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.llava_onevision import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    KeywordsStoppingCriteria,
    Llava_OneVision,
    SeparatorStyle,
    best_fit_attn_implementation,
    conv_templates,
    eval_logger,
    tokenizer_image_token,
)


@register_model("llava_onevision_refine")
class Llava_OneVision_Refine(Llava_OneVision):
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "qwen_1_5",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,
        customized_config: Optional[str] = None,
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",
        video_decode_backend: str = "decord",
        frame_schedule: Optional[List[int]] = [4,8,16,32],
        lambda_attn: Optional[float] = 0.5,
        lambda_rele: Optional[float] = 1.0,
        candidate_multiplier: Optional[int] = 4,
        encoder_model_name: Optional[str] = "openai/clip-vit-large-patch14",
        image_cache_dir: Optional[str] = None,
        frame_debug_dir: Optional[str] = None,
        ctof_attn_layers: Optional[Union[int, float, str]] = 19.21,
        open_ended: bool = False,
        entropy_top_k: int = 20,
        early_exit: bool = False,
        entropy_threshold_start: float = 0.1,
        entropy_threshold_step: float = 0.2,
        **kwargs,
    ) -> None:
        assert video_decode_backend == "decord", "llava_onevision_refine only supports video_decode_backend=decord"
        assert token_strategy == "single", "llava_onevision_refine expects token_strategy=single"

        super().__init__(
            pretrained=pretrained,
            truncation=truncation,
            device=device,
            batch_size=batch_size,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device_map,
            conv_template=conv_template,
            use_cache=use_cache,
            truncate_context=truncate_context,
            customized_config=customized_config,
            max_frames_num=max_frames_num,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
            token_strategy=token_strategy,
            video_decode_backend=video_decode_backend,
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
        self.open_ended = open_ended
        self.entropy_top_k = int(entropy_top_k)
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

        from core.adapters.llava_onevision_attention import apply_llava_onevision_attention_patch
        apply_llava_onevision_attention_patch()

    def _normalize_frame_schedule(self, frame_schedule):
        if frame_schedule is None or frame_schedule == "":
            return None
        if isinstance(frame_schedule, str):
            frame_schedule = json.loads(frame_schedule)
        return [int(x) for x in frame_schedule]

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

    def _build_query_text(self, doc, context):
        if isinstance(doc, dict) and "question" in doc:
            options = doc.get("options", [])
            options = "\n".join(options) if isinstance(options, list) else str(options)
            return f"{doc['question']}\n{options}".strip()
        return context.replace(DEFAULT_IMAGE_TOKEN, "").strip()

    def _tokenize(self, prompts):
        input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompts]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id).to(self.device)
        return input_ids, input_ids.ne(pad_token_id).to(self.device), pad_token_id

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

                conv, prompt = self._build_prompt(context)
                query_text = self._build_query_text(doc, context)
                video_id = os.path.splitext(os.path.basename(str(video_path)))[0]
                input_ids, attention_masks, pad_token_id = self._tokenize([prompt])

                current_gen_kwargs = dict(gen_kwargs)
                current_gen_kwargs.pop("until", None)
                current_gen_kwargs.setdefault("max_new_tokens", 1024)
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

                adapter = LlavaOneVisionCToFAdapter(
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
                    open_ended=self.open_ended,
                    entropy_top_k=self.entropy_top_k,
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
