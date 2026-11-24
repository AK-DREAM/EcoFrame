import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from core.adapters.qwen2_5_vl import Qwen25VLCToFAdapter
from core.runner import CToFRunner
from core.utils import ctof_num_options
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


@register_model("qwen2_5_vl_refine")
class Qwen2_5_VL_Refine(Qwen2_5_VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
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
        assert interleave_visuals is False, "qwen2_5_vl_refine only supports non-interleaved video QA"
        assert attn_implementation == "flash_attention_2", "qwen2_5_vl_refine expects attn_implementation=flash_attention_2"

        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            fps=fps,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
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

        from core.adapters.qwen2_5_vl_attention import apply_qwen2_5_vl_attention_patch
        apply_qwen2_5_vl_attention_patch()

    def _normalize_frame_schedule(self, frame_schedule):
        if frame_schedule is None or frame_schedule == "":
            return None
        if isinstance(frame_schedule, str):
            frame_schedule = json.loads(frame_schedule)
        return [int(x) for x in frame_schedule]

    def _build_context(self, context):
        context = context.replace("<image>", "").strip()
        if self.reasoning_prompt:
            context = context + self.reasoning_prompt
        return context

    def _build_query_text(self, doc, context):
        if isinstance(doc, dict) and "question" in doc:
            options = doc.get("options", [])
            options = "\n".join(options) if isinstance(options, list) else str(options)
            return f"{doc['question']}\n{options}".strip()
        return context.strip()

    def _build_gen_kwargs(self, gen_kwargs):
        current_gen_kwargs = {
            "max_new_tokens": 32768,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            **gen_kwargs,
        }
        current_gen_kwargs.pop("until", None)
        if current_gen_kwargs["temperature"] > 0:
            current_gen_kwargs["do_sample"] = True
        else:
            current_gen_kwargs["do_sample"] = False
            current_gen_kwargs["temperature"] = None
            current_gen_kwargs["top_p"] = None
        current_gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        current_gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        return current_gen_kwargs

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        padding_side = "left" if self.batch_size > 1 else "right"

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            for context, gen_kwargs, doc_to_visual, doc_id, task, split in zip(
                contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits
            ):
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)[0]
                context = self._build_context(context)
                query_text = self._build_query_text(doc, context)
                video_id = os.path.splitext(os.path.basename(str(video_path)))[0]

                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
                until = [item for item in until if item != "\n\n"]

                adapter = Qwen25VLCToFAdapter(
                    model=self.model,
                    processor=self.processor,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    video_path=video_path,
                    context=context,
                    system_prompt=self.system_prompt,
                    gen_kwargs=self._build_gen_kwargs(gen_kwargs),
                    use_cache=self.use_cache,
                    num_options=ctof_num_options(task),
                    max_pixels=self.max_pixels,
                    min_pixels=self.min_pixels,
                    padding_side=padding_side,
                    open_ended=self.open_ended,
                    entropy_top_k=self.entropy_top_k,
                )

                with torch.inference_mode():
                    outputs = self.ctof_runner.run(
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

                if isinstance(outputs, str):
                    ans = outputs
                else:
                    sequences = outputs["sequences"].sequences if hasattr(outputs["sequences"], "sequences") else outputs["sequences"]
                    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(outputs["input_ids"], sequences)]
                    answers = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    ans = answers[0]
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]

                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
