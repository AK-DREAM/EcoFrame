import json
import os
from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from core.adapters.internvl3 import InternVL3CToFAdapter
from core.runner import CToFRunner
from core.utils import ctof_num_options
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3 import DEFAULT_GEN_KWARGS, InternVL3


@register_model("internvl3_refine")
class InternVL3Refine(InternVL3):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3-8B",
        modality: str = "video",
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: str = "1",
        num_frame: int = 32,
        max_num: int = 12,
        total_max_num: int = 64,
        use_flash_attn: bool = True,
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
    ):
        assert modality == "video", "internvl3_refine only supports modality=video"

        super().__init__(
            pretrained=pretrained,
            modality=modality,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            num_frame=num_frame,
            max_num=max_num,
            total_max_num=total_max_num,
            use_flash_attn=use_flash_attn,
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

        from core.adapters.internvl3_attention import apply_internvl3_attention_patch

        apply_internvl3_attention_patch()

    def _normalize_frame_schedule(self, frame_schedule):
        if frame_schedule is None or frame_schedule == "":
            return None
        if isinstance(frame_schedule, str):
            frame_schedule = json.loads(frame_schedule)
        return [int(x) for x in frame_schedule]

    def _build_query_text(self, doc, context):
        if isinstance(doc, dict) and "question" in doc:
            options = doc.get("options", [])
            options = "\n".join(options) if isinstance(options, list) else str(options)
            return f"{doc['question']}\n{options}".strip()
        return context.strip()

    def _build_gen_kwargs(self, gen_kwargs):
        current_gen_kwargs = {**DEFAULT_GEN_KWARGS, **gen_kwargs}
        current_gen_kwargs.pop("until", None)
        return {k: v for k, v in current_gen_kwargs.items() if k in DEFAULT_GEN_KWARGS}

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        input_size = getattr(self._config, "force_image_size", None) or getattr(self._config.vision_config, "image_size", 448)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits = zip(*chunk)
            for context, gen_kwargs, doc_to_visual, doc_id, task, split in zip(
                contexts, all_gen_kwargs, doc_to_visuals, doc_ids, tasks, splits
            ):
                doc = self.task_dict[task][split][doc_id]
                video_path = doc_to_visual(doc)[0]
                query_text = self._build_query_text(doc, context)
                video_id = os.path.splitext(os.path.basename(str(video_path)))[0]

                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]

                adapter = InternVL3CToFAdapter(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    video_path=video_path,
                    context=context,
                    gen_kwargs=self._build_gen_kwargs(gen_kwargs),
                    use_cache=None,
                    num_options=ctof_num_options(task),
                    max_num=self.max_num,
                    total_max_num=self.total_max_num,
                    input_size=input_size,
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
                    ans = self.tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)[0]
                    ans = ans.split(outputs["stop_str"])[0].strip()
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]

                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
