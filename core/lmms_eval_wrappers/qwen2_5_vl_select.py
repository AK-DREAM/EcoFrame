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
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL


SELECTED_FRAMES_PATH = ""


@register_model("qwen2_5_vl_select")
class Qwen2_5_VL_Select(Qwen2_5_VL):
    def __init__(
        self,
        *args,
        selected_frames_path: str = SELECTED_FRAMES_PATH,
        open_ended: bool = False,
        entropy_top_k: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert self.interleave_visuals is False, "qwen2_5_vl_select only supports non-interleaved video QA"
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
        vr = VideoReader(video_path, ctx=cpu(0))
        if self.selected_frames_by_doc_id is None:
            total_frames = len(vr)
            frame_idx = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            if total_frames - 1 not in frame_idx:
                frame_idx = np.unique(np.append(frame_idx, total_frames - 1))
        else:
            frame_idx = self.selected_frames_by_doc_id[int(doc_id)]
        return _sample_frames(vr, video_path, frame_idx)

    def _build_context(self, context):
        context = context.replace("<image>", "").strip()
        if self.reasoning_prompt:
            context = context + self.reasoning_prompt
        return context

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
        current_gen_kwargs["return_dict_in_generate"] = True
        current_gen_kwargs["output_scores"] = True
        return current_gen_kwargs

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

    def _compute_sequence_truncated_entropy(self, score_tensors):
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
                frames = self.load_video_select(video_path, doc_id)

                until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
                until = [item for item in until if item != "\n\n"]

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            },
                            {"type": "text", "text": context},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=[text],
                    videos=[frames],
                    padding=True,
                    padding_side=padding_side,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)
                current_gen_kwargs = self._build_gen_kwargs(gen_kwargs)

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_start = time.perf_counter()
                    outputs = self.model.generate(
                        **inputs,
                        **current_gen_kwargs,
                        use_cache=self.use_cache,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    generate_latency_ms = (time.perf_counter() - generate_start) * 1000

                sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
                score_tensors = outputs.scores if hasattr(outputs, "sequences") else None
                generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
                ans = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]

                metadata = {"generate_latency_ms": generate_latency_ms}
                if score_tensors:
                    if self.open_ended:
                        metadata.update(self._compute_sequence_truncated_entropy(score_tensors)[0])
                    else:
                        metadata.update(self._compute_option_entropy(score_tensors[0], ctof_num_options(task))[0])
                res.append(GenerationResult(text=ans, metadata=metadata))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
