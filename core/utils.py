import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from transformers import AutoModel, AutoProcessor


def _tic(device=None):
    if device is not None and torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()

def _toc(start, device=None):
    return round((_tic(device) - start) * 1000.0, 3)

def _sample_frames(vr, video_path, frame_indices):
    frame_indices = [int(idx) for idx in frame_indices]
    try:
        return vr.get_batch(frame_indices).asnumpy()
    except Exception:
        cap, frames = cv2.VideoCapture(video_path), []
        try:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"OpenCV failed to read frame {idx} from {video_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return np.stack(frames)

def _parse_attn_layer(attn_layer):
    if attn_layer is None:
        attn_layer = 19.21
    if isinstance(attn_layer, float):
        start, end = map(int, f"{attn_layer:.2f}".split("."))
        return list(range(start, end + 1))
    if isinstance(attn_layer, int):
        return [attn_layer]
    if isinstance(attn_layer, str):
        if attn_layer.lower() == "none":
            return None
        if "." in attn_layer:
            start, end = map(int, attn_layer.split("."))
            return list(range(start, end + 1))
        return [int(attn_layer)]
    return attn_layer

def ctof_num_options(task):
    return {"videomme": 4, "longvideobench_val_v": 5}.get(task, 4)

def _nearest_selected_positions(selected_indices_np, candidate_indices_np):
    right_pos = np.searchsorted(selected_indices_np, candidate_indices_np, side="left")
    right_pos = np.clip(right_pos, 0, len(selected_indices_np) - 1)
    left_pos = np.clip(right_pos - 1, 0, len(selected_indices_np) - 1)
    left_dist = np.abs(candidate_indices_np - selected_indices_np[left_pos])
    right_dist = np.abs(candidate_indices_np - selected_indices_np[right_pos])
    return np.where(left_dist <= right_dist, left_pos, right_pos)

def _greedy_select_frames(total_frames, selected_indices, candidate_indices, candidate_scores, select_num, score_lambda):
    """Greedily select frames by multiplicative relevance and dynamic coverage.

    Args:
        total_frames: Number of frames in the video.
        selected_indices: Frames used as coverage seeds and excluded from selection.
        candidate_indices: Frames allowed to be selected from.
        candidate_scores: Relevance score for each candidate frame.
        select_num: Number of new frames to select.
        score_lambda: Exponent applied to candidate_scores.
    """
    selected_indices_np = np.sort(np.asarray(selected_indices, dtype=np.int64))
    candidate_indices_np = np.asarray(candidate_indices, dtype=np.int64)
    candidate_scores_np = np.asarray(candidate_scores, dtype=np.float32)
    frame_scale = 1.0 / float(total_frames)
    eps = 1e-12

    selected_mask = np.isin(candidate_indices_np, selected_indices_np)

    if len(selected_indices_np) == 0:
        coverage_scores = np.ones(len(candidate_indices_np), dtype=np.float32)
    else:
        nearest_pos = _nearest_selected_positions(selected_indices_np, candidate_indices_np)
        coverage_scores = np.abs(candidate_indices_np - selected_indices_np[nearest_pos]).astype(np.float32) * frame_scale

    added = []

    for _ in range(select_num):
        if not np.any(~selected_mask):
            break
        aggregated_scores = float(score_lambda) * np.log(np.clip(candidate_scores_np, eps, None)) + np.log(np.clip(coverage_scores, eps, None))
        aggregated_scores[selected_mask] = -np.inf
        chosen_pos = int(np.argmax(aggregated_scores))
        chosen = int(candidate_indices_np[chosen_pos])
        added.append(chosen)
        selected_mask[candidate_indices_np == chosen] = True
        coverage_scores = np.minimum(coverage_scores, np.abs(candidate_indices_np - chosen) * frame_scale)

    return added

def _random_select_frames(selected_indices, candidate_indices, select_num):
    candidate_indices_np = np.asarray(candidate_indices, dtype=np.int64)
    selected_mask = np.isin(candidate_indices_np, np.asarray(selected_indices, dtype=np.int64))
    added = []
    for _ in range(select_num):
        if not np.any(~selected_mask):
            break
        avail_pos = np.flatnonzero(~selected_mask)
        chosen_pos = int(np.random.choice(avail_pos))
        chosen = int(candidate_indices_np[chosen_pos])
        added.append(chosen)
        selected_mask[candidate_indices_np == chosen] = True
    return added

class _EmbedEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        if torch.device(device).type == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def encode_images(self, images):
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(self.model.parameters()).dtype)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs).float()
        return F.normalize(image_embeds, dim=-1).cpu()

    def encode_text(self, text):
        inputs = self.processor(text=[text], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs).float()
        return F.normalize(text_embeds, dim=-1).squeeze(0).cpu()


def _get_encoder(owner, model_name, device):
    if not hasattr(owner, "_ctof_encoder"):
        owner._ctof_encoder = _EmbedEncoder(model_name, device)
    return owner._ctof_encoder


def _load_encoder_image_cache(cache_dir, video_id):
    if cache_dir is None:
        return {}, None
    if video_id is None:
        raise ValueError("video_id is required when image_cache_dir is set")

    cache_path = os.path.join(cache_dir, f"{video_id}.npz")
    with np.load(cache_path) as data:
        frame_indices = data["frame_idx"].astype(np.int64)
        image_embeds = F.normalize(torch.from_numpy(data["embedding"]).float(), dim=-1).cpu()
    return {int(idx): embed for idx, embed in zip(frame_indices.tolist(), image_embeds)}, frame_indices


def _extend_candidate_pool(
    total_frames,
    candidate_pool_indices,
    selected_indices,
    frame_scores,
    proposal_num,
    lambda_attn,
    avail_indices,
):
    avail_indices_np = np.asarray(avail_indices, dtype=np.int64)
    if frame_scores is None:
        new_candidate_indices = avail_indices_np[np.linspace(0, len(avail_indices_np) - 1, proposal_num, dtype=int)].astype(int).tolist()
    else:
        selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
        nearest_coarse_pos = _nearest_selected_positions(selected_indices_np, avail_indices_np)
        proposal_scores = np.asarray(frame_scores, dtype=np.float32)[nearest_coarse_pos]
        new_candidate_indices = _greedy_select_frames(total_frames, candidate_pool_indices, avail_indices_np, proposal_scores, proposal_num, lambda_attn)

    candidate_pool_indices = sorted([*candidate_pool_indices, *new_candidate_indices])
    return candidate_pool_indices


def _encoder_candidate_scores(vr, candidate_indices, query_text, encoder, image_cache, text_cache):
    profile = {"video_decode_ms": 0.0, "frame_encode_ms": 0.0}
    missing_indices = list(dict.fromkeys(int(idx) for idx in candidate_indices if int(idx) not in image_cache))
    if missing_indices:
        decode_start = _tic()
        frames = vr.get_batch(missing_indices).asnumpy()
        profile["video_decode_ms"] = _toc(decode_start)
        encode_start = _tic(encoder.device)
        image_embeds = encoder.encode_images(frames)
        profile["frame_encode_ms"] = _toc(encode_start, encoder.device)
        for frame_idx, image_embed in zip(missing_indices, image_embeds):
            image_cache[frame_idx] = image_embed

    if "embed" not in text_cache:
        text_cache["embed"] = encoder.encode_text(query_text).float()
    image_embeds = torch.stack([image_cache[int(idx)] for idx in candidate_indices]).float()
    scores = image_embeds @ text_cache["embed"]
    z_scores = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-6)
    relevance_scores = (len(candidate_indices) * torch.softmax(z_scores, dim=0)).tolist()
    return relevance_scores, profile

def _select_from_candidate_pool(
    vr,
    total_frames,
    candidate_pool_indices,
    target_num,
    lambda_rele,
    query_text,
    encoder,
    encoder_image_cache,
    encoder_text_cache,
):
    candidate_pool_relevance_scores, profile = _encoder_candidate_scores(
        vr,
        candidate_pool_indices,
        query_text,
        encoder,
        encoder_image_cache,
        encoder_text_cache,
    )
    selected = _greedy_select_frames(
        total_frames,
        [],
        candidate_pool_indices,
        candidate_pool_relevance_scores,
        target_num,
        lambda_rele,
    )
    return sorted(selected), [round(float(x), 2) for x in candidate_pool_relevance_scores], profile

def _write_debug(frame_debug_dir, payload):
    if frame_debug_dir is None:
        return
    os.makedirs(frame_debug_dir, exist_ok=True)
    path = os.path.join(frame_debug_dir, "debug.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
