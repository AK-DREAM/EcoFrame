import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from lmms_eval.tasks import get_task_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="FOCUS frame selection from cached CLIP embeddings.")
    parser.add_argument("--dataset_name", type=str, default="videomme")
    parser.add_argument("--cache_root", type=str, default="./cache/features")
    parser.add_argument("--encoder_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--max_num_frames", type=int, default=32)
    parser.add_argument("--pool_fps", type=float, default=4.0)
    parser.add_argument("--coarse_every_sec", type=float, default=16.0)
    parser.add_argument("--fine_every_sec", type=float, default=1.0)
    parser.add_argument("--zoom_ratio", type=float, default=0.25)
    parser.add_argument("--final_min_arms", type=int, default=4)
    parser.add_argument("--final_max_arms", type=int, default=32)
    parser.add_argument("--min_coarse_segments", type=int, default=8)
    parser.add_argument("--min_zoom_segments", type=int, default=4)
    parser.add_argument("--extra_samples_per_region", type=int, default=2)
    parser.add_argument("--min_variance_threshold", type=float, default=1e-6)
    parser.add_argument("--fine_uniform_ratio", type=float, default=0.5)
    parser.add_argument("--interpolation_method", type=str, default="nearest", choices=["nearest", "uniform"])
    parser.add_argument("--top_ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.06)
    parser.add_argument("--min_gap_sec", type=float, default=1.0)
    parser.add_argument("--disable_gap_below_sec", type=float, default=0.2)
    parser.add_argument("--gap_ratio_of_avg", type=float, default=0.25)
    parser.add_argument("--region_half_window_sec", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="./baselines/selected_frames_focus")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.encoder_name is None:
        args.encoder_name = args.encoder_model_name.rsplit("/", 1)[-1]
    return args


def get_cache_path(args, video_id):
    return Path(args.cache_root) / args.dataset_name / args.encoder_name / f"{video_id}.npz"


def get_output_path(args):
    out_dir = Path(args.output_file) / args.dataset_name / args.encoder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"selected_frames.json"
    if not base.exists():
        return base
    i = 1
    while True:
        path = out_dir / f"selected_frames_{i}.json"
        if not path.exists():
            return path
        i += 1


def normalize_score(score):
    score = np.asarray(score, dtype=np.float64)
    if score.max() > score.min():
        return (score - score.min()) / (score.max() - score.min())
    return np.ones_like(score)


def setup_arms(total_frames, coarse_stride):
    num_regions = max(1, total_frames // coarse_stride)
    region_size = max(1, total_frames // num_regions)
    arms = []
    for arm_id in range(num_regions):
        start = arm_id * region_size
        end = min((arm_id + 1) * region_size, total_frames) - 1
        arms.append(dict(
            arm_id=arm_id,
            start=start,
            end=end,
            samples=0,
            mean_sim=0.0,
            variance=0.0,
            focus_score=0.0,
            sampled_indices=[],
            sampled_scores=[],
        ))
    return arms


def coarse_sampling(arms, rng, extra_samples_per_region):
    out = []
    for arm in arms:
        start, end = arm["start"], arm["end"]
        center = (start + end) // 2
        sampled = [center]
        candidates = [idx for idx in range(start, end + 1) if idx != center]
        if candidates:
            count = min(extra_samples_per_region, len(candidates))
            sampled.extend(rng.choice(candidates, size=count, replace=False).tolist())
        arm["sampled_indices"] = sampled
        out.extend(sampled)
    return out


def update_arms_with_scores(arms, indices, similarities, min_variance_threshold):
    idx_to_score = {int(idx): float(score) for idx, score in zip(indices, similarities)}
    for arm in arms:
        existing = {idx for idx, _ in arm["sampled_scores"]}
        for idx in idx_to_score:
            if not (arm["start"] <= idx <= arm["end"]):
                continue
            if idx in existing:
                continue
            arm["sampled_indices"].append(idx)
            arm["sampled_scores"].append((idx, idx_to_score[idx]))

        if arm["sampled_scores"]:
            scores = np.array([score for _, score in arm["sampled_scores"]], dtype=np.float64)
            arm["samples"] = len(scores)
            arm["mean_sim"] = float(np.mean(scores))
            arm["variance"] = float(np.var(scores)) if len(scores) > 1 else 0.0

    total_samples = sum(arm["samples"] for arm in arms)
    for arm in arms:
        n = arm["samples"]
        score = arm["mean_sim"]
        if total_samples > 1 and n > 0:
            var = max(arm["variance"], min_variance_threshold)
            score += math.sqrt(max(0.0, 2 * math.log(total_samples) * var / n))
            score += 3 * math.log(total_samples) / n
        arm["focus_score"] = float(score)


def top_focus_arms(arms, zoom_ratio, min_zoom_segments):
    count = max(min_zoom_segments, int(np.ceil(len(arms) * zoom_ratio)))
    return sorted(arms, key=lambda x: x["focus_score"], reverse=True)[: min(count, len(arms))]


def fine_sampling(arms, total_frames, fine_stride, half_window, coarse_indices, args, rng):
    coarse_set = set(coarse_indices)
    out = set()
    for arm in arms:
        center = (arm["start"] + arm["end"]) // 2
        start = max(0, center - half_window)
        end = min(total_frames - 1, center + half_window)
        window = end - start + 1
        required = max(1, window // fine_stride)
        uniform_count = int(round(required * args.fine_uniform_ratio))
        random_count = required - uniform_count

        if uniform_count > 0:
            if window <= uniform_count:
                idx = (start + end) // 2
                if idx not in coarse_set:
                    out.add(idx)
            else:
                step = window / uniform_count
                for i in range(uniform_count):
                    idx = min(start + int(i * step + step / 2), end)
                    if idx not in coarse_set and idx not in out:
                        out.add(idx)

        candidates = [idx for idx in range(start, end + 1) if idx not in coarse_set and idx not in out]
        if random_count > 0 and candidates:
            count = min(random_count, len(candidates))
            out.update(rng.choice(candidates, size=count, replace=False).astype(int).tolist())

    return sorted(out)


def estimate_scores_in_arm(arm):
    sampled = arm["sampled_scores"]
    if not sampled:
        return {}
    sampled_idx = np.array([idx for idx, _ in sampled])
    sampled_score = np.array([score for _, score in sampled])
    return {
        idx: float(sampled_score[int(np.argmin(np.abs(sampled_idx - idx)))])
        for idx in range(arm["start"], arm["end"] + 1)
    }


def select_top_sampled(all_sampled_scores, k, top_ratio):
    count = int(round(top_ratio * min(k, len(all_sampled_scores))))
    ranked = sorted(all_sampled_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:count]]


def select_remaining(arms, remaining, selected, fps, min_gap_sec, args, rng):
    if remaining <= 0:
        return []

    gap = int(round(min_gap_sec * fps)) if min_gap_sec > 0 else 0
    chosen = set(selected)
    num_arms = int(np.ceil(len(arms) * max(0.0, min(1.0, args.zoom_ratio))))
    num_arms = min(max(args.final_min_arms, num_arms), args.final_max_arms, len(arms))
    arms = sorted(arms, key=lambda x: x["focus_score"], reverse=True)[:num_arms]
    per_arm = [remaining // num_arms + (i < remaining % num_arms) for i in range(num_arms)]
    out = []

    for arm, need in zip(arms, per_arm):
        candidates = [idx for idx in range(arm["start"], arm["end"] + 1) if idx not in chosen]
        if not candidates or need <= 0:
            continue

        if args.interpolation_method == "uniform":
            if gap > 0:
                gap_candidates = [idx for idx in candidates if all(abs(idx - prev) >= gap for prev in chosen)]
                candidates = gap_candidates or candidates
            positions = rng.choice(len(candidates), size=min(need, len(candidates)), replace=False)
        else:
            estimated = estimate_scores_in_arm(arm)
            scored = [(idx, estimated[idx]) for idx in candidates if idx in estimated and (gap == 0 or all(abs(idx - prev) >= gap for prev in chosen))]
            if not scored:
                scored = [(idx, estimated[idx]) for idx in candidates if idx in estimated]
            if not scored:
                if gap > 0:
                    gap_candidates = [idx for idx in candidates if all(abs(idx - prev) >= gap for prev in chosen)]
                    candidates = gap_candidates or candidates
                positions = rng.choice(len(candidates), size=min(need, len(candidates)), replace=False)
            else:
                candidates, scores = zip(*scored)
                candidates = list(candidates)
                scores = normalize_score(scores)
                logits = scores / max(args.temperature, 1e-12)
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()
                positions = rng.choice(len(candidates), size=min(need, len(candidates)), replace=False, p=probs)
        for pos in positions:
            idx = int(candidates[pos])
            out.append(idx)
            chosen.add(idx)

    return out


def focus_select(score, args, rng, frame_idx):
    total_frames = len(score)
    duration = total_frames / args.pool_fps
    desired_coarse = max(args.min_coarse_segments, int(np.ceil(duration / max(args.coarse_every_sec, 1e-6))))
    coarse_stride = max(1, int(np.floor(total_frames / desired_coarse)))
    fine_stride = min(max(1, int(round(args.fine_every_sec * args.pool_fps))), coarse_stride)
    if args.region_half_window_sec is None:
        half_window_sec = max(1.0 / args.pool_fps, duration / max(1, desired_coarse) / 2)
    else:
        half_window_sec = args.region_half_window_sec
    half_window = max(1, int(round(half_window_sec * args.pool_fps)))

    arms = setup_arms(total_frames, coarse_stride)
    coarse_indices = coarse_sampling(arms, rng, args.extra_samples_per_region)
    coarse_scores = [score[idx] for idx in coarse_indices]
    update_arms_with_scores(arms, coarse_indices, coarse_scores, args.min_variance_threshold)

    selected_arms = top_focus_arms(arms, args.zoom_ratio, args.min_zoom_segments)
    fine_indices = fine_sampling(selected_arms, total_frames, fine_stride, half_window, coarse_indices, args, rng)
    fine_scores = [score[idx] for idx in fine_indices]
    update_arms_with_scores(arms, fine_indices, fine_scores, args.min_variance_threshold)
    clip_score_frames = len(coarse_indices) + len(fine_indices)

    all_sampled_scores = {idx: float(score[idx]) for idx in coarse_indices}
    all_sampled_scores.update({idx: float(score[idx]) for idx in fine_indices})
    selected = select_top_sampled(all_sampled_scores, args.max_num_frames, args.top_ratio)
    avg_spacing_sec = duration / max(1, args.max_num_frames)
    if avg_spacing_sec <= args.disable_gap_below_sec:
        min_gap_sec = 0.0
    else:
        min_gap_sec = min(args.gap_ratio_of_avg * avg_spacing_sec, args.min_gap_sec)
    selected.extend(select_remaining(arms, args.max_num_frames - len(selected), selected, args.pool_fps, min_gap_sec, args, rng))

    return sorted(list(dict.fromkeys(selected)))[:args.max_num_frames], clip_score_frames


def select_frames(score, frame_idx, args, sample_idx):
    rng = np.random.default_rng(args.seed + sample_idx)
    selected, clip_score_frames = focus_select(normalize_score(score), args, rng, frame_idx)
    return [int(frame_idx[idx]) for idx in selected], clip_score_frames


def main(args):
    task = get_task_dict([args.dataset_name])[args.dataset_name]
    datas = task.dataset[task.config.test_split]
    model = CLIPModel.from_pretrained(args.encoder_model_name, torch_dtype=torch.float16, device_map={"": args.device}).eval()
    processor = CLIPProcessor.from_pretrained(args.encoder_model_name)

    outs = []
    current_video_id = None
    current_image_features = None
    current_frame_idx = None
    device_type = args.device.split(":")[0]
    total_clip_score_frames = 0

    for sample_idx, data in enumerate(tqdm(datas, desc=f"{args.dataset_name}-{args.encoder_name}-focus", dynamic_ncols=True)):
        video_id = Path(task.doc_to_visual(data)[0]).stem
        if video_id != current_video_id:
            cache = np.load(get_cache_path(args, video_id))
            current_image_features = torch.from_numpy(cache["embedding"]).to(args.device)
            current_frame_idx = cache["frame_idx"]
            current_video_id = video_id

        inputs = processor(text=data["question"], return_tensors="pt", padding="max_length", truncation=True).to(args.device)
        with torch.inference_mode(), torch.autocast(device_type, dtype=torch.float16):
            text_features = model.get_text_features(**inputs)
        score = torch.nn.functional.cosine_similarity(text_features, current_image_features, dim=-1).cpu().numpy()
        selected_frames, clip_score_frames = select_frames(score, current_frame_idx, args, sample_idx)
        outs.append(selected_frames)
        total_clip_score_frames += clip_score_frames

    output_path = get_output_path(args)
    with open(output_path, "w") as f:
        json.dump(outs, f)
    print(f"Saved selected frames to {output_path}")
    print(f"CLIP score frames: total={total_clip_score_frames}, avg={total_clip_score_frames / len(datas):.2f}")


if __name__ == "__main__":
    main(parse_arguments())
