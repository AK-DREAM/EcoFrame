import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from lmms_eval.tasks import get_task_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Select frames from cached video embeddings.")

    parser.add_argument("--dataset_name", type=str, default="videomme")
    parser.add_argument("--cache_root", type=str, default="./cache/features")
    parser.add_argument("--encoder_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--max_num_frames", type=int, default=32)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--t1", type=float, default=0.8)
    parser.add_argument("--t2", type=float, default=-100)
    parser.add_argument("--all_depth", type=int, default=5)
    parser.add_argument("--output_file", type=str, default="./baselines/selected_frames_aks")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.encoder_name is None:
        args.encoder_name = args.encoder_model_name.rsplit("/", 1)[-1]
    return args


def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score) // 2]
            score2 = score[len(score) // 2:]
            fn1 = fn[:len(score) // 2]
            fn2 = fn[len(score) // 2:]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []

    return no_split_scores + all_split_score, no_split_fn + all_split_fn


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


def select_frames(score, frame_idx, args):
    nums = int(len(score) / args.ratio)
    score = np.array([score[num * args.ratio] for num in range(nums)])
    frame_idx = [int(frame_idx[num * args.ratio]) for num in range(nums)]

    if len(score) >= args.max_num_frames:
        normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
        split_scores, split_fns = meanstd(
            len(score),
            [dict(score=normalized_data, depth=0)],
            args.max_num_frames,
            [frame_idx],
            args.t1,
            args.t2,
            args.all_depth,
        )
        out = []
        for s, f in zip(split_scores, split_fns):
            f_num = int(args.max_num_frames / 2 ** (s["depth"]))
            topk = heapq.nlargest(f_num, range(len(s["score"])), s["score"].__getitem__)
            out.extend([int(f[t]) for t in topk])
        out.sort()
        return out

    return frame_idx


def main(args):
    task = get_task_dict([args.dataset_name])[args.dataset_name]
    datas = task.dataset[task.config.test_split]
    model = CLIPModel.from_pretrained(
        args.encoder_model_name,
        torch_dtype=torch.float16,
        device_map={"": args.device},
    ).eval()
    processor = CLIPProcessor.from_pretrained(args.encoder_model_name)

    outs = []
    current_video_id = None
    current_image_features = None
    current_frame_idx = None
    device_type = args.device.split(":")[0]

    for data in tqdm(datas, desc=f"{args.dataset_name}-{args.encoder_name}", dynamic_ncols=True):
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
        outs.append(select_frames(score, current_frame_idx, args))

    output_path = get_output_path(args)
    with open(output_path, "w") as f:
        json.dump(outs, f)
    print(f"Saved selected frames to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
