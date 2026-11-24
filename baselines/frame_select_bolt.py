import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from lmms_eval.tasks import get_task_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Select frames from cached video embeddings.")

    parser.add_argument("--dataset_name", type=str, default="activitynetqa")
    parser.add_argument("--cache_root", type=str, default="./cache/features")
    parser.add_argument("--encoder_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--max_num_frames", type=int, default=32)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--output_file", type=str, default="./baselines/selected_frames_bolt")
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


def select_frames(score, frame_idx, args):
    nums = int(len(score) / args.ratio)
    score = np.array([score[num * args.ratio] for num in range(nums)], dtype=np.float64)
    frame_idx = [int(frame_idx[num * args.ratio]) for num in range(nums)]

    score = score - np.min(score)
    score = score / np.max(score)
    score = score ** args.alpha
    cdf = np.cumsum(score / np.sum(score))
    thresholds = np.linspace(
        1 / args.max_num_frames,
        1 - 1 / args.max_num_frames,
        args.max_num_frames,
    )
    sampled = np.searchsorted(cdf, thresholds, side="left")
    return [frame_idx[i] for i in sampled]


def main(args):
    task = get_task_dict([args.dataset_name])[args.dataset_name]
    datas = task.dataset[task.config.test_split]
    model = AutoModel.from_pretrained(
        args.encoder_model_name,
        torch_dtype=torch.float16,
        device_map={"": args.device},
    ).eval()
    processor = AutoProcessor.from_pretrained(args.encoder_model_name)

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
