import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from lmms_eval.tasks import get_task_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Cache video embeddings.")
    parser.add_argument("--dataset_name", type=str, default="activitynetqa")
    parser.add_argument("--cache_root", type=str, default="./cache/features")
    parser.add_argument("--encoder_model_name", type=str, default="openai/clip-vit-large-patch14")
    # openai/clip-vit-base-patch32 google/siglip-so400m-patch14-384
    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--sample_fps", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.encoder_name is None:
        args.encoder_name = args.encoder_model_name.rsplit("/", 1)[-1]
    return args


def sample_frame_indices(vr, sample_fps):
    video_fps = vr.get_avg_fps()
    duration = len(vr) / video_fps
    timestamps = np.arange(0, duration, 1.0 / sample_fps)
    return np.minimum((timestamps * video_fps).astype(np.int64), len(vr) - 1)


def encode_video(video_path, model, processor, args):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=0)
    frame_idx = sample_frame_indices(vr, args.sample_fps)
    embeddings = []

    for start in range(0, len(frame_idx), args.batch_size):
        batch_idx = frame_idx[start:start + args.batch_size].tolist()

        try:
            frames = vr.get_batch(batch_idx).asnumpy()
        except Exception:
            cap, frames = cv2.VideoCapture(video_path), []
            try:
                for idx in batch_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"OpenCV failed to read frame {idx} from {video_path}")
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                cap.release()
            frames = np.stack(frames)
        
        inputs = processor(images=list(frames), return_tensors="pt").to(args.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        embeddings.append(image_features.cpu().numpy().astype(np.float16))

    return np.concatenate(embeddings, axis=0), frame_idx

def main():
    args = parse_args()
    cache_dir = Path(args.cache_root) / args.dataset_name / args.encoder_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    task = get_task_dict([args.dataset_name])[args.dataset_name]
    video_paths = list(dict.fromkeys(task.doc_to_visual(doc)[0] for doc in task.dataset[task.config.test_split]))
    video_paths = video_paths[args.shard_id::args.num_shards]

    model = AutoModel.from_pretrained(
        args.encoder_model_name,
        torch_dtype=torch.float16,
        device_map=args.device,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.encoder_model_name)

    for video_path in tqdm(video_paths, desc=f"{args.dataset_name}-{args.encoder_name}-shard{args.shard_id}", dynamic_ncols=True):
        cache_path = cache_dir / f"{Path(video_path).stem}.npz"
        if cache_path.exists() and not args.overwrite:
            continue

        embedding, frame_idx = encode_video(video_path, model, processor, args)
        np.savez(cache_path, embedding=embedding, frame_idx=frame_idx)


if __name__ == "__main__":
    main()
