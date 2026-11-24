import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CLIP image encode latency on VideoMME frames.")
    parser.add_argument("--dataset_name", type=str, default="videomme")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--encoder_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--encoder_name", type=str, default=None)
    # google/siglip-so400m-patch14-384 openai/clip-vit-base-patch32
    parser.add_argument("--sample_fps", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_videos", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--duration_only", action="store_true")
    args = parser.parse_args()
    if args.encoder_name is None:
        args.encoder_name = args.encoder_model_name.rsplit("/", 1)[-1]
    return args


def load_video_ids(args):
    if args.dataset_name == "videomme":
        parquet_path = Path(args.dataset_path) / "videomme.parquet"
        return pd.read_parquet(parquet_path)["videoID"].drop_duplicates().tolist()
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def get_video_path(args, video_id):
    if args.dataset_name == "videomme":
        return Path(args.dataset_path) / "data" / f"{video_id}.mp4"
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def sample_frame_indices(vr, sample_fps):
    video_fps = vr.get_avg_fps()
    duration = len(vr) / video_fps
    timestamps = np.arange(0, duration, 1.0 / sample_fps)
    return np.minimum((timestamps * video_fps).astype(np.int64), len(vr) - 1)


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def benchmark_video(video_path, model, processor, args):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=0)
    frame_idx = sample_frame_indices(vr, args.sample_fps)
    latencies_ms = []

    full_batches = len(frame_idx) // args.batch_size
    for batch_id in range(full_batches):
        start = batch_id * args.batch_size
        batch_idx = frame_idx[start:start + args.batch_size].tolist()
        frames = vr.get_batch(batch_idx).asnumpy()
        inputs = processor(images=list(frames), return_tensors="pt").to(args.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()

        sync(args.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.get_image_features(**inputs)
        sync(args.device)

        latencies_ms.append((time.perf_counter() - t0) * 1000)

    return latencies_ms


def compute_average_video_duration(args):
    video_ids = load_video_ids(args)
    durations = []
    for video_id in tqdm(video_ids, desc=f"{args.dataset_name}-durations", dynamic_ncols=True):
        vr = VideoReader(str(get_video_path(args, video_id)), ctx=cpu(0), num_threads=0)
        durations.append(len(vr) / vr.get_avg_fps())

    durations = np.array(durations, dtype=np.float64)
    print(f"videos: {len(durations)}")
    print(f"mean_duration_sec: {durations.mean():.3f}")
    print(f"median_duration_sec: {np.median(durations):.3f}")
    print(f"p90_duration_sec: {np.percentile(durations, 90):.3f}")
    print(f"p95_duration_sec: {np.percentile(durations, 95):.3f}")


def main():
    args = parse_args()
    if args.duration_only:
        compute_average_video_duration(args)
        return

    model = AutoModel.from_pretrained(
        args.encoder_model_name,
        torch_dtype=torch.float16,
        device_map={"": args.device},
    ).eval()
    processor = AutoProcessor.from_pretrained(args.encoder_model_name)

    video_ids = load_video_ids(args)[:args.max_videos]
    all_latencies = []

    for video_id in tqdm(video_ids, desc=f"{args.dataset_name}-{args.encoder_name}-encode-latency", dynamic_ncols=True):
        latencies = benchmark_video(get_video_path(args, video_id), model, processor, args)
        all_latencies.extend(latencies)

    latencies = np.array(all_latencies, dtype=np.float64)
    print(f"videos: {len(video_ids)}")
    print(f"full_batches: {len(latencies)}")
    print(f"batch_size: {args.batch_size}")
    print(f"mean_ms: {latencies.mean():.3f}")
    print(f"median_ms: {np.median(latencies):.3f}")
    print(f"p90_ms: {np.percentile(latencies, 90):.3f}")
    print(f"p95_ms: {np.percentile(latencies, 95):.3f}")


if __name__ == "__main__":
    main()
