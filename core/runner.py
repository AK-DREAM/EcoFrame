import numpy as np
from decord import VideoReader, cpu

from .utils import (
    _extend_candidate_pool,
    _get_encoder,
    _load_encoder_image_cache,
    _parse_attn_layer,
    _select_from_candidate_pool,
    _write_debug,
)


class CToFRunner:
    def __init__(
        self,
        frame_schedule,
        lambda_attn,
        lambda_rele,
        candidate_multiplier,
        encoder_model_name,
        image_cache_dir,
        frame_debug_dir,
        attn_layers,
        early_exit=False,
        entropy_threshold_start=0.1,
        entropy_threshold_step=0.2,
    ):
        self.frame_schedule = frame_schedule
        self.lambda_attn = float(lambda_attn)
        self.lambda_rele = float(lambda_rele)
        self.candidate_multiplier = int(candidate_multiplier)
        self.encoder_model_name = encoder_model_name
        self.image_cache_dir = image_cache_dir
        self.frame_debug_dir = frame_debug_dir
        self.attn_layers = attn_layers
        self.early_exit = early_exit
        self.entropy_threshold_start = float(entropy_threshold_start)
        self.entropy_threshold_step = float(entropy_threshold_step)

    def run(self, adapter, video_path, video_id, query_text, debug_metadata):
        frame_schedule = [int(x) for x in self.frame_schedule]
        attn_layers = _parse_attn_layer(self.attn_layers)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=0)
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        length_multiplier = np.clip(np.sqrt(total_frames / fps / 120), 1, 4)

        encoder_image_cache, encoder_avail_indices = _load_encoder_image_cache(self.image_cache_dir, video_id)
        if encoder_avail_indices is None:
            encoder_avail_indices = np.minimum((np.arange(0, total_frames / fps, 0.25) * fps).astype(np.int64), total_frames - 1)

        encoder = _get_encoder(self, self.encoder_model_name, adapter.device)
        encoder_text_cache = {}
        candidate_pool_indices = []
        selected_indices = None
        previous_frame_scores = None
        rounds = []
        early_exited = False
        for round_idx, target_num in enumerate(frame_schedule):
            profile = {}
            previous_num = 0 if round_idx == 0 else frame_schedule[round_idx - 1]
            candidate_pool_indices = _extend_candidate_pool(
                total_frames,
                candidate_pool_indices,
                selected_indices,
                previous_frame_scores,
                int(self.candidate_multiplier * (target_num - previous_num) * length_multiplier),
                self.lambda_attn,
                encoder_avail_indices,
            )
            selected_indices, candidate_pool_scores, step_profile = _select_from_candidate_pool(
                vr,
                total_frames,
                candidate_pool_indices,
                target_num,
                self.lambda_rele,
                query_text,
                encoder,
                encoder_image_cache,
                encoder_text_cache,
            )
            profile.update(step_profile)

            round_log = {
                "round": int(round_idx),
                "candidate_pool_frame_indices": [int(x) for x in candidate_pool_indices],
                "candidate_pool_scores": candidate_pool_scores,
                "selected_frame_indices": [int(x) for x in selected_indices],
            }
            if round_idx == len(frame_schedule) - 1:
                round_log["profile"] = profile
                rounds.append(round_log)
                break

            coarse_info, step_profile = adapter.run_coarse(vr, selected_indices, attn_layers)
            profile.update(step_profile)
            round_log["profile"] = profile
            previous_frame_scores = coarse_info["frame_significance"]
            coarse_info_for_log = dict(coarse_info)
            coarse_info_for_log["frame_significance"] = [round(float(x), 2) for x in coarse_info["frame_significance"]]
            round_log.update(coarse_info_for_log)
            rounds.append(round_log)

            if self.early_exit:
                entropy_threshold = self.entropy_threshold_start + round_idx * self.entropy_threshold_step
                round_log["entropy_threshold"] = entropy_threshold
                if coarse_info["coarse_option_entropy"] < entropy_threshold:
                    early_exited = True
                    break

        final_indices = selected_indices
        if early_exited:
            outputs = coarse_info["coarse_answer"]
            final_inference_ms = 0.0
        else:
            outputs, final_inference_ms = adapter.run_final(vr, final_indices)

        debug_payload = {
            "total_frames": int(total_frames),
            "frame_schedule": [int(x) for x in frame_schedule],
            "final_frame_indices": [int(idx) for idx in final_indices],
            "final_inference_ms": final_inference_ms,
            "early_exit": self.early_exit,
            "early_exited": early_exited,
            "exit_round": int(round_idx),
            "rounds": rounds,
            "attention_layers_used": None if attn_layers is None else [int(layer) for layer in attn_layers],
        }
        debug_payload.update(debug_metadata)
        _write_debug(self.frame_debug_dir, debug_payload)

        return outputs
