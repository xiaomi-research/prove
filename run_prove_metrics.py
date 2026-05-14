# Copyright 2026, MiLM Plus, Xiaomi Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
import argparse
import pandas as pd

from utils.dataset import DATASET
from utils.media_utils import get_video_frames, get_image, get_media_pairs, resize_frame
from utils.metrics import calculate_rc_s_score, calculate_rc_t_score
from utils.predictors import DinoPredictor


# Model paths
DINO_PATH = "/PATH/TO/YOUR/dinov2-giant"


def calculate_metrics(dir1, dir2, dir3_or_none, metrics, media_type,
                      dino_processor, dino_model,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      out_csv=None,
                      max_items: int = None):
    """
    Calculate metrics for PROVE dataset
    Supports both images and videos
    """
    use_mask = dir3_or_none is not None and dir3_or_none.lower() not in ("none", "")
    media_triplets = get_media_pairs(dir1, dir2, dir3_or_none if use_mask else None, media_type)

    if not media_triplets:
        raise ValueError(f"No matching {media_type} pairs found")

    if max_items is not None and max_items > 0:
        media_triplets = media_triplets[:max_items]
        print(f"[Limit] Processing only first {max_items} {media_type}s")

    # Check for mask-dependent metrics
    mask_dependent = {'rc_s', 'rc_t'}
    missing_mask_metrics = mask_dependent & set(metrics)
    if not use_mask and missing_mask_metrics:
        print(f"[Warning] No mask directory provided, the following metrics will return NaN: {', '.join(sorted(missing_mask_metrics))}")

    # Check if RC-T is requested for image dataset
    if media_type == "image" and "rc_t" in metrics:
        print("[Warning] RC-T metric is only supported for video datasets. Removing rc_t from metrics.")
        metrics = [m for m in metrics if m != "rc_t"]

    results = {k: [] for k in metrics}
    results['time'] = []
    per_item_rows = []

    # Initialize DINO predictor
    dino_predictor = DinoPredictor(dino_model, dino_processor, device=device)

    for ori_path, gen_path, mask_path in tqdm(media_triplets, desc="Calculating metrics"):
        start_time = time.time()
        try:
            if media_type == "video":
                # Process video
                gen_frames = get_video_frames(gen_path)[:81]
                if not gen_frames:
                    print(f"⚠️  Cannot read generated video: {gen_path}")
                    continue
                ori_frames = get_video_frames(ori_path)[:len(gen_frames)]
                if not ori_frames:
                    print(f"⚠️  Cannot read original video: {ori_path}")
                    continue
                if mask_path:
                    mask_frames = get_video_frames(mask_path)[:len(gen_frames)]
                else:
                    mask_frames = [None] * len(gen_frames)

                valid_len = min(len(ori_frames), len(gen_frames), len(mask_frames))
                if valid_len == 0:
                    print(f"⚠️  No valid frames: {gen_path}")
                    continue
                gen_frames = gen_frames[:valid_len]
                mask_frames = mask_frames[:valid_len]

                h, w = gen_frames[0].shape[:2]
                # Resize to uniform size
                if mask_path:
                    # mask_frames = [resize_frame(f, (w, h)) for f in mask_frames]
                    h, w = mask_frames[0].shape[:2]
                    # print(f"h: {h}, w: {w}")
                    gen_frames = [resize_frame(f, (w, h)) for f in gen_frames]

                metric_per_item = {k: [] for k in metrics}

                if 'rc_s' in metrics:
                    for f_gen, f_mask in zip(gen_frames, mask_frames):
                        if use_mask:
                            score = calculate_rc_s_score(f_gen, f_mask, dino_predictor)
                            if score is not None and not np.isnan(score):
                                metric_per_item['rc_s'].append(float(score))

                if 'rc_t' in metrics:
                    if len(gen_frames) > 1:
                        for i in range(len(gen_frames) - 1):
                            score = calculate_rc_t_score(
                                image_t=gen_frames[i],
                                image_t1=gen_frames[i+1],
                                mask_t=mask_frames[i],
                                mask_t1=mask_frames[i+1],
                                predictor=dino_predictor
                            )
                            if score is not None and not np.isnan(score):
                                metric_per_item['rc_t'].append(float(score))
            else:  # image
                # Process image
                gen_img = get_image(gen_path)
                if gen_img is None or gen_img.size == 0:
                    print(f"⚠️  Cannot read generated image: {gen_path}")
                    continue
                if mask_path:
                    mask_img = get_image(mask_path)
                    if mask_img is None or mask_img.size == 0:
                        print(f"⚠️  Cannot read mask image: {mask_path}")
                        continue
                else:
                    mask_img = None

                metric_per_item = {k: [] for k in metrics}

                if 'rc_s' in metrics:
                    if use_mask:
                        score = calculate_rc_s_score(gen_img, mask_img, dino_predictor)
                        if score is not None and not np.isnan(score):
                            metric_per_item['rc_s'].append(float(score))

            end_time = time.time()
            cost_time = end_time - start_time
            results['time'].append(cost_time)

            per_item_summary = {}
            per_item_summary['time'] = cost_time
            for key in metric_per_item:
                if metric_per_item[key]:
                    mv = float(np.mean(metric_per_item[key]))
                    results[key].append(mv)
                    per_item_summary[key] = mv
                else:
                    per_item_summary[key] = np.nan

            case_id = os.path.basename(ori_path)
            per_item_summary['case_id'] = case_id
            per_item_rows.append(per_item_summary)

            out_line = []
            for show_k in ['rc_s', 'rc_t', 'time']:
                if show_k in metrics or show_k == 'time':
                    val = per_item_summary.get(show_k, np.nan)
                    if not np.isnan(val):
                        out_line.append(f"{show_k.upper()}={val:.4f}")
            if out_line:
                tqdm.write(f"{case_id} → " + " | ".join(out_line))

        except Exception as e:
            import traceback
            print(f"[Skip] Error processing: {gen_path} Error: {e}")
            traceback.print_exc()
            continue

    if not any(len(v) > 0 for v in results.values() if v is not None):
        raise ValueError("No metrics calculated successfully")

    avg_results = {k: (float(np.mean(results[k])) if results.get(k) else float('nan')) for k in results}
    avg_results["num"] = len(media_triplets)

    # Output CSV
    if out_csv is None:
        out_csv = os.path.join(os.getcwd(), f"{media_type}_metrics.csv")

    # Avoid time duplication
    metrics_for_col = [m for m in metrics if m != 'time']
    columns = ['case_id'] + metrics_for_col + ['time']

    norm_rows = []
    for row in per_item_rows:
        norm_rows.append({c: row.get(c, np.nan) for c in columns})

    avg_row = {'case_id': 'AVERAGE'}
    for m in metrics:
        avg_row[m] = avg_results.get(m, np.nan)
    avg_row['time'] = avg_results.get('time', np.nan)
    norm_rows.append(avg_row)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    try:
        df = pd.DataFrame(norm_rows, columns=columns)
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"Results saved: {out_csv}")
    except Exception as e:
        print(f"[Warning] pandas save failed ({e}), using csv fallback")
        import csv
        with open(out_csv, "w", newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for r in norm_rows:
                w.writerow(r)
        print(f"Results saved: {out_csv}")

    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["PROVE-M", "PROVE-H", "rord"],
                        help="Evaluation dataset")
    parser.add_argument("--result_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--metrics", type=str, nargs='+',
                        default=["rc_s", "rc_t"],
                        help="Metrics to calculate: rc_s rc_t")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--out_csv", type=str, default="metrics_prove.csv",
                        help="Path to save results CSV")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Custom mask directory path, overrides default mask path in DATASET")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Maximum number of items to process, default processes all")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    dino_processor = AutoImageProcessor.from_pretrained(DINO_PATH)
    dino_model = AutoModel.from_pretrained(DINO_PATH).to(device)

    args.out_csv = os.path.join(args.result_dir, args.out_csv)
    mask_dir = args.mask_dir if args.mask_dir else DATASET[args.dataset]["masks"]

    print("===================================="
          f"Dataset: {args.dataset} \n"
          f"Results directory: {args.result_dir} \n"
          f"Metrics: {args.metrics} \n"
          f"Results saved to: {args.out_csv} \n"
          f"Mask directory: {mask_dir} \n"
          "====================================")

    # Get media type from dataset configuration
    media_type = DATASET[args.dataset].get("type", "video")

    results = calculate_metrics(
        DATASET[args.dataset]["inputs"],
        args.result_dir,
        mask_dir,
        args.metrics,
        media_type,
        dino_processor,
        dino_model,
        device=device,
        out_csv=args.out_csv,
        max_items=args.max_items
    )

    print(f"\n✅ Processed {results['num']} items:")

    # Ensure time is always printed without duplication
    metrics_to_print = list(args.metrics)
    if 'time' not in metrics_to_print:
        metrics_to_print.append('time')

    for metric in metrics_to_print:
        val = results.get(metric, float('nan'))
        print(f"Average {metric.upper()} = {val:.4f}")
    print(f"CSV saved to: {args.out_csv}")