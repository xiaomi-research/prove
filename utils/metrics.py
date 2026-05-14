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

import numpy as np
from skimage.measure import label, regionprops
from PIL import Image
import cv2

from utils.bbox import find_bounding_square_1_3


def calculate_rc_s_score(image: np.ndarray, mask: np.ndarray, predictor):
    """
    Calculate RC-S score (based on calculate_mmd_score_patch_all)
    """
    gray_mask = np.array(Image.fromarray(mask))
    temp_mask = gray_mask
    if temp_mask.ndim == 3:
        if temp_mask.shape[2] == 1:
            temp_mask = temp_mask[:, :, 0]
        else:
            temp_mask = np.array(Image.fromarray(temp_mask).convert("L"))

    binary_image_global = (temp_mask > 127).astype(np.uint8) * 255

    lab = label(binary_image_global, connectivity=1)
    regions = regionprops(lab)
    if not regions:
        return None

    min_area = 200
    max_area = max(r.area for r in regions)
    if max_area < min_area:
        min_area = max_area

    comps_data = []
    for r in regions:
        if r.area < min_area:
            continue
        comp_mask = (lab == r.label).astype(np.uint8) * 255
        square_info = find_bounding_square_1_3(comp_mask)
        if square_info:
            x1, y1, x2, y2 = square_info
            comps_data.append((x1, y1, x2, y2))

    if not comps_data:
        return None

    total_score = 0.0
    total_weight = 0.0

    for (x1, y1, x2, y2) in comps_data:
        input_img = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        mask_fg = np.array(Image.fromarray(crop_mask).convert("L"))
        metrics = predictor.get_convolutional_metrics(
            input_img,
            mask_fg,
            kernel_size=16,
            stride=8,
            tau_mmd=3
        )
        total_score += metrics["g_mmd"]
        total_weight += 1

    if total_weight == 0:
        return None
    final_score = total_score / total_weight

    return final_score


def calculate_rc_t_score(image_t: np.ndarray, image_t1: np.ndarray,
                        mask_t: np.ndarray, mask_t1: np.ndarray,
                        predictor):
    """
    Calculate RC-T score (based on calculate_rts_score_all)
    """
    def _to_u8_gray(m: np.ndarray) -> np.ndarray:
        # Mask may be HxW or HxWxC, take single channel and keep uint8
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(np.uint8, copy=False)

    def _bin(m: np.ndarray) -> np.ndarray:
        m = _to_u8_gray(m)
        return (m > 127).astype(np.uint8) * 255

    # [Key] Take union to determine crop box
    bin_t = _bin(mask_t)
    bin_t1 = _bin(mask_t1)
    union_mask = cv2.bitwise_or(bin_t, bin_t1)

    # Find bounding box
    rect = find_bounding_square_1_3(union_mask)
    if rect is None:
        return None

    x1, y1, x2, y2 = rect

    # Unified cropping
    crop_img_t = image_t[y1:y2, x1:x2]
    crop_img_t1 = image_t1[y1:y2, x1:x2]

    # Crop masks (use binary mask directly)
    crop_mask_t = bin_t[y1:y2, x1:x2]
    crop_mask_t1 = bin_t1[y1:y2, x1:x2]

    # Return None if cropped area is empty
    if crop_img_t.size == 0 or crop_img_t1.size == 0:
        return None

    # Calculate score (only mask intersection area)
    rts_val = predictor.compute_rts_on_patch_mask_only_conv(
        crop_img_t, crop_img_t1,
        crop_mask_t, crop_mask_t1,
        kernel_size=16,
        stride=8,
        image_format="RGB"
    )

    return rts_val