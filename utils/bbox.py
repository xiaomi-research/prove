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
from PIL import Image


def find_bounding_square_1_3(binary_image):
    """
    Find bounding square with 1/3 expansion
    """
    expansion_ratio = 1/3
    if isinstance(binary_image, np.ndarray):
        mask = Image.fromarray(binary_image).convert("L")
    else:
        mask = binary_image.convert("L")

    bbox = mask.getbbox()
    if bbox is None:
        return None

    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper

    # Center-aligned smallest square
    size = max(width, height)
    cx = (left + right) / 2.0
    cy = (upper + lower) / 2.0
    start_x = int(round(cx - size / 2))
    start_y = int(round(cy - size / 2))

    # Target expansion 1/3, bounded by image edges
    img_w, img_h = mask.size
    dist_left = start_x
    dist_top = start_y
    dist_right = img_w - (start_x + size)
    dist_bottom = img_h - (start_y + size)

    sub_target = int(round(size * expansion_ratio))
    sub = min(sub_target, dist_left, dist_top, dist_right, dist_bottom, size // 2)
    sub = max(sub, 0)

    top_left_x = start_x - sub
    top_left_y = start_y - sub
    final_size = size + 2 * sub

    max_size = min(img_w, img_h)
    if final_size > max_size:
        final_size = max_size

    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    top_left_x = min(top_left_x, img_w - final_size)
    top_left_y = min(top_left_y, img_h - final_size)
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)

    final_right = top_left_x + final_size
    final_lower = top_left_y + final_size
    return int(top_left_x), int(top_left_y), int(final_right), int(final_lower)