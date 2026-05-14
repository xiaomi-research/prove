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
import cv2
from PIL import Image
import numpy as np


def get_video_frames(video_path, max_frames=81):
    """
    Extract frames from a video file
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    return frames


def get_image(image_path):
    """
    Load an image file
    """
    img = Image.open(image_path)
    return np.array(img)


def get_media_pairs(dir_ori, dir_gen, dir_mask=None, media_type="video"):
    if media_type == "video":
        ext = [".mp4"]
    else:
        ext = [".jpg", ".jpeg", ".png", ".bmp"]

    def get_files(dir_path):
        files = {}
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if any(f.lower().endswith(e) for e in ext):
                    name_no_ext = os.path.splitext(f)[0]
                    files[name_no_ext] = os.path.join(dir_path, f)
        return files

    ori_map = get_files(dir_ori)
    gen_map = get_files(dir_gen)

    if dir_mask and os.path.isdir(dir_mask):
        mask_map = get_files(dir_mask)
    else:
        mask_map = None

    pairs = []
    for name in sorted(set(ori_map.keys()) & set(gen_map.keys())):
        if mask_map is not None:
            if name not in mask_map:
                print(f"[Skip] Missing mask for: {name}")
                continue
            pairs.append((ori_map[name], gen_map[name], mask_map[name]))
        else:
            pairs.append((ori_map[name], gen_map[name], None))
    return pairs


def resize_frame(frame, target_size):
    """
    Resize frame to target size
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)