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
import torch
import cv2
import math


class DinoPredictor:
    """
    DINOv2 predictor for feature extraction and metric calculation
    """
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        target_size: int = 448,
    ) -> None:
        self.device = device
        self.processor = processor
        self.model = model
        self.target_size = target_size
        self.features = None
        self.input_size = None
        self.original_size = None
        self.patch_size = 14  # DINOv2
        self._feat_hw = None  # (H_feat, W_feat)
        self._resize_hw = None
        self._pad_hw = None

    @torch.no_grad()
    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Set image and extract features
        """
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != "RGB":
            image = image[..., ::-1]

        orig_H, orig_W = image.shape[:2]
        self.original_size = (orig_H, orig_W)

        # 1. Resize proportionally to target_size
        scale = self.target_size / max(orig_H, orig_W)
        new_H, new_W = int(orig_H * scale), int(orig_W * scale)
        self._resize_hw = (new_H, new_W)
        image_resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        # 2. Calculate padding to make dimensions divisible by 14
        pad_h = (self.patch_size - (new_H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (new_W % self.patch_size)) % self.patch_size
        self._pad_hw = (pad_h, pad_w)

        # Pad with RGB mean
        image_padded = cv2.copyMakeBorder(
            image_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(123, 116, 103)
        )

        # 3. Process image without resizing or cropping
        inputs = self.processor(
            images=image_padded,
            return_tensors="pt",
            do_resize=False,      # Disable built-in resize
            do_center_crop=False  # Disable built-in crop
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.input_size = tuple(inputs["pixel_values"].shape[-2:])

        outputs = self.model(**inputs)
        self.features = outputs.last_hidden_state  # (1, 1+L, C)

        # 4. Calculate feature map dimensions
        H_feat = (new_H + pad_h) // self.patch_size
        W_feat = (new_W + pad_w) // self.patch_size

        L = int(self.features.shape[1] - 1)
        if H_feat * W_feat != L:
            raise ValueError(f"Token count mismatch: H_feat*W_feat={H_feat*W_feat}, L={L}")

        self._feat_hw = (H_feat, W_feat)
        self.is_image_set = True

    @torch.no_grad()
    def get_convolutional_metrics(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        kernel_size: int = 32,
        stride: int = 16,
        tau_mmd: float = 0.3,
        image_format: str = "RGB"
    ):
        """
        Calculate convolutional metrics using sliding window
        """
        self.set_image(image, image_format)
        # Remove CLS token
        patch_features = self.features[:, 1:, :]
        B, L_tokens, C = patch_features.shape

        H_feat, W_feat = self._feat_hw

        # Reshape to (B, C, H, W)
        features_spatial = patch_features.permute(0, 2, 1).view(B, C, H_feat, W_feat)

        # Calculate scale factor based on longest side
        scale_factor = max(H_feat, W_feat) / 64.0

        adj_kernel_size = max(1, int(kernel_size * scale_factor))
        adj_stride = max(1, int(stride * scale_factor))

        # Process mask
        new_H, new_W = self._resize_hw
        pad_h, pad_w = self._pad_hw

        # 1. Resize mask with nearest neighbor
        mask_resized = cv2.resize(mask.astype(np.float32), (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        # 2. Pad with 0 (background)
        mask_padded = cv2.copyMakeBorder(
            mask_resized, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )

        mask_tensor = torch.as_tensor(mask_padded, device=self.device).float()
        if mask_tensor.ndim == 2: mask_tensor = mask_tensor[None, None, ...]
        elif mask_tensor.ndim == 3: mask_tensor = mask_tensor[None, ...]

        # 3. Downsample to feature map size using max pooling
        mask_small = torch.nn.functional.adaptive_max_pool2d(mask_tensor, output_size=(H_feat, W_feat))

        features_flat_global = features_spatial.view(C, -1).permute(1, 0)  # (N, C)
        global_bg_feats = features_flat_global[(mask_small.view(-1) == 0)]

        # Unfold sliding window
        feat_patches = torch.nn.functional.unfold(features_spatial, kernel_size=adj_kernel_size, stride=adj_stride)
        mask_patches = torch.nn.functional.unfold(mask_small, kernel_size=adj_kernel_size, stride=adj_stride)

        L_patches = feat_patches.shape[-1]
        K_sq = adj_kernel_size * adj_kernel_size

        feat_flat = feat_patches.view(1, C, K_sq, L_patches).permute(3, 1, 2, 0).squeeze(-1)
        mask_flat = mask_patches.view(1, 1, K_sq, L_patches).permute(3, 1, 2, 0).squeeze(-1).squeeze(1)

        valid_scores = []

        for i in range(L_patches):
            curr_mask = mask_flat[i]
            n_fg = (curr_mask > 0).sum()
            n_bg = (curr_mask == 0).sum()
            if n_fg > 5 and n_bg > 5:  # mixed (boundary patch)
                curr_feat = feat_flat[i]
                X_in = curr_feat[:, curr_mask > 0].T
                X_out = curr_feat[:, curr_mask == 0].T

                score = self.compute_rbf_mmd(X_in, X_out)
                valid_scores.append(score)

            elif n_fg > 5 and n_bg == 0:  # all-foreground (inner patch)
                curr_feat = feat_flat[i]
                X_in = curr_feat[:, curr_mask > 0].T
                X_out = global_bg_feats

                if X_out.shape[0] > 0: # Ensure there is background in the whole image
                    score = self.compute_rbf_mmd(X_in, X_out)
                    valid_scores.append(score)

        if not valid_scores:
            return {"g_mmd": float("nan"), "raw_mmd": float("nan")}

        avg_raw_mmd = sum(valid_scores) / len(valid_scores)
        safe_raw_mmd = max(0.0, avg_raw_mmd)
        g_mmd = math.exp(-safe_raw_mmd / max(1e-12, tau_mmd))

        return {
            "g_mmd": g_mmd,
            "raw_mmd": safe_raw_mmd,
        }

    @torch.no_grad()
    def compute_rts_on_patch_mask_only_conv(
        self,
        crop_img_t: np.ndarray,
        crop_img_t1: np.ndarray,
        crop_mask_t: np.ndarray,
        crop_mask_t1: np.ndarray,
        kernel_size: int = 32,
        stride: int = 16,
        min_points: int = 5,
        image_format: str = "RGB",
        epsilon: float = 1e-6,
    ):
        """
        Calculate RTS score using sliding window on mask intersection area
        """
        # 1. Extract features for frame t and get spatial parameters
        self.set_image(crop_img_t, image_format)

        # Remove CLS token and reshape
        patch_features_t = self.features[:, 1:, :]
        B, L_tokens, C = patch_features_t.shape
        H_feat, W_feat = self._feat_hw

        # Get padding and resize parameters for mask processing
        new_H, new_W = self._resize_hw
        pad_h, pad_w = self._pad_hw

        # Convert to (B, C, H, W)
        feat_spatial_t = patch_features_t.permute(0, 2, 1).view(B, C, H_feat, W_feat)

        # 2. Extract features for frame t1
        self.set_image(crop_img_t1, image_format)
        patch_features_t1 = self.features[:, 1:, :]
        feat_spatial_t1 = patch_features_t1.permute(0, 2, 1).view(B, C, H_feat, W_feat)

        # 3. Dynamically adjust sliding window parameters
        scale_factor = max(H_feat, W_feat) / 64.0
        adj_kernel_size = max(1, int(kernel_size * scale_factor))
        adj_stride = max(1, int(stride * scale_factor))

        # 4. Process masks
        def _process_mask(mask_arr):
            # 1. Resize with nearest neighbor
            m_resized = cv2.resize(mask_arr.astype(np.float32), (new_W, new_H), interpolation=cv2.INTER_NEAREST)

            # 2. Pad with 0 (background)
            m_padded = cv2.copyMakeBorder(
                m_resized, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=0
            )

            # 3. Convert to tensor
            m_tensor = torch.as_tensor(m_padded, device=self.device).float()
            if m_tensor.ndim == 2: m_tensor = m_tensor[None, None, ...]
            elif m_tensor.ndim == 3: m_tensor = m_tensor[None, ...]

            # 4. Downsample to feature map size using max pooling
            m_small = torch.nn.functional.adaptive_max_pool2d(m_tensor, output_size=(H_feat, W_feat))
            return m_small

        # Get processed masks and compare intersection
        m_t_small = _process_mask(crop_mask_t)
        m_t1_small = _process_mask(crop_mask_t1)
        mask_in = (m_t_small > 0.5) & (m_t1_small > 0.5)
        mask_in = mask_in.float()  # unfold requires numeric type

        # 5. Unfold sliding window to extract patches
        feat_patches_t = torch.nn.functional.unfold(feat_spatial_t, kernel_size=adj_kernel_size, stride=adj_stride)
        feat_patches_t1 = torch.nn.functional.unfold(feat_spatial_t1, kernel_size=adj_kernel_size, stride=adj_stride)
        mask_patches = torch.nn.functional.unfold(mask_in, kernel_size=adj_kernel_size, stride=adj_stride)

        L_patches = feat_patches_t.shape[-1]
        K_sq = adj_kernel_size * adj_kernel_size

        feat_flat_t = feat_patches_t.view(1, C, K_sq, L_patches).permute(3, 1, 2, 0).squeeze(-1)   # (L_patches, C, K_sq)
        feat_flat_t1 = feat_patches_t1.view(1, C, K_sq, L_patches).permute(3, 1, 2, 0).squeeze(-1) # (L_patches, C, K_sq)
        mask_flat = mask_patches.view(1, 1, K_sq, L_patches).permute(3, 1, 2, 0).squeeze(-1).squeeze(1) # (L_patches, K_sq)

        # 6. Calculate MMD for each patch
        valid_scores = []
        for i in range(L_patches):
            curr_mask = mask_flat[i]
            fg_idx = curr_mask > 0.5
            if fg_idx.sum().item() < min_points:
                continue

            X_t = feat_flat_t[i][:, fg_idx].T  # (n_fg, C)
            X_t1 = feat_flat_t1[i][:, fg_idx].T
            if X_t.shape[0] < min_points:
                continue

            score = self.compute_rbf_mmd(X_t, X_t1)
            valid_scores.append(score.item())

        if not valid_scores:
            return float('nan')

        avg_raw_mmd = sum(valid_scores) / len(valid_scores)

        return avg_raw_mmd

    def compute_rbf_mmd(self, X, Y, sigma_list=[10]):
        """
        Calculate RBF MMD between feature sets X (foreground) and Y (background)
        """
        # Normalize features
        X = torch.nn.functional.normalize(X, p=2, dim=1)
        Y = torch.nn.functional.normalize(Y, p=2, dim=1)

        XX = torch.cdist(X, X).pow(2)
        YY = torch.cdist(Y, Y).pow(2)
        XY = torch.cdist(X, Y).pow(2)

        mmd_val = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K_XX = torch.exp(-gamma * XX).mean()
            K_YY = torch.exp(-gamma * YY).mean()
            K_XY = torch.exp(-gamma * XY).mean()
            mmd_val += K_XX + K_YY - 2 * K_XY

        return mmd_val / len(sigma_list) * 1000