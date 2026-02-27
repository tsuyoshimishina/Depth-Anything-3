# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger


def export_to_volsplat(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str] | None = None,
    scene_key: str | None = None,
) -> None:
    """Export DA3 prediction to VolSplat RE10k .torch format.

    Produces a directory structure compatible with VolSplat's DatasetRE10k loader:
        {export_dir}/volsplat/test/000000.torch
        {export_dir}/volsplat/test/index.json

    Args:
        prediction: DA3 Prediction object with extrinsics, intrinsics, and images.
        export_dir: Root directory for export output.
        image_paths: Optional list of original image file paths. When provided,
            raw file bytes are used for higher quality. Otherwise falls back to
            JPEG-encoding prediction.processed_images.
        scene_key: Optional scene identifier. Defaults to the stem of the first
            image path, or "scene_0000".
    """
    assert prediction.extrinsics is not None, "Extrinsics required for VolSplat export"
    assert prediction.intrinsics is not None, "Intrinsics required for VolSplat export"
    assert prediction.processed_images is not None, "Processed images required for VolSplat export"

    N = prediction.processed_images.shape[0]
    H, W = prediction.processed_images.shape[1:3]

    # --- Scene key -----------------------------------------------------------
    if scene_key is None:
        if image_paths is not None and len(image_paths) > 0:
            scene_key = Path(image_paths[0]).stem
        else:
            scene_key = "scene_0000"

    # --- Validate image resolution --------------------------------------------
    VALID_SHAPES = {(360, 640), (720, 1280)}
    if image_paths is not None and len(image_paths) == N:
        probe = Image.open(image_paths[0])
        img_w, img_h = probe.size
    else:
        img_h, img_w = H, W
    if (img_h, img_w) not in VALID_SHAPES:
        valid_str = ", ".join(f"{h}x{w}" for h, w in sorted(VALID_SHAPES))
        raise ValueError(
            f"VolSplat requires images sized {valid_str}, "
            f"but got {img_h}x{img_w}. "
            f"Resize your input images or set dataset.skip_bad_shape=false on the VolSplat side."
        )

    # --- Images → JPEG bytes as uint8 tensors --------------------------------
    jpeg_images: list[torch.Tensor] = []
    if image_paths is not None and len(image_paths) == N:
        for path in image_paths:
            with open(path, "rb") as f:
                raw_bytes = f.read()
            jpeg_images.append(torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8))
    else:
        for i in range(N):
            img = Image.fromarray(prediction.processed_images[i])
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            jpeg_images.append(
                torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)
            )

    # --- Camera vector (N, 18) -----------------------------------------------
    cameras = torch.zeros(N, 18, dtype=torch.float32)
    intrinsics = prediction.intrinsics  # (N, 3, 3) pixel-space
    extrinsics = prediction.extrinsics  # (N, 3, 4) or (N, 4, 4) W2C

    for i in range(N):
        fx = intrinsics[i, 0, 0]
        fy = intrinsics[i, 1, 1]
        cx = intrinsics[i, 0, 2]
        cy = intrinsics[i, 1, 2]

        # Normalise by processing resolution
        cameras[i, 0] = fx / W
        cameras[i, 1] = fy / H
        cameras[i, 2] = cx / W
        cameras[i, 3] = cy / H

        # [4:6] unused — leave as 0.0

        # [6:18] W2C 3×4 flattened
        cameras[i, 6:18] = torch.from_numpy(extrinsics[i, :3, :4].flatten())

    # --- Build scene dict -----------------------------------------------------
    scene_dict = {
        "key": scene_key,
        "url": "",
        "timestamps": torch.arange(N, dtype=torch.int64),
        "cameras": cameras,
        "images": jpeg_images,
    }

    # --- Save -----------------------------------------------------------------
    out_dir = os.path.join(export_dir, "volsplat", "test")
    os.makedirs(out_dir, exist_ok=True)

    chunk_filename = "000000.torch"
    torch.save([scene_dict], os.path.join(out_dir, chunk_filename))

    index = {scene_key: chunk_filename}
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)

    logger.info(
        f"VolSplat export done: {N} frames, scene_key={scene_key!r}, "
        f"saved to {out_dir}/{chunk_filename}"
    )
