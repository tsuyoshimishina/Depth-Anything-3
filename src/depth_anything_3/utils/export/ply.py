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

# NOTE: The GLB files exported by trimesh lack the bufferView.target property,
# which seems to cause MeshLab to freeze when attempting to load them.
# This PLY export module was added as an alternative that is fully compatible with MeshLab.

"""PLY point cloud export using Open3D for MeshLab compatibility."""

from __future__ import annotations

import os

import numpy as np
import open3d as o3d
import trimesh

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .depth_vis import export_to_depth_vis
from .glb import (
    _compute_alignment_transform_first_cam_glTF_center_by_points,
    _depths_to_world_points_with_colors,
    _filter_and_downsample,
    get_conf_thresh,
    set_sky_depth,
)


def export_to_ply(
    prediction: Prediction,
    export_dir: str,
    num_max_points: int = 1_000_000,
    conf_thresh: float = 1.05,
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
    conf_thresh_percentile: float = 40.0,
    ensure_thresh_percentile: float = 90.0,
    sky_depth_def: float = 98.0,
    export_depth_vis: bool = True,
) -> str:
    """Generate a 3D point cloud and export it as a ``.ply`` file using Open3D.

    This function builds a point cloud from the predicted depth maps, aligns it to the
    first camera in glTF coordinates (X-right, Y-up, Z-backward), and writes the result
    to ``scene.ply``. The PLY format is widely supported by tools like MeshLab.

    Args:
        prediction: Model prediction containing depth, confidence, intrinsics, extrinsics,
            and pre-processed images.
        export_dir: Output directory where the PLY file will be written.
        num_max_points: Maximum number of points retained after downsampling.
        conf_thresh: Base confidence threshold used before percentile adjustments.
        filter_black_bg: Mark near-black background pixels for removal during confidence filtering.
        filter_white_bg: Mark near-white background pixels for removal during confidence filtering.
        conf_thresh_percentile: Lower percentile used when adapting the confidence threshold.
        ensure_thresh_percentile: Upper percentile clamp for the adaptive threshold.
        sky_depth_def: Percentile used to fill sky pixels with plausible depth values.
        export_depth_vis: Whether to export raster depth visualizations alongside the PLY.

    Returns:
        Path to the exported ``scene.ply`` file.
    """
    # Validate required prediction fields
    assert (
        prediction.processed_images is not None
    ), "Export to PLY: prediction.processed_images is required but not available"
    assert (
        prediction.depth is not None
    ), "Export to PLY: prediction.depth is required but not available"
    assert (
        prediction.intrinsics is not None
    ), "Export to PLY: prediction.intrinsics is required but not available"
    assert (
        prediction.extrinsics is not None
    ), "Export to PLY: prediction.extrinsics is required but not available"
    assert (
        prediction.conf is not None
    ), "Export to PLY: prediction.conf is required but not available"

    logger.info(f"conf_thresh_percentile: {conf_thresh_percentile}")
    logger.info(f"num max points: {num_max_points}")
    logger.info(f"Exporting to PLY with num_max_points: {num_max_points}")

    images_u8 = prediction.processed_images  # (N,H,W,3) uint8

    # Sky processing (if sky_mask is provided)
    if getattr(prediction, "sky_mask", None) is not None:
        set_sky_depth(prediction, prediction.sky_mask, sky_depth_def)

    # Confidence threshold filtering
    if filter_black_bg:
        prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
    if filter_white_bg:
        prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0
    conf_thr = get_conf_thresh(
        prediction,
        getattr(prediction, "sky_mask", None),
        conf_thresh,
        conf_thresh_percentile,
        ensure_thresh_percentile,
    )

    # Back-project to world coordinates and get colors
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,
        images_u8,
        prediction.conf,
        conf_thr,
    )

    # Compute alignment transform (same as GLB for consistency)
    A = _compute_alignment_transform_first_cam_glTF_center_by_points(
        prediction.extrinsics[0], points
    )

    if points.shape[0] > 0:
        points = trimesh.transform_points(points, A)

    # Clean and downsample
    points, colors = _filter_and_downsample(points, colors, num_max_points)

    # Export using Open3D
    os.makedirs(export_dir, exist_ok=True)
    out_path = os.path.join(export_dir, "scene.ply")

    if points.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(out_path, pcd)
        logger.info(f"Exported PLY with {points.shape[0]} points to {out_path}")
    else:
        # Create empty PLY file if no points
        pcd = o3d.geometry.PointCloud()
        o3d.io.write_point_cloud(out_path, pcd)
        logger.warning("No valid points to export, created empty PLY file")

    if export_depth_vis:
        export_to_depth_vis(prediction, export_dir)

    return out_path
