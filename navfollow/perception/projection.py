"""Pixel <-> 3D projection helpers (pinhole camera + TF2 intrinsics/extrinsics).

The chain implemented here is exactly the one used on the robot::

    pixel (u, v) + depth d
        -> optical-frame 3D point (x right, y down, z forward)
        -> TF camera_optical -> base_link   (extrinsic calibration)
        -> distance / bearing in the base frame

and for the evaluation also ``base_link -> map`` so that the follower's Kalman
filter can work in a fixed world frame.
"""
from __future__ import annotations

import math

import numpy as np

from ..camera import CameraIntrinsics
from ..tf_tree import TransformTree, TransformError


def deproject_pixel(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """Pinhole de-projection: pixel + depth -> 3D point in the optical frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z], dtype=float)


def project_point(point_optical: np.ndarray, K: np.ndarray) -> tuple[float, float, float]:
    """3D optical-frame point -> pixel (u, v) and depth."""
    x, y, z = (float(p) for p in point_optical)
    if abs(z) < 1e-9:
        return (0.0, 0.0, 0.0)
    return (K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2], z)


def bbox_depth(
    depth: np.ndarray, bbox: tuple[int, int, int, int], center_ratio: float = 0.5, min_valid_ratio: float = 0.35
) -> float | None:
    """Robust depth of a person bbox: median valid depth in the central patch.

    Sampling the bbox centre region (rather than the exact centre pixel) makes
    the measurement robust to the legs/background gap that a real depth camera
    sees between a person's legs.
    """
    x1, y1, x2, y2 = bbox
    h, w = depth.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    cx1 = int(x1 + bw * (0.5 - center_ratio / 2))
    cx2 = int(x1 + bw * (0.5 + center_ratio / 2))
    cy1 = int(y1 + bh * (0.5 - center_ratio / 2))
    cy2 = int(y1 + bh * (0.5 + center_ratio / 2))
    patch = depth[cy1 : cy2 + 1, cx1 : cx2 + 1].astype(np.float32)
    if patch.size == 0:
        return None
    valid = patch[(patch > 0.1) & (patch < 20.0)]
    if valid.size < max(3, int(min_valid_ratio * patch.size)):
        return None
    return float(np.median(valid))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def base_observation_from_bbox(
    bbox: tuple[int, int, int, int],
    depth: np.ndarray,
    K: np.ndarray,
    T_base_optical: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """bbox + depth -> (distance, bearing, base_xy) in ``base_link``.

    ``base_xy = (forward, left)``; bearing is ``atan2(left, forward)`` so a
    positive bearing means the target is to the left of the robot.
    """
    d = bbox_depth(depth, bbox)
    if d is None:
        return None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    p_opt = deproject_pixel(cx, cy, d, K)
    p_base = transform_points(T_base_optical, p_opt)[0]
    forward, left = float(p_base[0]), float(p_base[1])
    distance = math.hypot(forward, left)
    bearing = math.atan2(left, forward)
    return distance, bearing, np.array([forward, left, float(p_base[2])])


def lookup_base_optical(tf: TransformTree, stamp: float | None = None) -> np.ndarray:
    """T_base_link_camera_optical with explicit TF error propagation."""
    return tf.lookup("base_link", "camera_optical", stamp)


def world_xy_from_base(base_pose_xy: tuple[float, float, float], base_xy: np.ndarray) -> tuple[float, float]:
    """Transform a ``base_link`` xy point into the map frame."""
    x, y, theta = base_pose_xy
    c, s = math.cos(theta), math.sin(theta)
    return (x + c * float(base_xy[0]) - s * float(base_xy[1]), y + s * float(base_xy[0]) + c * float(base_xy[1]))


__all__ = [
    "deproject_pixel",
    "project_point",
    "bbox_depth",
    "transform_points",
    "base_observation_from_bbox",
    "lookup_base_optical",
    "world_xy_from_base",
    "TransformError",
    "CameraIntrinsics",
]
