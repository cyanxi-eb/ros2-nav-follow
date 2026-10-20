"""Synthetic RGB-D camera used by the offline perception pipeline.

The offline simulator does not have Gazebo, so the "depth camera" is a small
2.5-D renderer written with numpy:

* a pinhole camera with the same intrinsics as the real depth camera model,
* the ground plane is ray-cast per pixel (floor tiles / wall cells),
* each pedestrian is a vertical cylinder with a jacket colour, trousers and a
  head band,
* a z-buffer resolves occlusion between pedestrians and the background,
* wall occlusion is handled by a 2-D line-of-sight test against the occupancy
  grid (binary per pedestrian, which is enough for the re-identification test),
* a lens-distortion model is applied to the rendered RGB and depth images so
  that the perception pipeline really has to undistort before it can deproject.

The renderer is deliberately *not* a photorealistic ray tracer; the README and
``docs/evaluation.md`` state that limitation explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Pose2D, make_transform, quaternion_from_yaw, rpy_to_matrix
from .world import SimWorld

# REP-103: camera_link (x forward, y left, z up) -> camera_optical (x right, y down, z forward)
_OPTICAL_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class CameraIntrinsics:
    width: int = 160
    height: int = 120
    fx: float = 140.0
    fy: float = 140.0
    cx: float | None = None
    cy: float | None = None
    # OpenCV radial-tangential distortion model; small values like a cheap
    # wide-angle depth camera
    k1: float = -0.12
    k2: float = 0.025
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = (self.width - 1) / 2.0
        if self.cy is None:
            self.cy = (self.height - 1) / 2.0

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def hfov(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def vfov(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    def scaled(self, factor: float) -> "CameraIntrinsics":
        return CameraIntrinsics(
            width=int(round(self.width * factor)),
            height=int(round(self.height * factor)),
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=None,
            cy=None,
            k1=self.k1,
            k2=self.k2,
            p1=self.p1,
            p2=self.p2,
            k3=self.k3,
        )


@dataclass
class CameraMount:
    """Extrinsic calibration: camera in ``base_link``.

    ``pitch`` is the *downward* tilt in radians (positive = looking down), with
    the camera body frame following the URDF convention (x forward, y left,
    z up).  ``optical_matrix`` additionally applies the REP-103
    ``camera_link -> camera_optical`` rotation.
    """

    x: float = 0.12
    y: float = 0.0
    z: float = 1.05
    pitch: float = 0.14  # downward tilt [rad]
    yaw: float = 0.0
    roll: float = 0.0

    def body_matrix(self) -> np.ndarray:
        """T_base_link_camera_link."""
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(self.roll, self.pitch, self.yaw)
        T[:3, 3] = (self.x, self.y, self.z)
        return T

    def optical_matrix(self) -> np.ndarray:
        """T_base_link_camera_optical (body frame + REP-103 optical rotation)."""
        return self.body_matrix() @ make_transform((0.0, 0.0, 0.0), rotation=_OPTICAL_ROTATION)


class SyntheticCamera:
    """RGB-D renderer for the offline simulator."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        mount: CameraMount | None = None,
        max_range: float = 7.0,
        apply_distortion: bool = True,
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.mount = mount or CameraMount()
        self.max_range = max_range
        self.apply_distortion = apply_distortion
        self._build_rays()
        self._build_distortion_maps()

    # ------------------------------------------------------------------
    def _build_rays(self) -> None:
        K = self.intrinsics
        u = np.arange(K.width, dtype=np.float64)
        v = np.arange(K.height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        x = (uu - K.cx) / K.fx
        y = (vv - K.cy) / K.fy
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3) optical
        norm = np.linalg.norm(dirs, axis=-1)
        self._dirs_optical = dirs
        self._dirs_unit = dirs / norm[..., None]
        self._inv_norm = 1.0 / norm
        self._pixel_u = uu
        self._pixel_v = vv
        self._norm = norm

    def _build_distortion_maps(self) -> None:
        if not self.apply_distortion:
            self._maps = None
            return
        K = self.intrinsics
        H, W = K.height, K.width
        uu, vv = self._pixel_u, self._pixel_v
        distorted = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(distorted, K.K, K.D, P=K.K)
        mx = ideal[..., 0].reshape(H, W).astype(np.float32)
        my = ideal[..., 1].reshape(H, W).astype(np.float32)
        self._maps = (mx, my)

    # ------------------------------------------------------------------
    def world_from_camera(self, robot_pose: Pose2D) -> np.ndarray:
        """T_world_camera_optical for a robot pose."""
        T_world_base = make_transform((robot_pose.x, robot_pose.y, 0.0), quaternion_from_yaw(robot_pose.theta))
        return T_world_base @ self.mount.optical_matrix()

    def project(self, points_world: np.ndarray, robot_pose: Pose2D) -> np.ndarray:
        """Project (N,3) world points into pixels.  Returns (N,2) ``(u, v)`` plus depth."""
        T = np.linalg.inv(self.world_from_camera(robot_pose))
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        cam = pts @ T[:3, :3].T + T[:3, 3]
        depth = cam[:, 2]
        u = self.intrinsics.fx * cam[:, 0] / depth + self.intrinsics.cx
        v = self.intrinsics.fy * cam[:, 1] / depth + self.intrinsics.cy
        return np.stack([u, v, depth], axis=1)

    # ------------------------------------------------------------------
    def render(self, world: SimWorld, robot_pose: Pose2D) -> tuple[np.ndarray, np.ndarray]:
        """Render one RGB-D frame.

        Returns
        -------
        rgb : uint8 (H, W, 3)
        depth : float32 (H, W), metres along the optical axis, 0 = invalid
        """
        K = self.intrinsics
        H, W = K.height, K.width
        T_world_cam = self.world_from_camera(robot_pose)
        R = T_world_cam[:3, :3]
        cam_pos = T_world_cam[:3, 3]

        dirs = self._dirs_unit.reshape(-1, 3) @ R.T  # (N,3) world directions
        dz = dirs[:, 2]
        dx, dy = dirs[:, 0], dirs[:, 1]

        rgb = np.empty((H * W, 3), dtype=np.float64)
        depth = np.zeros(H * W, dtype=np.float64)

        # ---- background / above horizon --------------------------------
        rgb[:] = np.array([46.0, 48.0, 54.0])

        # ---- ground plane ----------------------------------------------
        floor_mask = dz < -1e-4
        t_raw = np.full(H * W, np.inf)
        t_raw[floor_mask] = -cam_pos[2] / dz[floor_mask]
        valid_range = np.isfinite(t_raw) & (t_raw > 0.25) & (t_raw < self.max_range)
        t_safe = np.where(valid_range, t_raw, 0.0)
        px = cam_pos[0] + dx * t_safe
        py = cam_pos[1] + dy * t_safe
        res = world.grid.resolution
        cols = np.floor((px - world.grid.origin[0]) / res).astype(np.int64)
        rows = np.floor((py - world.grid.origin[1]) / res).astype(np.int64)
        in_bounds = (cols >= 0) & (cols < world.grid.width) & (rows >= 0) & (rows < world.grid.height)
        valid = valid_range & in_bounds
        idx = np.nonzero(valid)[0]
        occ = np.zeros(H * W, dtype=bool)
        if idx.size:
            occ[idx] = world.grid.occupancy[rows[idx], cols[idx]]
        t_floor = np.where(valid, t_raw, 0.0)

        # floor tiles + distance shading
        tile = ((np.floor(px / 0.7) + np.floor(py / 0.7)) % 2.0).astype(np.int64)
        base = np.where(tile == 0, 172.0, 158.0)
        shade = 1.0 / (1.0 + 0.10 * np.clip(t_floor, 0.0, None))
        shade = np.clip(shade, 0.45, 1.0)
        floor_rgb = np.stack([base * 1.00, base * 0.99, base * 0.95], axis=1) * shade[:, None]
        wall_rgb = np.array([78.0, 80.0, 86.0])[None, :] * shade[:, None]
        color = np.where(occ[:, None], wall_rgb, floor_rgb)
        rgb[idx] = color[idx]
        depth[idx] = (t_floor[idx] * self._inv_norm.reshape(-1)[idx]).astype(np.float64)

        # ---- pedestrians (z-buffered cylinders) ------------------------
        for person in world.persons:
            if person.hidden:
                continue
            if not world.person_visible(person, robot_pose, half_fov=self.intrinsics.hfov * 0.5):
                continue
            self._render_person(world, person, cam_pos, dirs, rgb, depth)

        rgb_u8 = np.clip(rgb.reshape(H, W, 3), 0, 255).astype(np.uint8)
        depth_f = depth.reshape(H, W).astype(np.float32)
        if self._maps is not None:
            mx, my = self._maps
            rgb_u8 = cv2.remap(rgb_u8, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            depth_f = cv2.remap(
                depth_f, mx, my, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
        return rgb_u8, depth_f

    # ------------------------------------------------------------------
    def _render_person(
        self,
        world: SimWorld,
        person,
        cam_pos: np.ndarray,
        dirs: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        H, W = self.intrinsics.height, self.intrinsics.width
        o = cam_pos[:2]
        p = np.array([person.x, person.y], dtype=np.float64)
        r = person.radius
        d2 = dirs[:, :2]
        a = np.einsum("ij,ij->i", d2, d2)
        oc = o[None, :] - p[None, :]
        b = 2.0 * (d2 @ oc[0])
        c = float(oc[0] @ oc[0]) - r * r
        disc = b * b - 4.0 * a * c
        hit = disc > 0.0
        if not hit.any():
            return
        t = np.full(dirs.shape[0], np.inf)
        sq = np.sqrt(disc[hit])
        ta = (-b[hit] - sq) / (2.0 * a[hit])
        tb = (-b[hit] + sq) / (2.0 * a[hit])
        t[hit] = np.where(ta > 0.2, ta, tb)

        z_hit = cam_pos[2] + t * dirs[:, 2]
        ok = np.isfinite(t) & (z_hit >= 0.02) & (z_hit <= person.height)

        # top cap (looking slightly down: rare but keeps geometry correct)
        cap = dirs[:, 2] > 1e-6
        if cap.any():
            t_top = np.full(dirs.shape[0], np.inf)
            t_top[cap] = (person.height - cam_pos[2]) / dirs[cap, 2]
            top_pt = o[None, :] + d2 * t_top[:, None]
            inside = np.hypot(top_pt[:, 0] - p[0], top_pt[:, 1] - p[1]) <= r
            cand = cap & inside & (t_top > 0.2)
            better = cand & (t_top < t)
            t[better] = t_top[better]
            z_hit[better] = person.height
            ok = ok | better

        if not ok.any():
            return
        depth_person = t[ok] * self._inv_norm.reshape(-1)[ok]
        closer = depth_person < np.where(depth[ok] > 0, depth[ok], np.inf)
        idx_candidates = np.nonzero(ok)[0]
        if idx_candidates.size == 0:
            return
        sel = idx_candidates[closer]
        if sel.size == 0:
            return
        z_sel = z_hit[sel]
        jacket = np.array(person.color_rgb, dtype=np.float64)
        legs = np.array([56.0, 60.0, 68.0])
        head = np.array([188.0, 180.0, 172.0])  # low saturation, not a jacket blob
        shade = np.clip(1.25 / (1.0 + 0.12 * np.clip(t[sel], 0, None)), 0.55, 1.0)
        body = np.where(
            (z_sel[:, None] > 1.50),
            head[None, :],
            np.where((z_sel[:, None] > 0.72), jacket[None, :], legs[None, :]),
        )
        rgb[sel] = body * shade[:, None]
        depth[sel] = depth_person[closer]


def undistort_rgbd(
    rgb: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort the RGB-D pair with the camera calibration (OpenCV model)."""
    K = intrinsics.K
    D = intrinsics.D
    rgb_u = cv2.undistort(rgb, K, D)
    depth_u = cv2.undistort(depth, K, D)
    return rgb_u, depth_u
