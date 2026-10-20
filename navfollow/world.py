"""World model: occupancy map + pedestrians + lidar simulation + visibility.

The lidar is simulated with vectorised ray marching (fixed 6 cm step) and
analytic ray/circle intersection for pedestrians, which is fast enough to run
tens of thousands of scans on a CPU-only laptop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose2D
from .maps import GridMap
from .persons import Person


@dataclass
class LidarConfig:
    n_rays: int = 121
    angle_min: float = -2.35  # rad, robot frame
    angle_max: float = 2.35
    max_range: float = 5.0
    step: float = 0.06
    noise_std: float = 0.0


class SimWorld:
    """Mutable world state used by all trial runners."""

    def __init__(
        self,
        grid: GridMap,
        persons: list[Person] | None = None,
        lidar: LidarConfig | None = None,
    ) -> None:
        self.grid = grid
        self.persons: list[Person] = persons or []
        self.lidar = lidar or LidarConfig()
        self.grid.ensure_derived()
        # pre-computed ray-march distances
        self._ray_steps = np.arange(1, int(self.lidar.max_range / self.lidar.step) + 1) * self.lidar.step
        self._ray_angles = np.linspace(self.lidar.angle_min, self.lidar.angle_max, self.lidar.n_rays)

    # ------------------------------------------------------------------
    def add_person(self, person: Person) -> Person:
        self.persons.append(person)
        return person

    def step(self, dt: float, robot_pose: Pose2D | None = None) -> None:
        """Advance every person; pedestrians stop when the robot is too close."""
        for p in self.persons:
            if robot_pose is not None and p.yield_radius > 0.0:
                if math.hypot(p.x - robot_pose.x, p.y - robot_pose.y) < p.yield_radius:
                    continue
            p.step(dt)

    # ------------------------------------------------------------------
    def lidar_scan(self, pose: Pose2D, noise_rng: np.random.Generator | None = None) -> np.ndarray:
        """Return ranges (metres) for the configured beam fan."""
        cfg = self.lidar
        angles = pose.theta + self._ray_angles
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n,2)
        pts = np.empty((cfg.n_rays, self._ray_steps.size, 2), dtype=np.float32)
        pts[..., 0] = pose.x + dirs[:, 0:1] * self._ray_steps[None, :]
        pts[..., 1] = pose.y + dirs[:, 1:2] * self._ray_steps[None, :]
        res = self.grid.resolution
        cols = np.floor((pts[..., 0] - self.grid.origin[0]) / res).astype(np.int32)
        rows = np.floor((pts[..., 1] - self.grid.origin[1]) / res).astype(np.int32)
        inside = (
            (cols >= 0)
            & (cols < self.grid.width)
            & (rows >= 0)
            & (rows < self.grid.height)
        )
        occ = np.zeros_like(inside, dtype=bool)
        occ[inside] = self.grid.occupancy[rows[inside], cols[inside]]
        has_hit = occ.any(axis=1)
        first = np.argmax(occ, axis=1)
        ranges = np.full(cfg.n_rays, cfg.max_range, dtype=np.float64)
        ranges[has_hit] = self._ray_steps[first[has_hit]]

        # analytic ray/circle intersection with pedestrians
        o = np.array([pose.x, pose.y])
        for p in self.persons:
            if p.hidden:
                continue
            c = np.array([p.x, p.y])
            d = dirs
            oc = o - c
            a = np.einsum("ij,ij->i", d, d)
            b = 2.0 * (d @ oc)
            cc = float(oc @ oc) - p.radius * p.radius
            disc = b * b - 4 * a * cc
            mask = disc > 0
            if not mask.any():
                continue
            t = np.empty_like(disc)
            t[:] = np.inf
            sq = np.sqrt(disc[mask])
            t_near = (-b[mask] - sq) / (2 * a[mask])
            t_far = (-b[mask] + sq) / (2 * a[mask])
            t[mask] = np.where(t_near > 0, t_near, t_far)
            ranges = np.minimum(ranges, t)
        if noise_rng is not None and cfg.noise_std > 0:
            ranges = ranges + noise_rng.normal(0.0, cfg.noise_std, size=ranges.shape)
        return np.clip(ranges, 0.05, cfg.max_range)

    # ------------------------------------------------------------------
    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True when no occupied cell lies on the straight segment a -> b."""
        ax, ay = a
        bx, by = b
        dist = math.hypot(bx - ax, by - ay)
        if dist < 1e-6:
            return True
        step = self.grid.resolution * 0.9
        n = int(dist / step) + 1
        for i in range(1, n):
            t = i / n
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            if self.grid.cell_occupied(x, y):
                return False
        return True

    def person_visible(
        self,
        person: Person,
        pose: Pose2D,
        half_fov: float = 0.52,
        camera_offset: tuple[float, float] = (0.12, 0.0),
    ) -> bool:
        """Ground-truth visibility from the camera mount (used for metrics only)."""
        if person.hidden:
            return False
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        cam = (
            pose.x + c * camera_offset[0] - s * camera_offset[1],
            pose.y + s * camera_offset[0] + c * camera_offset[1],
        )
        dx, dy = person.x - cam[0], person.y - cam[1]
        dist = math.hypot(dx, dy)
        if dist < 0.25:
            return True
        bearing = math.atan2(dy, dx) - pose.theta
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        angular_radius = math.atan2(person.radius, dist)
        if abs(bearing) > half_fov + angular_radius:
            return False
        return self.line_of_sight(cam, (person.x, person.y))

    # ------------------------------------------------------------------
    def robot_collides(self, pose: Pose2D) -> bool:
        if self.grid.robot_blocked(pose.x, pose.y):
            return True
        for p in self.persons:
            if p.hidden:
                continue
            if math.hypot(p.x - pose.x, p.y - pose.y) < p.radius + self.grid.robot_radius:
                return True
        return False

    def person_contacts_robot(self, pose: Pose2D) -> bool:
        for p in self.persons:
            if p.hidden:
                continue
            if math.hypot(p.x - pose.x, p.y - pose.y) < p.radius + self.grid.robot_radius:
                return True
        return False

    def dynamic_circles(self) -> list[tuple[float, float, float]]:
        return [(p.x, p.y, p.radius) for p in self.persons if not p.hidden]
