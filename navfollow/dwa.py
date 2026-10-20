"""Dynamic Window Approach (DWA) local planner.

A Nav2-DWB-flavoured local controller: sample ``(v, w)`` pairs inside the
dynamic window accessible within one control period, forward-simulate each
candidate for the prediction horizon, reject candidates that collide, and score
the survivors on:

* ``path_dist``  - distance of the end pose to the global path,
* ``goal_align`` - heading toward the lookahead point,
* ``path_align`` - alignment with the local path tangent,
* ``clearance``  - distance-to-obstacle at the end pose,
* ``velocity``   - prefer fast progress,
* ``dynamic``    - keep away from moving persons.

The same scoring terms (and names) exist in ``config/nav2_params.yaml`` under
the DWB controller so the offline and ROS2 configurations stay comparable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Pose2D, angle_diff, clamp
from .maps import GridMap
from .planning import (
    closest_path_index,
    heading_of_segment,
    lookahead_point,
    path_length,
    point_at_distance,
    point_to_path_distance,
    project_to_path,
)


@dataclass
class DWAConfig:
    max_v: float = 0.55
    min_v: float = -0.10
    max_w: float = 1.20
    max_acc_v: float = 1.00  # m/s^2
    max_acc_w: float = 4.00  # rad/s^2
    v_samples: int = 5
    w_samples: int = 11
    predict_time: float = 1.2
    sim_dt: float = 0.10
    robot_radius: float = 0.18
    goal_tolerance: float = 0.35
    lookahead: float = 0.85
    weight_path_dist: float = 2.2
    weight_goal_align: float = 0.9
    weight_path_align: float = 0.9
    weight_clearance: float = 0.8
    weight_velocity: float = 0.7
    weight_dynamic: float = 1.4
    # obstacle costs
    lethal_distance: float = 0.20
    safety_distance: float = 0.35
    dynamic_safety: float = 0.55
    # stop when very close to the goal
    slow_radius: float = 1.2


@dataclass
class DWACommand:
    v: float
    w: float
    score: float
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    candidates: int = 0
    admissible: int = 0


class DWAPlanner:
    """DWA local planner over a static occupancy grid + dynamic person circles."""

    def __init__(self, grid: GridMap, config: DWAConfig | None = None) -> None:
        self.grid = grid
        self.cfg = config or DWAConfig()
        self.grid.ensure_derived()

    # ------------------------------------------------------------------
    def _collides(self, x: float, y: float, persons: list[tuple[float, float, float]]) -> bool:
        if self.grid.robot_blocked(x, y):
            return True
        for (px, py, pr) in persons:
            if math.hypot(px - x, py - y) < pr + self.cfg.robot_radius + 0.05:
                return True
        return False

    def _clearance_cost(self, x: float, y: float) -> float:
        """0 = already at lethal distance, 1 = safe.  Uses the margin layer."""
        cfg = self.cfg
        if self.grid.margin_blocked(x, y):
            # inside the margin ring: interpolate by scanning a few radii
            for r in (0.30, 0.26, 0.22):
                if not self.grid.robot_blocked(x, y):
                    return 0.5 * (r / cfg.safety_distance)
            return 0.0
        return 1.0

    # ------------------------------------------------------------------
    def compute(
        self,
        pose: Pose2D,
        velocity: tuple[float, float],
        path: list[tuple[float, float]],
        persons: list[tuple[float, float, float]] | None = None,
    ) -> DWACommand | None:
        """Return the best admissible command, or ``None`` when every sample collides."""
        cfg = self.cfg
        persons = persons or []
        if len(path) < 2:
            return None
        v0, w0 = velocity

        # dynamic window
        v_lo = max(cfg.min_v, v0 - cfg.max_acc_v * cfg.sim_dt)
        v_hi = min(cfg.max_v, v0 + cfg.max_acc_v * cfg.sim_dt)
        w_lo = max(-cfg.max_w, w0 - cfg.max_acc_w * cfg.sim_dt)
        w_hi = min(cfg.max_w, w0 + cfg.max_acc_w * cfg.sim_dt)
        vs = np.linspace(v_lo, v_hi, cfg.v_samples)
        ws = np.linspace(w_lo, w_hi, cfg.w_samples)

        idx, _d = closest_path_index(path, (pose.x, pose.y))
        path_len = path_length(path)
        _dev, s_here, _proj = project_to_path(path, (pose.x, pose.y))
        s_carrot = min(path_len, s_here + cfg.lookahead)
        carrot = point_at_distance(path, s_carrot)
        goal = path[-1]
        dist_to_goal = math.hypot(goal[0] - pose.x, goal[1] - pose.y)

        # Local tangent = direction from the robot to the carrot.  Using the
        # global end point here (as a naive implementation does) makes the robot
        # cut corners and is one of the failure cases in docs/failure_cases.md.
        if math.hypot(carrot[0] - pose.x, carrot[1] - pose.y) > 0.15:
            tangent_heading = math.atan2(carrot[1] - pose.y, carrot[0] - pose.x)
        else:
            tangent_heading = pose.theta
        # aim at the carrot until the goal is close, then at the goal itself
        target = carrot if dist_to_goal > cfg.slow_radius else goal

        n_steps = max(1, int(round(cfg.predict_time / cfg.sim_dt)))
        best: DWACommand | None = None
        n_candidates = 0
        n_admissible = 0

        for v in vs:
            for w in ws:
                n_candidates += 1
                x, y, th = pose.x, pose.y, pose.theta
                traj = [(x, y)]
                collided = False
                margin_hits = 0
                for _ in range(n_steps):
                    x += v * math.cos(th) * cfg.sim_dt
                    y += v * math.sin(th) * cfg.sim_dt
                    th += w * cfg.sim_dt
                    if self._collides(x, y, persons):
                        collided = True
                        break
                    if self.grid.margin_blocked(x, y):
                        margin_hits += 1
                    traj.append((x, y))
                if collided:
                    continue
                n_admissible += 1

                d_path, _ = point_to_path_distance(path, (x, y))
                if dist_to_goal < cfg.goal_tolerance:
                    goal_align = 1.0
                else:
                    goal_theta = math.atan2(target[1] - y, target[0] - x)
                    goal_align = math.cos(angle_diff(goal_theta, th))
                path_align = math.cos(angle_diff(tangent_heading, th))
                clearance = 1.0 - min(1.0, margin_hits / max(1, n_steps))
                vel_term = clamp(v / max(cfg.max_v, 1e-6), 0.0, 1.0)
                # dynamic obstacle penalty: minimum predicted distance to persons
                dyn = 1.0
                for (px, py, pr) in persons:
                    for (tx, ty) in traj[:: max(1, len(traj) // 4)]:
                        d = math.hypot(px - tx, py - ty)
                        limit = pr + cfg.dynamic_safety
                        if d < limit:
                            dyn = min(dyn, max(0.0, d / max(limit, 1e-6)))

                score = (
                    cfg.weight_path_dist * (1.0 - min(1.0, d_path / 1.5))
                    + cfg.weight_goal_align * goal_align
                    + cfg.weight_path_align * path_align
                    + cfg.weight_clearance * clearance
                    + cfg.weight_velocity * vel_term
                    + cfg.weight_dynamic * dyn
                )
                # decelerate near the goal
                if dist_to_goal < cfg.slow_radius:
                    score += 0.35 * (1.0 - dist_to_goal / cfg.slow_radius)
                if best is None or score > best.score:
                    best = DWACommand(
                        v=float(v),
                        w=float(w),
                        score=float(score),
                        trajectory=traj,
                        candidates=n_candidates,
                        admissible=n_admissible,
                    )
        if best is None:
            return None
        best.candidates = n_candidates
        best.admissible = n_admissible
        return best


def rotate_in_place_command(w: float = 0.8) -> tuple[float, float]:
    return 0.0, w
