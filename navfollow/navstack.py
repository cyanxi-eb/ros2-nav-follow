"""A Nav2-like navigation stack for the offline simulator.

Pipeline per trial::

    set_goal -> A* global plan (planning costmap) -> DWA local control
             -> stall / no-admissible detection -> recovery behaviour
             -> clear + replan (relaxed inflation) -> resume tracking

The recovery behaviour mirrors the Nav2 default behaviour tree in
``config/nav2_params.yaml``::

    RecoveryNode
      Sequence
        ClearEntireCostmap
        Spin
        BackUp
        Wait
      Sequence
        ClearEntireCostmap
        Spin
        Wait

We record every event so the evaluation can report how often recovery was
needed, which is a much more honest metric than only reporting successes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .dwa import DWAConfig, DWAPlanner
from .geometry import Pose2D, angle_diff, clamp
from .maps import GridMap
from .planning import (
    PlanningGrid,
    astar,
    build_planning_grid,
    closest_path_index,
    path_length,
    point_to_path_distance,
    project_to_path,
    smooth_path,
)


class NavStatus(Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    COLLISION = "collision"


@dataclass
class NavConfig:
    control_dt: float = 0.10
    goal_tolerance: float = 0.35
    max_time: float = 120.0
    replan_period: float = 2.0
    stall_timeout: float = 3.5
    progress_epsilon: float = 0.06
    max_recovery_attempts: int = 4
    spin_speed: float = 0.85
    spin_duration: float = 1.4
    backup_speed: float = 0.13
    backup_duration: float = 1.0
    deviation_replan: float = 0.9
    max_recovery_time: float = 20.0


@dataclass
class NavStep:
    v: float
    w: float
    status: NavStatus
    recovery: str | None = None
    events: list[dict] = field(default_factory=list)
    path: list[tuple[float, float]] | None = None


class Nav2LikeStack:
    """Global planner + local planner + recovery behaviours."""

    def __init__(self, grid: GridMap, config: NavConfig | None = None, dwa_config: DWAConfig | None = None) -> None:
        self.grid = grid
        self.cfg = config or NavConfig()
        self.dwa_config = dwa_config or DWAConfig()
        self.dwa = DWAPlanner(grid, self.dwa_config)
        self.pgrid_strict = build_planning_grid(grid, 0.1, 0.34, 0.62)
        self.pgrid_relaxed = build_planning_grid(grid, 0.1, 0.20, 0.50)
        self.goal: tuple[float, float] | None = None
        self.path: list[tuple[float, float]] | None = None
        self.path_length: float = 0.0
        self.time: float = 0.0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.goal = None
        self.path = None
        self.path_length = 0.0
        self.events: list[dict] = []
        self.recoveries: int = 0
        self.replan_count: int = 0
        self.status = NavStatus.RUNNING
        self._recovery_queue: deque[tuple[str, float]] = deque()
        self._recovery_timer = 0.0
        self._recovery_started: float | None = None
        self._recovery_active = False
        self._recovery_attempts = 0
        self._last_replan = -1e9
        self._last_progress_s = 0.0
        self._last_progress_t = 0.0
        self.trajectory: list[tuple[float, float]] = []
        self.planned_paths: list[list[tuple[float, float]]] = []

    def refresh_maps(self) -> None:
        """Rebuild derived layers after the world map changed."""
        self.grid.build_derived()
        self.dwa.grid = self.grid
        self.pgrid_strict = build_planning_grid(self.grid, 0.1, 0.34, 0.62)
        self.pgrid_relaxed = build_planning_grid(self.grid, 0.1, 0.20, 0.50)

    # ------------------------------------------------------------------
    def set_goal(self, goal: tuple[float, float]) -> None:
        self.goal = goal
        self.path = None
        self.path_length = 0.0
        self.status = NavStatus.RUNNING
        self._last_replan = -1e9
        self._recovery_queue.clear()
        self._recovery_active = False
        self._recovery_attempts = 0

    def plan(self, pose: Pose2D, relaxed: bool = False) -> list[tuple[float, float]] | None:
        if self.goal is None:
            return None
        pgrid = self.pgrid_relaxed if relaxed else self.pgrid_strict
        raw = astar(pgrid, (pose.x, pose.y), self.goal)
        if raw is None:
            return None
        self.replan_count += 1
        path = smooth_path(pgrid, raw)
        self.path = path
        self.path_length = path_length(path)
        self.planned_paths.append(path)
        self._last_replan = self.time
        self._last_progress_s = 0.0
        self._last_progress_t = self.time
        return path

    # ------------------------------------------------------------------
    def step(
        self,
        pose: Pose2D,
        velocity: tuple[float, float],
        dt: float | None = None,
        persons: list[tuple[float, float, float]] | None = None,
    ) -> NavStep:
        dt = dt if dt is not None else self.cfg.control_dt
        self.time += dt
        events: list[dict] = []
        self.trajectory.append((pose.x, pose.y))

        if self.goal is None:
            return NavStep(0.0, 0.0, self.status, events=events, path=self.path)
        goal_dist = math.hypot(self.goal[0] - pose.x, self.goal[1] - pose.y)
        if goal_dist < self.cfg.goal_tolerance:
            self.status = NavStatus.SUCCEEDED
            return NavStep(0.0, 0.0, self.status, events=events, path=self.path)
        if self.time > self.cfg.max_time:
            self.status = NavStatus.TIMEOUT
            return NavStep(0.0, 0.0, self.status, events=events, path=self.path)

        # -------------------- recovery behaviour ----------------------
        if self._recovery_active:
            cmd, done = self._recovery_tick(pose, dt)
            if done:
                self._recovery_active = False
                self._recovery_queue.clear()
                self.replan_count += 1
                path = astar(self.pgrid_relaxed, (pose.x, pose.y), self.goal)
                if path is not None:
                    self.path = smooth_path(self.pgrid_relaxed, path)
                    self.path_length = path_length(self.path)
                    self.planned_paths.append(self.path)
                    self._last_replan = self.time
                    self._last_progress_s = 0.0
                    self._last_progress_t = self.time
                    events.append(self._event("replan_after_recovery"))
                if self._recovery_attempts >= self.cfg.max_recovery_attempts:
                    self.status = NavStatus.FAILED
                    return NavStep(0.0, 0.0, self.status, recovery="failed", events=events, path=self.path)
            return NavStep(cmd[0], cmd[1], self.status, recovery=self._recovery_name(), events=events, path=self.path)

        # -------------------- global path -----------------------------
        need_plan = self.path is None
        if not need_plan and self.time - self._last_replan > self.cfg.replan_period:
            deviation, _ = point_to_path_distance(self.path, (pose.x, pose.y))
            if deviation > self.cfg.deviation_replan:
                need_plan = True
                events.append(self._event("replan_deviation", deviation=round(deviation, 3)))
        if need_plan:
            path = self.plan(pose)
            if path is None:
                events.append(self._event("plan_failed"))
                self._start_recovery(pose)
                return NavStep(0.0, 0.0, self.status, recovery="no_path", events=events, path=None)
            events.append(self._event("replan"))

        # -------------------- local control ---------------------------
        cmd = self.dwa.compute(pose, velocity, self.path, persons=persons)
        if cmd is None:
            events.append(self._event("no_admissible_velocity"))
            self._start_recovery(pose)
            return NavStep(0.0, 0.0, self.status, recovery="no_admissible", events=events, path=self.path)

        # stall detection: no forward progress along the tracked path for N seconds
        if self._stalled(pose):
            events.append(self._event("stalled"))
            self._start_recovery(pose)
            return NavStep(0.0, 0.0, self.status, recovery="stalled", events=events, path=self.path)

        return NavStep(cmd.v, cmd.w, self.status, events=events, path=self.path)

    # ------------------------------------------------------------------
    def _stalled(self, pose: Pose2D) -> bool:
        """No arc-length progress along the tracked path for ``stall_timeout``.

        Path-progress is a much better stall signal than raw displacement: a
        robot rotating in place or oscillating in front of an unexpected
        obstacle makes no progress and must trigger the recovery behaviour.
        """
        if self.path is None or len(self.path) < 2:
            return False
        _dev, s, _proj = project_to_path(self.path, (pose.x, pose.y))
        if s - self._last_progress_s > self.cfg.progress_epsilon:
            self._last_progress_s = s
            self._last_progress_t = self.time
            return False
        return (self.time - self._last_progress_t) > self.cfg.stall_timeout

    def _start_recovery(self, pose: Pose2D) -> None:
        self.recoveries += 1
        self._recovery_attempts += 1
        self._recovery_active = True
        self._recovery_started = self.time
        self._recovery_queue = deque(
            [("clear", 0.0), ("spin", self.cfg.spin_duration), ("backup", self.cfg.backup_duration)]
        )
        self._recovery_timer = 0.0
        self.path = None
        self.events.append(self._event("recovery_start", attempt=self._recovery_attempts))

    def _recovery_name(self) -> str:
        if not self._recovery_active:
            return "idle"
        if self._recovery_queue:
            return self._recovery_queue[0][0]
        return "spin"

    def _recovery_tick(self, pose: Pose2D, dt: float) -> tuple[tuple[float, float], bool]:
        """Returns (command, finished)."""
        if not self._recovery_queue:
            return (0.0, 0.0), True
        kind, duration = self._recovery_queue[0]
        if kind == "clear":
            self._recovery_queue.popleft()
            self.events.append(self._event("recovery_clear_costmap"))
            return (0.0, 0.0), False
        self._recovery_timer += dt
        if kind == "spin":
            cmd = (0.0, self.cfg.spin_speed)
        elif kind == "backup":
            # stop backing up if the robot would hit something behind it
            bx = pose.x - 0.18 * math.cos(pose.theta)
            by = pose.y - 0.18 * math.sin(pose.theta)
            cmd = (-self.cfg.backup_speed, 0.0) if not self.grid.robot_blocked(bx, by) else (0.0, 0.0)
        else:
            cmd = (0.0, 0.0)
        if self._recovery_timer >= duration:
            self._recovery_queue.popleft()
            self._recovery_timer = 0.0
            self.events.append(self._event(f"recovery_{kind}_done"))
            if not self._recovery_queue:
                return cmd, True
        if self._recovery_started is not None and self.time - self._recovery_started > self.cfg.max_recovery_time:
            self._recovery_queue.clear()
            return (0.0, 0.0), True
        return cmd, False

    def _event(self, name: str, **kwargs) -> dict:
        ev = {"type": name, "t": round(self.time, 3)}
        ev.update(kwargs)
        self.events.append(ev)
        return ev
