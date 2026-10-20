"""Control algorithms used by ``follow_controller``.

ROS-free port of ``navfollow.follow``: constant-velocity Kalman filter, PID
distance/bearing loops and the SEARCH/TRACK/OCCLUDED/REACQUIRE/ARRIVED/STOP
state machine.  The node adds ROS2 IO (subscriptions, cmd_vel publishing, Nav2
NavigateToPose actions) on top of this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Kalman + PID
# ---------------------------------------------------------------------------
class KalmanCV2D:
    """Constant-velocity Kalman filter on ``[x, y, vx, vy]`` (map frame)."""

    def __init__(self, accel_std: float = 1.2, meas_std: float = 0.12) -> None:
        self.accel_std = accel_std
        self.meas_std = meas_std
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.initialized = False

    def reset(self, x: float, y: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([0.05, 0.05, 0.5, 0.5])
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return self.x
        dt = max(1e-3, dt)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = self.accel_std**2
        Q = q * np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ],
            float,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, mx: float, my: float) -> np.ndarray:
        if not self.initialized:
            self.reset(mx, my)
            return self.x
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.meas_std**2
        y = np.array([mx, my]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(math.hypot(self.x[2], self.x[3]))


@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    integral_limit: float = 1.0


class PID:
    """Scalar PID with anti-windup and derivative damping."""

    def __init__(self, config: PIDConfig) -> None:
        self.cfg = config
        self.integral = 0.0
        self.prev_error: float | None = None

    def update(self, error: float, dt: float) -> float:
        dt = max(1e-4, dt)
        self.integral = clamp(self.integral + error * dt, -self.cfg.integral_limit, self.cfg.integral_limit)
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        out = self.cfg.kp * error + self.cfg.ki * self.integral + self.cfg.kd * derivative
        return clamp(out, self.cfg.out_min, self.cfg.out_max)


# ---------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------
class FollowState(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    OCCLUDED = "OCCLUDED"
    REACQUIRE = "REACQUIRE"
    ARRIVED = "ARRIVED"
    STOP = "STOP"


@dataclass
class FollowParams:
    desired_distance: float = 1.40
    body_radius_offset: float = 0.22
    max_v: float = 0.62
    min_v: float = -0.12
    max_w: float = 1.20
    kp_dist: float = 1.30
    kd_dist: float = 0.12
    kp_bearing: float = 1.70
    kd_bearing: float = 0.06
    lost_timeout: float = 0.35
    occluded_timeout: float = 3.2
    reacquire_timeout: float = 6.0
    search_timeout: float = 12.0
    prediction_horizon: float = 0.18
    reid_high: float = 0.78
    reid_low: float = 0.52
    reid_consecutive: int = 3
    motion_gate_base: float = 1.6
    motion_gate_growth: float = 0.6
    arrival_hold: float = 3.0


class FollowStateMachine:
    """PID + Kalman + ReID dual-threshold person follower."""

    def __init__(self, params: FollowParams | None = None) -> None:
        self.p = params or FollowParams()
        self.distance_pid = PID(
            PIDConfig(kp=self.p.kp_dist, kd=self.p.kd_dist, out_min=self.p.min_v, out_max=self.p.max_v)
        )
        self.bearing_pid = PID(
            PIDConfig(
                kp=self.p.kp_bearing,
                kd=self.p.kd_bearing,
                out_min=-self.p.max_w,
                out_max=self.p.max_w,
            )
        )
        self.kalman = KalmanCV2D()
        self.state = FollowState.SEARCH
        self.last_seen: float | None = None
        self.occluded_since: float | None = None
        self.reacquire_since: float | None = None
        self.search_since: float | None = None
        self._weak: dict[int, int] = {}
        self.reacquire_attempts = 0
        self.reacquire_success = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = FollowState.SEARCH
        self.kalman = KalmanCV2D()
        self.last_seen = None
        self.occluded_since = None
        self.reacquire_since = None
        self.search_since = None
        self._weak.clear()

    def _to_base(self, world_xy, robot_pose):
        x, y, theta = robot_pose
        dx, dy = world_xy[0] - x, world_xy[1] - y
        c, s = math.cos(theta), math.sin(theta)
        return c * dx + s * dy, -s * dx + c * dy

    def _reacquire(self, now: float, candidates: list, reid_similarity: list[float], gate_ok: list[bool]):
        """Dual-threshold ReID re-association.

        ``candidates`` are ``(track_id, world_xy)`` tuples from the perception
        node; ``reid_similarity`` / ``gate_ok`` are aligned per candidate.
        """
        predicted = self.kalman.position if self.kalman.initialized else None
        for (track_id, world_xy), similarity, gate in zip(candidates, reid_similarity, gate_ok):
            if not gate or predicted is None:
                continue
            if similarity >= self.p.reid_high:
                return track_id, world_xy
            if similarity >= self.p.reid_low:
                self._weak[track_id] = self._weak.get(track_id, 0) + 1
                if self._weak[track_id] >= self.p.reid_consecutive:
                    return track_id, world_xy
        return None

    def _accept(self, track_id: int, world_xy, now: float) -> None:
        if self.kalman.initialized:
            dt = 0.1 if self.last_seen is None else max(1e-3, min(4.0, now - self.last_seen))
            self.kalman.predict(dt)
            self.kalman.update(world_xy[0], world_xy[1])
        else:
            self.kalman.reset(world_xy[0], world_xy[1])
        self.last_seen = now
        self.state = FollowState.TRACK
        self._weak.clear()
        self.reacquire_success += 1

    # ------------------------------------------------------------------
    def step(
        self,
        now: float,
        dt: float,
        robot_pose: tuple[float, float, float],
        target_world: tuple[float, float] | None,
        candidates: list | None = None,
        reid_similarity: list[float] | None = None,
        gate_ok: list[bool] | None = None,
        target_track_id: int | None = None,
    ) -> dict:
        """Advance the state machine one control period.

        Returns a dict with ``v``, ``w``, ``state``, ``target_track_id`` and
        ``predicted_world`` for the node to publish / actuate.
        """
        p = self.p
        out = {"v": 0.0, "w": 0.0, "state": self.state, "target_track_id": target_track_id}
        if target_world is not None:
            if not self.kalman.initialized:
                self.kalman.reset(target_world[0], target_world[1])
            else:
                if self.last_seen is not None:
                    self.kalman.predict(min(2.0, max(1e-3, now - self.last_seen)))
                self.kalman.update(target_world[0], target_world[1])
            self.last_seen = now
            if self.state in (FollowState.SEARCH, FollowState.OCCLUDED, FollowState.REACQUIRE):
                self.state = FollowState.TRACK
            predicted = self.kalman.position
            # latency compensation: aim where the target will be
            px, py = self.kalman.velocity
            predicted = (predicted[0] + px * p.prediction_horizon, predicted[1] + py * p.prediction_horizon)
            forward, left = self._to_base(predicted, robot_pose)
            distance = math.hypot(forward, left) + p.body_radius_offset
            bearing = math.atan2(left, forward)
            v = self.distance_pid.update(distance - p.desired_distance, dt)
            if abs(bearing) > 0.7:
                v *= max(0.2, math.cos(bearing))
            if self.kalman.speed > 0.04:
                c, s = math.cos(robot_pose[2]), math.sin(robot_pose[2])
                v = clamp(v + c * px + s * py, p.min_v, p.max_v)
            w = self.bearing_pid.update(bearing, dt)
            out.update(v=float(v), w=float(w), state=self.state, predicted_world=predicted,
                       distance=float(distance), bearing=float(bearing))
            return out

        # ---- target not visible --------------------------------------
        if self.state == FollowState.SEARCH:
            if self.search_since is None:
                self.search_since = now
            match = self._reacquire(now, candidates or [], reid_similarity or [], gate_ok or [])
            if match is not None:
                self._accept(match[0], match[1], now)
                out.update(state=self.state, target_track_id=match[0])
                return out
            if now - self.search_since > p.search_timeout:
                self.state = FollowState.STOP
            out.update(state=self.state, w=0.55)
            return out

        if self.state in (FollowState.TRACK, FollowState.ARRIVED):
            since = now - self.last_seen if self.last_seen is not None else 1e9
            if since > p.lost_timeout:
                self.state = FollowState.OCCLUDED
                self.occluded_since = now

        if self.state == FollowState.OCCLUDED:
            match = self._reacquire(now, candidates or [], reid_similarity or [], gate_ok or [])
            if match is not None:
                self._accept(match[0], match[1], now)
                out.update(state=self.state, target_track_id=match[0])
                return out
            since = now - self.last_seen if self.last_seen is not None else 1e9
            if since > p.occluded_timeout:
                self.state = FollowState.REACQUIRE
                self.reacquire_since = now
                self.reacquire_attempts += 1
                out.update(state=self.state, w=0.55)
                return out
            if self.kalman.initialized:
                predicted = self.kalman.position
                forward, left = self._to_base(predicted, robot_pose)
                distance = math.hypot(forward, left) + p.body_radius_offset
                bearing = math.atan2(left, forward)
                v = 0.6 * self.distance_pid.update(distance - p.desired_distance, dt)
                w = self.bearing_pid.update(bearing, dt)
                out.update(v=float(v), w=float(w), state=self.state, predicted_world=predicted)
            return out

        if self.state == FollowState.REACQUIRE:
            match = self._reacquire(now, candidates or [], reid_similarity or [], gate_ok or [])
            if match is not None:
                self._accept(match[0], match[1], now)
                out.update(state=self.state, target_track_id=match[0])
                return out
            elapsed = now - self.reacquire_since if self.reacquire_since is not None else 1e9
            if elapsed > p.reacquire_timeout:
                self.state = FollowState.STOP
            direction = 1.0 if int(elapsed / 3.0) % 2 == 0 else -1.0
            out.update(state=self.state, w=direction * 0.55)
            return out

        out["state"] = FollowState.STOP
        return out
