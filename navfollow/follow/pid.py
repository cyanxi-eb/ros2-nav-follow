"""PID controllers for the differential-drive follower.

Two loops run at the control rate (10 Hz):

* distance loop -> linear velocity,
* bearing loop  -> angular velocity,

with derivative damping, output clamping, conditional integration (anti-windup)
and a "turn in place when the bearing error is large" priority rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..geometry import clamp


@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    integral_limit: float = 1.0


class PID:
    """Scalar PID with anti-windup and derivative-on-measurement damping."""

    def __init__(self, config: PIDConfig) -> None:
        self.cfg = config
        self.integral = 0.0
        self.prev_error: float | None = None
        self.last_output = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None
        self.last_output = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(1e-4, dt)
        p = self.cfg.kp * error
        self.integral = clamp(
            self.integral + error * dt,
            -self.cfg.integral_limit,
            self.cfg.integral_limit,
        )
        i = self.cfg.ki * self.integral
        if self.prev_error is None:
            self.prev_error = error
        d = self.cfg.kd * (error - self.prev_error) / dt
        self.prev_error = error
        out = p + i + d
        clamped = clamp(out, self.cfg.out_min, self.cfg.out_max)
        # conditional integration: do not wind up against the output limit
        if abs(out - clamped) > 1e-9:
            self.integral = clamp(
                self.integral - error * dt,
                -self.cfg.integral_limit,
                self.cfg.integral_limit,
            )
        self.last_output = clamped
        return clamped


@dataclass
class FollowPIDConfig:
    desired_distance: float = 1.40
    min_distance: float = 0.70
    emergency_distance: float = 0.55
    max_v: float = 0.62
    min_v: float = -0.12
    max_w: float = 1.20
    kp_dist: float = 1.30
    kd_dist: float = 0.12
    kp_bearing: float = 1.70
    kd_bearing: float = 0.06
    turn_first_angle: float = 0.70
    turn_first_scale: float = 0.20
    distance_deadband: float = 0.05


class FollowPID:
    """Distance + bearing PID pair specialised for person following."""

    def __init__(self, config: FollowPIDConfig | None = None) -> None:
        self.cfg = config or FollowPIDConfig()
        self.distance_pid = PID(
            PIDConfig(
                kp=self.cfg.kp_dist,
                kd=self.cfg.kd_dist,
                out_min=self.cfg.min_v,
                out_max=self.cfg.max_v,
                integral_limit=0.6,
            )
        )
        self.bearing_pid = PID(
            PIDConfig(
                kp=self.cfg.kp_bearing,
                kd=self.cfg.kd_bearing,
                out_min=-self.cfg.max_w,
                out_max=self.cfg.max_w,
                integral_limit=0.5,
            )
        )
        self._prev_bearing: float | None = None

    def reset(self) -> None:
        self.distance_pid.reset()
        self.bearing_pid.reset()
        self._prev_bearing = None

    def compute(self, distance: float, bearing: float, dt: float) -> tuple[float, float]:
        """Return ``(v, w)`` for a target at ``distance`` / ``bearing``."""
        cfg = self.cfg
        # dead-band around the desired distance avoids dithering on the spot
        dist_error = distance - cfg.desired_distance
        if abs(dist_error) < cfg.distance_deadband:
            dist_error = 0.0
        v = self.distance_pid.update(dist_error, dt)
        if bearing is not None and abs(bearing) > cfg.turn_first_angle:
            weight = max(0.0, math.cos(abs(bearing)))
            v *= max(cfg.turn_first_scale, weight)
        if distance < cfg.emergency_distance:
            v = cfg.min_v
        elif distance < cfg.min_distance:
            v = min(0.0, v)
        if bearing is None:
            w = 0.0
        else:
            w = self.bearing_pid.update(bearing, dt)
        return (clamp(v, cfg.min_v, cfg.max_v), clamp(w, -cfg.max_w, cfg.max_w))

    def spin(self, speed: float = 0.5) -> tuple[float, float]:
        return 0.0, speed
