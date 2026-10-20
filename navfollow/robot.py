"""Differential-drive kinematics and a simple odometry integrator."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose2D, wrap_angle


@dataclass
class RobotConfig:
    radius: float = 0.18
    wheel_separation: float = 0.32
    max_v: float = 0.55
    max_w: float = 1.20
    max_acc_v: float = 0.6
    max_acc_w: float = 2.2


def integrate_pose(pose: Pose2D, v: float, w: float, dt: float) -> Pose2D:
    """Exact unicycle integration for a constant command over ``dt``."""
    if abs(w) < 1e-9:
        return Pose2D(pose.x + v * dt * math.cos(pose.theta), pose.y + v * dt * math.sin(pose.theta), pose.theta)
    theta_new = pose.theta + w * dt
    radius = v / w
    return Pose2D(
        pose.x + radius * (math.sin(theta_new) - math.sin(pose.theta)),
        pose.y - radius * (math.cos(theta_new) - math.cos(pose.theta)),
        wrap_angle(theta_new),
    )


class OdomIntegrator:
    """Integrates wheel commands into ``odom`` with optional bias / noise.

    The bias terms let the simulator reproduce the real-world failure mode
    documented in ``docs/failure_cases.md``: a small yaw bias makes the
    odometry frame drift away from the map frame, which is exactly what
    ``odom -> base_link`` TF and the wheel calibration fix.
    """

    def __init__(
        self,
        yaw_bias: float = 0.0,
        linear_scale: float = 1.0,
        noise_std_v: float = 0.0,
        noise_std_w: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.yaw_bias = yaw_bias
        self.linear_scale = linear_scale
        self.noise_std_v = noise_std_v
        self.noise_std_w = noise_std_w
        self._rng = __import__("random").Random(seed)

    def step(self, odom: Pose2D, v: float, w: float, dt: float) -> Pose2D:
        v_hat = v * self.linear_scale
        if self.noise_std_v:
            v_hat += self._rng.gauss(0.0, self.noise_std_v)
        w_hat = w + self.yaw_bias
        if self.noise_std_w:
            w_hat += self._rng.gauss(0.0, self.noise_std_w)
        return integrate_pose(odom, v_hat, w_hat, dt)
