"""Constant-velocity Kalman filter for the followed target.

State: ``[x, y, vx, vy]`` in the ``map`` frame.  The follower uses it for two
things:

* smoothing the noisy RGB-D target position (distance jitter),
* extrapolating the target through a 1-3 s occlusion so the robot keeps moving
  smoothly while ReID waits for the person to reappear.
"""
from __future__ import annotations

import numpy as np


class KalmanCV2D:
    """2-D constant-velocity Kalman filter."""

    def __init__(self, accel_std: float = 1.2, meas_std: float = 0.12) -> None:
        self.accel_std = accel_std
        self.meas_std = meas_std
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0
        self.initialized = False
        self.updates = 0
        self.misses = 0

    # ------------------------------------------------------------------
    def reset(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.diag([0.05, 0.05, 0.5, 0.5])
        self.initialized = True
        self.updates = 0
        self.misses = 0

    def _matrices(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(1e-3, float(dt))
        F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        q = self.accel_std * self.accel_std
        Q = q * np.array(
            [
                [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                [dt**3 / 2.0, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2.0, 0.0, dt**2],
            ],
            dtype=float,
        )
        R = np.eye(2) * (self.meas_std**2)
        return F, Q, R

    # ------------------------------------------------------------------
    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return self.x
        F, Q, _R = self._matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, mx: float, my: float) -> np.ndarray:
        if not self.initialized:
            self.reset(mx, my)
            return self.x
        _F, _Q, R = self._matrices(0.1)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        z = np.array([mx, my], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.updates += 1
        return self.x

    # ------------------------------------------------------------------
    def predict_position(self, dt: float) -> tuple[float, float]:
        """Extrapolated position ``dt`` seconds ahead (no state mutation)."""
        if not self.initialized:
            return (float(self.x[0]), float(self.x[1]))
        return (float(self.x[0] + self.x[2] * dt), float(self.x[1] + self.x[3] * dt))

    @property
    def position(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))
