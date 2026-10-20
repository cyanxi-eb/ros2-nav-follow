"""Small 2D/3D geometry helpers shared by simulation, perception and control.

Everything here is deliberately dependency-light (numpy only) so that the same
functions can run inside a ROS2 node or in the offline simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------
def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (angle + math.pi) % TWO_PI - math.pi
    # map the excluded -pi endpoint to +pi
    return math.pi if wrapped == -math.pi else wrapped


def angle_diff(target: float, source: float) -> float:
    """Shortest signed rotation from ``source`` to ``target``."""
    return wrap_angle(target - source)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def sgn(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


# ---------------------------------------------------------------------------
# rotation / transform matrices
# ---------------------------------------------------------------------------
def rot2(theta: float) -> np.ndarray:
    """2x2 rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/ROS convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (x, y, z, w) (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=float)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def make_transform(
    translation: tuple[float, float, float] | np.ndarray,
    quaternion: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation + rotation/quaternion."""
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float)
    if quaternion is not None:
        T[:3, :3] = quaternion_to_matrix(quaternion)
    elif rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=float)
    return T


def transform_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=float)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def transform_from_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    return make_transform((x, y, z), quaternion_from_yaw(yaw))


def apply_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to an (N,3) array of points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts @ T[:3, :3].T + T[:3, 3]


def yaw_from_matrix(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------------------
# planar pose
# ---------------------------------------------------------------------------
@dataclass
class Pose2D:
    """Planar pose (x [m], y [m], theta [rad])."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def matrix(self) -> np.ndarray:
        return transform_from_pose(self.x, self.y, self.theta)

    def inverse(self) -> "Pose2D":
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.theta)

    def transform_point(self, px: float, py: float) -> tuple[float, float]:
        """Map a point from the world frame into this pose's local frame."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dy = px - self.x, py - self.y
        return c * dx + s * dy, -s * dx + c * dy

    def compose(self, other: "Pose2D") -> "Pose2D":
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.theta + other.theta),
        )

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.theta)


def relative_pose(pose: Pose2D, target_xy: tuple[float, float]) -> tuple[float, float]:
    """Return (distance, bearing) of a world point relative to a planar pose.

    bearing > 0 means "to the left" (counter-clockwise) in the robot frame.
    """
    dx = target_xy[0] - pose.x
    dy = target_xy[1] - pose.y
    distance = math.hypot(dx, dy)
    bearing = wrap_angle(math.atan2(dy, dx) - pose.theta)
    return distance, bearing


def normalized(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < eps:
        return np.zeros_like(vec)
    return vec / n
