"""A minimal TF2-style transform tree.

The simulator and the ROS2 nodes share the same tree layout::

    map -> odom -> base_link -> camera_link -> camera_optical

``base_link -> camera_link`` carries the camera extrinsic calibration
(``docs/calibration.md``); ``camera_link -> camera_optical`` is the fixed
REP-103 rotation (x forward / y left / z up  ->  x right / y down / z forward).

The implementation supports:

* static and dynamic edges,
* a small timestamped history with linear + slerp interpolation,
* ``lookup(target, source, stamp)`` following ROS2 ``lookup_transform``
  semantics (returns ``T_target_source``, i.e. maps points from ``source`` to
  ``target``),
* explicit "stale buffer" errors so the perception pipeline can drop frames
  whose timestamps cannot be resolved instead of silently using wrong poses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    make_transform,
    matrix_to_quaternion,
    quaternion_slerp,
    transform_inverse,
)


class TransformError(RuntimeError):
    """Base error for the transform tree."""


class FrameNotFound(TransformError):
    pass


class ExtrapolationError(TransformError):
    """Requested timestamp is outside the buffered history."""


@dataclass
class _Edge:
    parent: str
    child: str
    static: bool = False
    # list of (stamp, 4x4 T_parent_child)
    samples: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def push(self, stamp: float, matrix: np.ndarray, max_history: int) -> None:
        self.samples.append((float(stamp), np.asarray(matrix, dtype=float)))
        self.samples.sort(key=lambda s: s[0])
        if len(self.samples) > max_history:
            self.samples = self.samples[-max_history:]

    def at(self, stamp: float | None, tolerance: float = 0.25) -> np.ndarray:
        if not self.samples:
            raise ExtrapolationError(f"no sample for edge {self.parent}->{self.child}")
        if self.static:
            return self.samples[-1][1]
        if len(self.samples) == 1:
            sample_stamp, sample_T = self.samples[0]
            if stamp is not None and abs(stamp - sample_stamp) > tolerance:
                raise ExtrapolationError(
                    f"stamp {stamp:.3f} outside the single buffered sample at {sample_stamp:.3f}"
                )
            return sample_T
        if stamp is None:
            return self.samples[-1][1]
        first_t, first_T = self.samples[0]
        last_t, last_T = self.samples[-1]
        if stamp <= first_t:
            if first_t - stamp > tolerance:
                raise ExtrapolationError(
                    f"stamp {stamp:.3f} older than buffer start {first_t:.3f}"
                )
            return first_T
        if stamp >= last_t:
            if stamp - last_t > tolerance:
                raise ExtrapolationError(
                    f"stamp {stamp:.3f} newer than buffer end {last_t:.3f}"
                )
            return last_T
        for (t0, T0), (t1, T1) in zip(self.samples, self.samples[1:]):
            if t0 <= stamp <= t1:
                if t1 - t0 < 1e-9:
                    return T1
                alpha = (stamp - t0) / (t1 - t0)
                return _interpolate(T0, T1, alpha)
        return last_T


def _interpolate(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    t0, t1 = T0[:3, 3], T1[:3, 3]
    q0 = matrix_to_quaternion(T0[:3, :3])
    q1 = matrix_to_quaternion(T1[:3, :3])
    q = quaternion_slerp(q0, q1, alpha)
    T = np.eye(4)
    from .geometry import quaternion_to_matrix

    T[:3, :3] = quaternion_to_matrix(q)
    T[:3, 3] = t0 + alpha * (t1 - t0)
    return T


class TransformTree:
    """ROS2-flavoured transform tree with per-edge timestamp history."""

    def __init__(self, max_history: int = 512, tolerance: float = 0.25) -> None:
        self.max_history = max_history
        self.tolerance = tolerance
        self._edges: dict[str, _Edge] = {}  # keyed by child frame
        self._frames: set[str] = set()

    # -- construction ------------------------------------------------------
    def add_frame(self, frame_id: str, parent: str | None = None) -> None:
        self._frames.add(frame_id)
        if parent is not None:
            self._frames.add(parent)

    def set_static(
        self,
        parent: str,
        child: str,
        translation=(0.0, 0.0, 0.0),
        quaternion=None,
        rotation=None,
    ) -> None:
        T = make_transform(translation, quaternion=quaternion, rotation=rotation)
        self._set(parent, child, T, stamp=0.0, static=True)

    def set_transform(
        self,
        parent: str,
        child: str,
        matrix: np.ndarray | None = None,
        translation=None,
        quaternion=None,
        stamp: float = 0.0,
    ) -> None:
        if matrix is None:
            T = make_transform(translation or (0.0, 0.0, 0.0), quaternion=quaternion)
        else:
            T = np.asarray(matrix, dtype=float)
        self._set(parent, child, T, stamp=stamp, static=False)

    def _set(self, parent: str, child: str, T: np.ndarray, stamp: float, static: bool) -> None:
        if parent == child:
            raise TransformError("cannot add self transform")
        self._frames.update((parent, child))
        edge = self._edges.get(child)
        if edge is None:
            edge = _Edge(parent=parent, child=child, static=static)
            self._edges[child] = edge
        elif edge.parent != parent:
            raise TransformError(
                f"frame {child} already has parent {edge.parent}, cannot reparent to {parent}"
            )
        edge.static = edge.static or static
        edge.push(stamp, T, self.max_history)

    # -- lookup ------------------------------------------------------------
    def _path_to_root(self, frame: str) -> list[tuple[str, str, np.ndarray | None, float | None]]:
        """Return list of (from, to, T_to_from, stamp) walking child -> root.

        ``T_to_from`` maps points from ``from`` to ``to``.
        """
        if frame not in self._frames:
            raise FrameNotFound(f"frame {frame!r} not in tree")
        path = []
        current = frame
        seen = set()
        while current in self._edges:
            if current in seen:  # defensive: should be impossible
                raise TransformError(f"cycle detected at {current}")
            seen.add(current)
            edge = self._edges[current]
            # edge.samples stores T_parent_child -> T_child_parent is the inverse
            path.append((current, edge.parent, None, None))
            current = edge.parent
        return path

    def _edge_matrix(self, child: str, stamp: float | None) -> np.ndarray:
        edge = self._edges[child]
        return edge.at(stamp, self.tolerance)

    def lookup(self, target_frame: str, source_frame: str, stamp: float | None = None) -> np.ndarray:
        """Return T_target_source mapping points from ``source_frame`` to ``target_frame``."""
        if target_frame == source_frame:
            return np.eye(4)
        for frame in (target_frame, source_frame):
            if frame not in self._frames:
                raise FrameNotFound(f"frame {frame!r} not in tree")
        # walk source up to root collecting edges
        src_chain: list[tuple[str, np.ndarray]] = []  # (frame, T_frame_parent)
        current = source_frame
        seen = set()
        while current in self._edges:
            if current in seen:
                raise TransformError(f"cycle detected at {current}")
            seen.add(current)
            T_parent_child = self._edge_matrix(current, stamp)
            # map points from child -> parent
            src_chain.append((current, T_parent_child))
            current = self._edges[current].parent
        src_root = current

        dst_chain: list[tuple[str, np.ndarray]] = []
        current = target_frame
        seen = set()
        while current in self._edges:
            if current in seen:
                raise TransformError(f"cycle detected at {current}")
            seen.add(current)
            T_parent_child = self._edge_matrix(current, stamp)
            dst_chain.append((current, T_parent_child))
            current = self._edges[current].parent
        dst_root = current

        if src_root != dst_root:
            raise TransformError(
                f"frames {source_frame!r} and {target_frame!r} are not connected "
                f"(roots {src_root!r} / {dst_root!r})"
            )

        # T_child_root = product of T_child_parent @ ... walking up
        T_src_root = np.eye(4)
        for _frame, T_parent_child in src_chain:
            T_src_root = T_parent_child @ T_src_root

        # T_target_root: same expression from target
        T_dst_root = np.eye(4)
        for _frame, T_parent_child in dst_chain:
            T_dst_root = T_parent_child @ T_dst_root

        # T_target_source = inv(T_dst_root) @ T_src_root
        return transform_inverse(T_dst_root) @ T_src_root

    def can_transform(self, target_frame: str, source_frame: str, stamp: float | None = None) -> bool:
        try:
            self.lookup(target_frame, source_frame, stamp)
            return True
        except TransformError:
            return False

    # -- inspection --------------------------------------------------------
    @property
    def frames(self) -> set[str]:
        return set(self._frames)

    def parent_of(self, frame: str) -> str | None:
        edge = self._edges.get(frame)
        return edge.parent if edge else None

    def latest_stamp(self) -> float:
        stamps = [s[0] for e in self._edges.values() for s in e.samples]
        return max(stamps) if stamps else 0.0


def optical_rotation() -> np.ndarray:
    """REP-103: camera_link (x fwd, y left, z up) -> camera_optical (x right, y down, z fwd)."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )


def build_robot_tree(
    camera_mount: tuple[float, float, float] = (0.12, 0.0, 1.05),
    camera_pitch: float = 0.14,
    camera_yaw: float = 0.0,
    camera_roll: float = 0.0,
) -> TransformTree:
    """Build the standard map->odom->base_link->camera_link->camera_optical tree.

    ``camera_mount`` is ``(x, y, z)`` in ``base_link``; ``camera_pitch`` is the
    downward tilt in radians (positive = looking down).
    """
    from .geometry import rpy_to_matrix

    tf = TransformTree()
    tf.add_frame("map")
    tf.set_static("map", "odom")
    tf.set_static("odom", "base_link")
    tf.set_static(
        "base_link",
        "camera_link",
        translation=camera_mount,
        rotation=rpy_to_matrix(camera_roll, camera_pitch, camera_yaw),
    )
    tf.set_static("camera_link", "camera_optical", rotation=optical_rotation())
    return tf
