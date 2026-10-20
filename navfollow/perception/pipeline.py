"""Perception pipeline: detect -> track -> ReID -> depth -> TF -> target observation.

This is the software equivalent of ``perception_node.py`` in the ROS2 package.
It consumes a synchronised RGB-D frame (with timestamps), performs the same
steps a real node would:

1. drop the frame if RGB/depth timestamps are too far apart,
2. look up ``camera_optical -> base_link`` from TF at the *image* timestamp,
3. undistort RGB-D with the calibration,
4. detect persons, associate them with tracks (ByteTrack-style),
5. extract ReID descriptors and record a bounded gallery for the locked target,
6. sample depth inside each bbox, deproject and transform to ``base_link`` and
   ``map``, yielding distance / bearing for the follower.

Every frame carries the TF timestamp so the failure case "stale transform"
can be reproduced and tested (see ``docs/failure_cases.md``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..camera import CameraIntrinsics, undistort_rgbd
from ..geometry import Pose2D, quaternion_from_yaw
from ..tf_tree import ExtrapolationError, FrameNotFound, TransformError, TransformTree, build_robot_tree
from .detector import ColorPersonDetector, Detection, Detector
from .projection import base_observation_from_bbox, transform_points
from .reid import ColorHistogramReID, ReIDConfig, ReIDGallery, motion_gate
from .tracker import IoUTracker, Track, TrackerOutput


@dataclass
class TrackedDetection:
    """A detection with its track id and metric observations."""

    detection: Detection
    track_id: int
    feature: np.ndarray | None = None
    depth: float | None = None
    base_xy: np.ndarray | None = None
    world_xy: tuple[float, float] | None = None
    distance: float | None = None
    bearing: float | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.detection.bbox

    @property
    def score(self) -> float:
        return self.detection.score

    @property
    def valid_geometry(self) -> bool:
        return self.world_xy is not None and self.distance is not None and self.bearing is not None


@dataclass
class FrameResult:
    stamp: float
    detections: list[TrackedDetection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    target: TrackedDetection | None = None
    target_track_id: int | None = None
    dropped: bool = False
    drop_reason: str | None = None
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None

    @property
    def target_visible(self) -> bool:
        return self.target is not None and self.target.valid_geometry


class PerceptionPipeline:
    """RGB-D perception front-end for the follower."""

    def __init__(
        self,
        detector: Detector | None = None,
        tracker: IoUTracker | None = None,
        reid: ColorHistogramReID | None = None,
        intrinsics: CameraIntrinsics | None = None,
        tf: TransformTree | None = None,
        time_tolerance: float = 0.05,
        reid_config: ReIDConfig | None = None,
        undistort: bool = True,
    ) -> None:
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.detector = detector or ColorPersonDetector()
        self.tracker = tracker or IoUTracker()
        self.reid_config = reid_config or ReIDConfig()
        self.reid = reid or ColorHistogramReID(self.reid_config)
        self.gallery = ReIDGallery(reid=self.reid)
        self.tf = tf or build_robot_tree()
        self.time_tolerance = time_tolerance
        self.undistort = undistort
        self.target_track_id: int | None = None
        self.last_stamp: float | None = None
        self.frames_processed = 0
        self.frames_dropped = 0
        self.drop_reasons: dict[str, int] = {}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.tracker.reset()
        self.gallery.clear()
        self.target_track_id = None
        self.last_stamp = None
        self.frames_processed = 0
        self.frames_dropped = 0
        self.drop_reasons = {}

    # ------------------------------------------------------------------
    def _drop(self, stamp: float, reason: str) -> FrameResult:
        self.frames_dropped += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        return FrameResult(stamp=stamp, dropped=True, drop_reason=reason, target_track_id=self.target_track_id)

    def _update_tf(self, robot_pose: Pose2D, stamp: float) -> None:
        """Publish the simulated AMCL/odometry transform for this frame."""
        self.tf.set_transform(
            "odom",
            "base_link",
            translation=(robot_pose.x, robot_pose.y, 0.0),
            quaternion=quaternion_from_yaw(robot_pose.theta),
            stamp=stamp,
        )

    # ------------------------------------------------------------------
    def lock_target(self, tracked: TrackedDetection) -> None:
        self.target_track_id = tracked.track_id
        self.gallery.add(tracked.feature)

    def lock_track(self, track_id: int, feature: np.ndarray | None = None) -> None:
        """Re-bind the locked target to a new track id after ReID re-acquisition."""
        self.target_track_id = track_id
        self.gallery.add(feature)

    def lock_target_by_world(self, world_xy: tuple[float, float], detections: list[TrackedDetection]) -> bool:
        """Initial target selection ("click the person on screen"), by position."""
        candidates = [d for d in detections if d.valid_geometry]
        if not candidates:
            return False
        best = min(
            candidates,
            key=lambda d: math.hypot(d.world_xy[0] - world_xy[0], d.world_xy[1] - world_xy[1]),
        )
        self.lock_target(best)
        return True

    def similarity_to_target(self, feature: np.ndarray | None) -> float:
        return self.gallery.best_similarity(feature)

    def motion_gate(
        self,
        predicted_world: tuple[float, float] | None,
        detection: TrackedDetection,
        seconds_since_seen: float,
    ) -> bool:
        if detection.world_xy is None:
            return False
        return motion_gate(predicted_world, detection.world_xy, seconds_since_seen, self.reid_config)

    # ------------------------------------------------------------------
    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        rgb_stamp: float,
        depth_stamp: float | None = None,
        robot_pose: Pose2D | None = None,
    ) -> FrameResult:
        depth_stamp = rgb_stamp if depth_stamp is None else depth_stamp
        # ---- time synchronisation --------------------------------------
        if abs(rgb_stamp - depth_stamp) > self.time_tolerance:
            return self._drop(rgb_stamp, "depth_rgb_timestamp_mismatch")
        dt = 0.1 if self.last_stamp is None else max(1e-3, rgb_stamp - self.last_stamp)
        self.last_stamp = rgb_stamp

        # ---- transforms at the image stamp -----------------------------
        if robot_pose is not None:
            self._update_tf(robot_pose, rgb_stamp)
        try:
            T_base_optical = self.tf.lookup("base_link", "camera_optical", rgb_stamp)
            T_map_base = self.tf.lookup("map", "base_link", rgb_stamp)
        except (FrameNotFound, ExtrapolationError):
            return self._drop(rgb_stamp, "tf_unavailable")
        except TransformError:
            return self._drop(rgb_stamp, "tf_disconnected")

        # ---- undistort + detect ---------------------------------------
        if self.undistort:
            rgb_u, depth_u = undistort_rgbd(rgb, depth, self.intrinsics)
        else:
            rgb_u, depth_u = rgb, depth
        detections = self.detector.detect(rgb_u)
        tracker_out: TrackerOutput = self.tracker.update(detections, dt)

        # ---- metric observations --------------------------------------
        # every track that received a detection in *this* frame becomes an
        # observation (matched tracks and newly created tracks alike)
        tracked: list[TrackedDetection] = []
        for track in tracker_out.tracks:
            if track.time_since_update > 1e-9 or track.last_detection is None:
                continue
            det = track.last_detection
            td = TrackedDetection(detection=det, track_id=track.track_id)
            td.feature = self.reid.extract(rgb_u, det.bbox)
            observation = base_observation_from_bbox(det.bbox, depth_u, self.intrinsics.K, T_base_optical)
            if observation is not None:
                distance, bearing, base_xyz = observation
                td.distance = distance
                td.bearing = bearing
                td.base_xy = base_xyz
                td.depth = float(observation[0])
                world = transform_points(T_map_base, base_xyz)[0]
                td.world_xy = (float(world[0]), float(world[1]))
            tracked.append(td)
            track.feature = td.feature if td.feature is not None else track.feature

        # ---- target bookkeeping ---------------------------------------
        target: TrackedDetection | None = None
        if self.target_track_id is not None:
            for td in tracked:
                if td.track_id == self.target_track_id:
                    target = td
                    break
            if target is not None:
                self.gallery.add(target.feature)

        self.frames_processed += 1
        return FrameResult(
            stamp=rgb_stamp,
            detections=tracked,
            tracks=tracker_out.tracks,
            target=target,
            target_track_id=self.target_track_id,
            rgb=rgb_u,
            depth=depth_u,
        )
