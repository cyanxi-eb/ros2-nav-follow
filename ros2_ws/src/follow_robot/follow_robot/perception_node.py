#!/usr/bin/env python3
"""Perception node: RGB-D -> person detection -> tracking/ReID -> target in base_link.

Subscribes
----------
/camera/color/image_raw     sensor_msgs/Image        RGB (mono8/bgr8 accepted)
/camera/depth/image_raw     sensor_msgs/Image        16UC1 mm or 32FC1 m
/camera/color/camera_info   sensor_msgs/CameraInfo   intrinsics (+ distortion)
/follow/select_target       std_msgs/Bool            lock the largest detection as target

Publishes
---------
/follow/target_pose         geometry_msgs/PoseStamped   target in base_link (x fwd)
/follow/target_world        geometry_msgs/PointStamped  target in map
/follow/target_visible      std_msgs/Bool
/follow/target_track_id     std_msgs/Int32
/follow/reid_feature        std_msgs/Float32MultiArray  appearance descriptor
/follow/candidates          std_msgs/String            JSON for the controller
/follow/debug_image         sensor_msgs/Image            annotated RGB
/follow/status              std_msgs/String             JSON diagnostics

The node is documented as "not executed on the development machine" (no ROS2
installed there); it is the deployment counterpart of the offline pipeline in
``navfollow/perception``.
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from follow_robot.perception_core import (
    ByteTrackTracker,
    ColorHistogramReID,
    ColorPersonDetector,
    YoloV8PersonDetector,
    bbox_depth,
    bearing_range_from_base_point,
    deproject_pixel,
)


class PerceptionNode(Node):
    """Detect / track / re-identify the person and publish a metric target."""

    def __init__(self) -> None:
        super().__init__("perception_node")
        self.declare_parameter("detector", "yolo")            # yolo | color
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("depth_units", 0.001)           # 16UC1 -> metres
        self.declare_parameter("sync_slop", 0.05)              # seconds
        self.declare_parameter("max_distance", 6.0)            # metres
        self.declare_parameter("reid_high", 0.78)
        self.declare_parameter("reid_low", 0.52)
        self.declare_parameter("undistort", True)

        p = self.get_parameter
        detector_type = p("detector").get_parameter_value().string_value
        if detector_type == "color":
            self.detector = ColorPersonDetector()
        else:
            self.detector = YoloV8PersonDetector(
                model_path=p("model_path").get_parameter_value().string_value,
                conf=p("conf").get_parameter_value().double_value,
                device=p("device").get_parameter_value().string_value,
            )
        self.tracker = ByteTrackTracker()
        self.reid = ColorHistogramReID(
            high=p("reid_high").get_parameter_value().double_value,
            low=p("reid_low").get_parameter_value().double_value,
        )
        self.depth_scale = p("depth_units").get_parameter_value().double_value
        self.max_distance = p("max_distance").get_parameter_value().double_value
        self.undistort = p("undistort").get_parameter_value().bool_value
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_track_id: int | None = None
        self._pending_lock = False
        self.last_stamp: float | None = None
        self.camera_info: CameraInfo | None = None

        self.pose_pub = self.create_publisher(PoseStamped, "/follow/target_pose", 10)
        self.world_pub = self.create_publisher(PointStamped, "/follow/target_world", 10)
        self.visible_pub = self.create_publisher(Bool, "/follow/target_visible", 10)
        self.track_pub = self.create_publisher(Int32, "/follow/target_track_id", 10)
        self.feature_pub = self.create_publisher(Float32MultiArray, "/follow/reid_feature", 10)
        self.candidates_pub = self.create_publisher(String, "/follow/candidates", 10)
        self.debug_pub = self.create_publisher(Image, "/follow/debug_image", 10)
        self.status_pub = self.create_publisher(String, "/follow/status", 10)

        self.create_subscription(Bool, "/follow/select_target", self.on_select_target, 10)

        slop = p("sync_slop").get_parameter_value().double_value
        self.rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/depth/image_raw")
        self.info_sub = Subscriber(self, CameraInfo, "/camera/color/camera_info")
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], queue_size=10, slop=slop
        )
        self.sync.registerCallback(self.on_rgbd)
        self.get_logger().info(f"perception_node ready (detector={detector_type})")

    # ------------------------------------------------------------------
    def on_select_target(self, msg: Bool) -> None:
        """Operator click: lock the current/largest detection as the target."""
        if msg.data:
            self._pending_lock = True

    def _lookup(self, target: str, source: str, stamp) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(target, source, stamp, timeout=None)
        except TransformException as exc:  # pragma: no cover - runtime only
            self.get_logger().warn(f"TF {target}<-{source} unavailable: {exc}")
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        # quaternion -> 3x3 rotation
        x, y, z, w = r.x, r.y, r.z, r.w
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (t.x, t.y, t.z)
        return T

    # ------------------------------------------------------------------
    def on_rgbd(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        stamp = rgb_msg.header.stamp
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        dt = 0.1 if self.last_stamp is None else max(1e-3, stamp_s - self.last_stamp)
        if self.last_stamp is not None and dt > 1.0:
            self.get_logger().warn(f"dropping frame, dt={dt:.2f}s (timestamp jump)")
            return
        self.last_stamp = stamp_s

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) * self.depth_scale
        else:
            depth = depth_raw.astype(np.float32)
        K = np.array(info_msg.k, dtype=float).reshape(3, 3)
        D = np.array(info_msg.d, dtype=float) if info_msg.d else np.zeros(5)

        if self.undistort and np.any(np.abs(D) > 1e-6):
            rgb = cv2.undistort(rgb, K, D)
            depth = cv2.undistort(depth, K, D)

        T_base_optical = self._lookup("base_link", "camera_optical", stamp)
        T_map_base = self._lookup("map", "base_link", stamp)
        if T_base_optical is None or T_map_base is None:
            return

        detections = self.detector.detect(rgb)
        tracks = self.tracker.update(detections, dt)

        candidates: list[dict] = []
        debug = rgb.copy()
        target_pose = None
        target_world = None
        for track in tracks:
            bbox = tuple(int(v) for v in track.bbox)
            depth_m = bbox_depth(depth, bbox, valid_max=self.max_distance)
            if depth_m is None:
                continue
            u, v = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            point_optical = deproject_pixel(u, v, depth_m, K)
            point_base = T_base_optical[:3, :3] @ point_optical + T_base_optical[:3, 3]
            distance, bearing = bearing_range_from_base_point(point_base)
            if distance > self.max_distance:
                continue
            point_map = T_map_base[:3, :3] @ point_base + T_map_base[:3, 3]
            # initial target selection: "largest person in view" (equivalent to
            # the operator clicking the person on the debug image)
            feature = self.reid.extract(rgb, bbox)
            similarity = self.reid.similarity(feature)
            if track.track_id == self.target_track_id:
                self.reid.add(feature)
                similarity = 1.0
            candidates.append(
                {
                    "track_id": int(track.track_id),
                    "distance": round(distance, 3),
                    "bearing": round(bearing, 3),
                    "world": [round(float(point_map[0]), 3), round(float(point_map[1]), 3)],
                    "similarity": round(float(similarity), 3),
                    "score": round(float(track.score), 3),
                    "area": int((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)),
                }
            )
            color = (0, 230, 0) if track.track_id == self.target_track_id else (240, 180, 40)
            cv2.rectangle(debug, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                debug,
                f"id{track.track_id} {distance:.1f}m",
                (bbox[0], max(12, bbox[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            if track.track_id == self.target_track_id:
                target_pose = (point_base, distance, bearing)
                target_world = point_map

        # initial target selection (operator click or "largest in view")
        if self._pending_lock and candidates:
            best = max(candidates, key=lambda c: c["area"])
            self.target_track_id = best["track_id"]
            self._pending_lock = False
            self.get_logger().info(f"locked target track {self.target_track_id}")

        visible = target_pose is not None
        self.visible_pub.publish(Bool(data=visible))
        self.track_pub.publish(Int32(data=-1 if self.target_track_id is None else self.target_track_id))
        self.candidates_pub.publish(String(data=json.dumps({"stamp": stamp_s, "candidates": candidates})))
        if visible:
            point_base, distance, bearing = target_pose
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = "base_link"
            msg.pose.position.x = float(point_base[0])
            msg.pose.position.y = float(point_base[1])
            msg.pose.orientation.w = 1.0
            self.pose_pub.publish(msg)
            wmsg = PointStamped()
            wmsg.header.stamp = stamp
            wmsg.header.frame_id = "map"
            wmsg.point.x = float(target_world[0])
            wmsg.point.y = float(target_world[1])
            self.world_pub.publish(wmsg)
            feature = self.reid.features[-1] if self.reid.features else None
            if feature is not None:
                self.feature_pub.publish(Float32MultiArray(data=[float(v) for v in feature]))

        if self.debug_pub.get_subscription_count():
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="rgb8"))
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "tracked": len(tracks),
                        "target_visible": visible,
                        "target_track_id": self.target_track_id,
                        "target_distance": None if target_pose is None else round(target_pose[1], 3),
                        "target_bearing": None if target_pose is None else round(target_pose[2], 3),
                        "stamp": round(stamp_s, 3),
                    }
                )
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
