#!/usr/bin/env python3
"""Follow controller node: PID + Kalman + ReID state machine -> /cmd_vel.

Subscribes
----------
/follow/target_world       geometry_msgs/PointStamped   target position in map
/follow/candidates         std_msgs/String              JSON from perception_node
/follow/target_visible     std_msgs/Bool
/follow/target_track_id    std_msgs/Int32
/odom                      nav_msgs/Odometry           fallback if TF is missing

Publishes
---------
/cmd_vel                   geometry_msgs/Twist      differential-drive command
/follow/state              std_msgs/String          TRACK / OCCLUDED / REACQUIRE / ...
/follow/status             std_msgs/String          JSON telemetry (used by bridges)

Uses Nav2's ``NavigateToPose`` action for the SEARCH behaviour: when the target
has been lost, the controller first spins; after ``search_timeout`` it asks Nav2
to drive to the last known target position / patrol point instead of blindly
rotating forever.

The node was NOT executed on the development machine (no ROS2 there); see
``docs/ros2_integration.md`` for the Ubuntu 24.04 + Jazzy + Gazebo Harmonic
bring-up and ``docs/real_robot.md`` for the Raspberry Pi 5 deployment.
"""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from follow_robot.control_core import FollowParams, FollowState, FollowStateMachine


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class FollowController(Node):
    """Thin ROS2 wrapper around :class:`FollowStateMachine`."""

    def __init__(self) -> None:
        super().__init__("follow_controller")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("desired_distance", 1.40)
        self.declare_parameter("max_v", 0.62)
        self.declare_parameter("max_w", 1.20)
        self.declare_parameter("reid_high", 0.78)
        self.declare_parameter("reid_low", 0.52)
        self.declare_parameter("reid_consecutive", 3)
        self.declare_parameter("occluded_timeout", 3.2)
        self.declare_parameter("reacquire_timeout", 6.0)
        self.declare_parameter("use_nav2_action", True)

        p = self.get_parameter
        params = FollowParams(
            desired_distance=p("desired_distance").get_parameter_value().double_value,
            max_v=p("max_v").get_parameter_value().double_value,
            max_w=p("max_w").get_parameter_value().double_value,
            reid_high=p("reid_high").get_parameter_value().double_value,
            reid_low=p("reid_low").get_parameter_value().double_value,
            reid_consecutive=p("reid_consecutive").get_parameter_value().integer_value,
            occluded_timeout=p("occluded_timeout").get_parameter_value().double_value,
            reacquire_timeout=p("reacquire_timeout").get_parameter_value().double_value,
        )
        self.fsm = FollowStateMachine(params)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_world: tuple[float, float] | None = None
        self.target_visible = False
        self.target_track_id: int | None = None
        self.candidates: list[dict] = []
        self.last_candidates_stamp = 0.0
        self.last_known_world: tuple[float, float] | None = None
        self.use_nav2_action = p("use_nav2_action").get_parameter_value().bool_value
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose") if self.use_nav2_action else None
        self._nav_goal_active = False

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "/follow/state", 10)
        self.status_pub = self.create_publisher(String, "/follow/status", 10)
        self.create_subscription(PointStamped, "/follow/target_world", self.on_target_world, 10)
        self.create_subscription(Bool, "/follow/target_visible", self.on_visible, 10)
        self.create_subscription(Int32, "/follow/target_track_id", self.on_track_id, 10)
        self.create_subscription(String, "/follow/candidates", self.on_candidates, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)

        rate = p("control_rate").get_parameter_value().double_value
        self.create_timer(1.0 / max(1.0, rate), self.on_timer)
        self.get_logger().info("follow_controller ready")

    # ------------------------------------------------------------------
    def on_target_world(self, msg: PointStamped) -> None:
        self.target_world = (msg.point.x, msg.point.y)
        self.last_known_world = self.target_world

    def on_visible(self, msg: Bool) -> None:
        self.target_visible = msg.data

    def on_track_id(self, msg: Int32) -> None:
        self.target_track_id = None if msg.data < 0 else int(msg.data)

    def on_candidates(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.candidates = payload.get("candidates", [])

    def on_odom(self, msg: Odometry) -> None:
        # fallback robot pose if TF map->base_link is not available
        self._odom_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            _yaw_from_quaternion(msg.pose.pose.orientation),
        )

    # ------------------------------------------------------------------
    def _robot_pose(self) -> tuple[float, float, float] | None:
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except TransformException:
            return getattr(self, "_odom_pose", None)
        t = tf.transform.translation
        return (t.x, t.y, _yaw_from_quaternion(tf.transform.rotation))

    def _publish_cmd(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def _maybe_send_nav_goal(self) -> None:
        if self.nav_client is None or self._nav_goal_active:
            return
        if self.last_known_world is None or not self.nav_client.wait_for_server(timeout_sec=0.1):
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x = self.last_known_world[0]
        goal.pose.pose.position.y = self.last_known_world[1]
        goal.pose.pose.orientation.w = 1.0
        self._nav_goal_active = True
        send_future = self.nav_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_nav_goal_sent)

    def _on_nav_goal_sent(self, future) -> None:  # pragma: no cover - runtime only
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 search goal rejected")
            self._nav_goal_active = False
            return
        goal_handle.get_result_async().add_done_callback(lambda _f: setattr(self, "_nav_goal_active", False))

    # ------------------------------------------------------------------
    def on_timer(self) -> None:
        pose = self._robot_pose()
        if pose is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.1
        target_world = self.target_world if self.target_visible else None

        candidates = self.candidates if self.candidates else []
        reid_similarity = [float(c.get("similarity", 0.0)) for c in candidates]
        gate_ok = []
        predicted = self.fsm.kalman.position if self.fsm.kalman.initialized else None
        seconds_since = (now - self.fsm.last_seen) if self.fsm.last_seen is not None else 0.0
        gate_radius = self.fsm.p.motion_gate_base + self.fsm.p.motion_gate_growth * max(0.0, seconds_since)
        for c in candidates:
            world = c.get("world")
            gate = (
                predicted is not None
                and world is not None
                and math.hypot(world[0] - predicted[0], world[1] - predicted[1]) <= gate_radius
            )
            gate_ok.append(bool(gate))

        candidate_pairs = [
            (int(c["track_id"]), (float(c["world"][0]), float(c["world"][1])))
            for c in candidates
            if c.get("world") is not None and not c.get("similarity", 0.0) >= 1.0
        ]
        result = self.fsm.step(
            now=now,
            dt=dt,
            robot_pose=pose,
            target_world=target_world,
            candidates=candidate_pairs,
            reid_similarity=reid_similarity,
            gate_ok=gate_ok,
            target_track_id=self.target_track_id,
        )
        self._publish_cmd(result["v"], result["w"])
        self.state_pub.publish(String(data=result["state"].value))
        if result["state"] == FollowState.SEARCH:
            self._maybe_send_nav_goal()
        if result["state"] == FollowState.STOP:
            self._publish_cmd(0.0, 0.0)
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "state": result["state"].value,
                        "v": round(result["v"], 3),
                        "w": round(result["w"], 3),
                        "distance": result.get("distance"),
                        "bearing": result.get("bearing"),
                        "target_visible": self.target_visible,
                        "target_track_id": self.target_track_id,
                        "candidates": len(candidates),
                        "reacquire_attempts": self.fsm.reacquire_attempts,
                        "reacquire_success": self.fsm.reacquire_success,
                        "pose": [round(v, 3) for v in pose],
                    }
                )
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FollowController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
