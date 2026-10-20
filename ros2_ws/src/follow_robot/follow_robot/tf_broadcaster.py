#!/usr/bin/env python3
"""TF broadcaster for the follow robot.

Publishes the transform chain used by the perception / control nodes:

* ``odom -> base_link``  from ``/odom`` (wheel odometry + EKF on the real robot),
* ``base_link -> camera_link`` static, from the calibrated extrinsic parameters,
* ``camera_link -> camera_optical`` static (REP-103 optical convention),
* ``base_link -> laser`` static (RPLIDAR A1 mount).

``map -> odom`` is published by AMCL when SLAM/localisation is running; this
node deliberately does not publish it (do not create two publishers for the
same transform).
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _fill(ts: TransformStamped, parent: str, child: str, xyz, rpy) -> None:
    ts.header.frame_id = parent
    ts.child_frame_id = child
    ts.transform.translation.x = xyz[0]
    ts.transform.translation.y = xyz[1]
    ts.transform.translation.z = xyz[2]
    qx, qy, qz, qw = _quat_from_rpy(*rpy)
    ts.transform.rotation.x = qx
    ts.transform.rotation.y = qy
    ts.transform.rotation.z = qz
    ts.transform.rotation.w = qw


class TfBroadcaster(Node):
    def __init__(self) -> None:
        super().__init__("tf_broadcaster")
        # calibrated extrinsics (see docs/calibration.md)
        self.declare_parameter("camera_x", 0.12)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 1.05)
        self.declare_parameter("camera_roll", 0.0)
        self.declare_parameter("camera_pitch", 0.14)  # downward tilt [rad]
        self.declare_parameter("camera_yaw", 0.0)
        self.declare_parameter("laser_x", 0.08)
        self.declare_parameter("laser_z", 0.22)
        self.declare_parameter("publish_odom", True)

        p = self.get_parameter
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.dyn_broadcaster = TransformBroadcaster(self)
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        _fill(
            ts,
            "base_link",
            "camera_link",
            (
                p("camera_x").get_parameter_value().double_value,
                p("camera_y").get_parameter_value().double_value,
                p("camera_z").get_parameter_value().double_value,
            ),
            (
                p("camera_roll").get_parameter_value().double_value,
                p("camera_pitch").get_parameter_value().double_value,
                p("camera_yaw").get_parameter_value().double_value,
            ),
        )
        ts2 = TransformStamped()
        ts2.header.stamp = ts.header.stamp
        _fill(ts2, "camera_link", "camera_optical", (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, -math.pi / 2))
        ts3 = TransformStamped()
        ts3.header.stamp = ts.header.stamp
        _fill(
            ts3,
            "base_link",
            "laser",
            (
                p("laser_x").get_parameter_value().double_value,
                0.0,
                p("laser_z").get_parameter_value().double_value,
            ),
            (0.0, 0.0, 0.0),
        )
        self.static_broadcaster.sendTransform([ts, ts2, ts3])
        if p("publish_odom").get_parameter_value().bool_value:
            self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.get_logger().info("tf_broadcaster ready (static extrinsics published)")

    def on_odom(self, msg: Odometry) -> None:
        ts = TransformStamped()
        ts.header.stamp = msg.header.stamp
        ts.header.frame_id = "odom"
        ts.child_frame_id = "base_link"
        ts.transform.translation.x = msg.pose.pose.position.x
        ts.transform.translation.y = msg.pose.pose.position.y
        ts.transform.translation.z = 0.0
        ts.transform.rotation = msg.pose.pose.orientation
        self.dyn_broadcaster.sendTransform(ts)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TfBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
