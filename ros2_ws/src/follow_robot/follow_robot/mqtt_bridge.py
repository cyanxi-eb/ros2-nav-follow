#!/usr/bin/env python3
"""MQTT telemetry bridge: republishes robot status to an MQTT broker.

Subscribes ``/follow/status`` (JSON string) and publishes the same payload to
``<prefix>/state`` at a fixed rate, plus a retained ``<prefix>/online`` flag.
QoS 0 by default (telemetry is lossy-tolerant); ``paho-mqtt`` is an optional
runtime dependency::

    pip install paho-mqtt

The bridge keeps the ROS executor alive even when the broker is unreachable;
messages are dropped with a warning instead of blocking the controller.
"""
from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MqttBridge(Node):
    def __init__(self) -> None:
        super().__init__("mqtt_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 1883)
        self.declare_parameter("prefix", "follow_robot")
        self.declare_parameter("rate_hz", 2.0)
        p = self.get_parameter
        self.prefix = p("prefix").get_parameter_value().string_value
        self._latest: str | None = None
        self._client = None
        try:
            import paho.mqtt.client as mqtt  # optional dependency
        except ImportError:
            self.get_logger().warn("paho-mqtt not installed; mqtt_bridge idle")
        else:
            self._client = mqtt.Client()
            self._client.connect_async(
                p("host").get_parameter_value().string_value,
                p("port").get_parameter_value().integer_value,
                keepalive=30,
            )
            self._client.loop_start()
            self._client.publish(f"{self.prefix}/online", "1", retain=True)
        self.create_subscription(String, "/follow/status", self.on_status, 10)
        rate = max(0.2, p("rate_hz").get_parameter_value().double_value)
        self.create_timer(1.0 / rate, self.publish_state)
        self.get_logger().info(f"mqtt_bridge ready (topic {self.prefix}/state)")

    def on_status(self, msg: String) -> None:
        self._latest = msg.data

    def publish_state(self) -> None:
        if self._client is None or self._latest is None:
            return
        try:
            self._client.publish(f"{self.prefix}/state", self._latest)
        except Exception as exc:  # pragma: no cover - runtime only
            self.get_logger().warn(f"mqtt publish failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MqttBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
