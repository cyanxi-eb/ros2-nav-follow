#!/usr/bin/env python3
"""WebSocket telemetry bridge: pushes robot status to a browser dashboard.

A tiny ``websockets`` server runs in a daemon thread and broadcasts the JSON
status received on ``/follow/status``.  ``websockets`` is an optional runtime
dependency::

    pip install websockets

The server binds to ``127.0.0.1`` by default; set ``host=0.0.0.0`` to expose it
on the robot LAN (add authentication / TLS in front of it for anything public).
"""
from __future__ import annotations

import asyncio
import json
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WsBridge(Node):
    def __init__(self) -> None:
        super().__init__("ws_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 8765)
        self.declare_parameter("rate_hz", 5.0)
        p = self.get_parameter
        self.host = p("host").get_parameter_value().string_value
        self.port = p("port").get_parameter_value().integer_value
        self.clients: set = set()
        self._latest: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self.create_subscription(String, "/follow/status", self.on_status, 10)
        rate = max(0.5, p("rate_hz").get_parameter_value().double_value)
        self.create_timer(1.0 / rate, self.broadcast)
        self.get_logger().info(f"ws_bridge ready on ws://{self.host}:{self.port}")

    # ------------------------------------------------------------------
    def _run_server(self) -> None:
        try:
            import websockets
        except ImportError:
            self.get_logger().warn("websockets not installed; ws_bridge idle")
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(websocket) -> None:
            self.clients.add(websocket)
            try:
                while True:
                    await websocket.wait_closed()
            finally:
                self.clients.discard(websocket)

        async def serve() -> None:
            async with websockets.serve(handler, self.host, self.port):
                self.get_logger().info("websocket server listening")
                await asyncio.Future()

        self._loop.run_until_complete(serve())

    def on_status(self, msg: String) -> None:
        self._latest = msg.data

    def broadcast(self) -> None:
        if self._latest is None or self._loop is None or not self.clients:
            return
        payload = self._latest

        async def send() -> None:
            dead = []
            for client in list(self.clients):
                try:
                    await client.send(payload)
                except Exception:
                    dead.append(client)
            for client in dead:
                self.clients.discard(client)

        asyncio.run_coroutine_threadsafe(send(), self._loop)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WsBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
