"""Rendering utilities for the demo GIFs.

Two renderers are used:

* :class:`TopDownRenderer` - a 2-D top-down view of the world with the costmap
  inflation layer, the global plan, lidar hits, the robot and the pedestrians.
* :func:`camera_panel` / :func:`depth_panel` - the synthetic RGB-D camera image
  used by the perception pipeline, with detection boxes and the depth map.

The demo scripts sample these frames during *real* simulation runs (the same
code that produces the evaluation numbers), then assemble them into GIFs with
:func:`save_gif`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .maps import GridMap


class TopDownRenderer:
    """Top-down world renderer at a fixed scale."""

    def __init__(self, grid: GridMap, scale: int = 22, show_inflation: bool = True) -> None:
        self.grid = grid
        self.scale = scale
        self.width = int(round(grid.size_m[0] * scale))
        self.height = int(round(grid.size_m[1] * scale))
        self._base = self._build_base()
        self._inflation = self._build_inflation() if show_inflation else None

    # ------------------------------------------------------------------
    def w2p(self, x: float, y: float) -> tuple[int, int]:
        px = int(round((x - self.grid.origin[0]) * self.scale))
        py = int(round(self.height - (y - self.grid.origin[1]) * self.scale))
        return px, py

    def _build_base(self) -> np.ndarray:
        occ = self.grid.occupancy.astype(np.uint8)
        occ = cv2.resize(occ, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        floor = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        floor[:] = (206, 204, 196)
        # faint 1 m grid
        for gx in range(0, int(self.grid.size_m[0]) + 1):
            px, _ = self.w2p(float(gx), 0.0)
            if 0 <= px < self.width:
                floor[:, px] = (192, 190, 182)
        for gy in range(0, int(self.grid.size_m[1]) + 1):
            _, py = self.w2p(0.0, float(gy))
            if 0 <= py < self.height:
                floor[py, :] = (192, 190, 182)
        walls = occ.astype(bool)
        floor[walls] = (72, 74, 82)
        return floor

    def _build_inflation(self) -> np.ndarray:
        self.grid.ensure_derived()
        margin = self.grid.blocked_margin.astype(np.uint8)
        core = self.grid.blocked_robot.astype(np.uint8)
        margin = cv2.resize(margin, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        core = cv2.resize(core, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        overlay[margin.astype(bool)] = (255, 214, 158)
        overlay[core.astype(bool)] = (232, 120, 96)
        # soft layers so the map stays readable
        weights = np.zeros((self.height, self.width, 1), dtype=np.float32)
        weights[margin.astype(bool)] = 0.18
        weights[core.astype(bool)] = 0.40
        self._inflation_weights = weights
        return overlay

    # ------------------------------------------------------------------
    def render(
        self,
        pose=None,
        persons: list | None = None,
        path: list[tuple[float, float]] | None = None,
        planned_paths: list[list[tuple[float, float]]] | None = None,
        lidar_scan: np.ndarray | None = None,
        lidar_angles: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
        target_xy: tuple[float, float] | None = None,
        predicted_xy: tuple[float, float] | None = None,
        captions: list[str] | None = None,
        target_person_id: int | None = 1,
        flash: bool = False,
    ) -> np.ndarray:
        frame = self._base.copy()
        if self._inflation is not None:
            mask = (self._inflation > 0).any(axis=2)
            w = self._inflation_weights[mask]
            frame[mask] = ((1.0 - w) * frame[mask] + w * self._inflation[mask]).astype(np.uint8)
        # planned paths (history in grey, current in cyan)
        if planned_paths:
            for old in planned_paths[:-1]:
                self._polyline(frame, old, (170, 170, 170), 1)
        if path:
            self._polyline(frame, path, (0, 185, 235), 2)
            for i in range(0, len(path), 3):
                px, py = self.w2p(*path[i])
                cv2.circle(frame, (px, py), 2, (0, 150, 210), -1)
        if goal is not None:
            gx, gy = self.w2p(*goal)
            cv2.drawMarker(frame, (gx, gy), (40, 170, 60), cv2.MARKER_CROSS, 14, 2)
        # lidar hits
        if lidar_scan is not None:
            angle_min, angle_max = lidar_angles if lidar_angles else (-2.35, 2.35)
            angles = pose.theta + np.linspace(angle_min, angle_max, len(lidar_scan))
            pts = np.stack(
                [
                    pose.x + lidar_scan * np.cos(angles),
                    pose.y + lidar_scan * np.sin(angles),
                ],
                axis=1,
            )
            for x, y in pts:
                px, py = self.w2p(float(x), float(y))
                if 0 <= px < self.width and 0 <= py < self.height:
                    frame[py, px] = (90, 200, 120)
        # persons
        for person in persons or []:
            if getattr(person, "hidden", False):
                continue
            px, py = self.w2p(person.x, person.y)
            radius = max(3, int(round(person.radius * self.scale)))
            color = person.color_rgb
            cv2.circle(frame, (px, py), radius, color, -1, lineType=cv2.LINE_AA)
            if target_person_id is not None and getattr(person, "person_id", None) == target_person_id:
                cv2.circle(frame, (px, py), radius + 3, (255, 215, 0), 2, lineType=cv2.LINE_AA)
        # helper lines: robot -> observed target / prediction
        if pose is not None and target_xy is not None:
            self._line(frame, (pose.x, pose.y), target_xy, (255, 120, 160), 1)
        if pose is not None and predicted_xy is not None:
            self._line(frame, (pose.x, pose.y), predicted_xy, (255, 170, 60), 1)
            px, py = self.w2p(*predicted_xy)
            cv2.drawMarker(frame, (px, py), (255, 170, 60), cv2.MARKER_DIAMOND, 10, 1)
        # robot
        if pose is not None:
            self._draw_robot(frame, pose)
        if flash:
            frame = (0.55 * frame + 0.45 * np.array([255, 120, 120], dtype=float)).astype(np.uint8)
        if captions:
            self._draw_captions(frame, captions)
        return frame

    # ------------------------------------------------------------------
    def _polyline(self, frame: np.ndarray, path: list[tuple[float, float]], color, thickness: int) -> None:
        pts = np.array([self.w2p(x, y) for x, y in path], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(frame, [pts], False, color, thickness, lineType=cv2.LINE_AA)

    def _line(self, frame, a, b, color, thickness: int) -> None:
        cv2.line(frame, self.w2p(*a), self.w2p(*b), color, thickness, lineType=cv2.LINE_AA)

    def _draw_robot(self, frame: np.ndarray, pose) -> None:
        px, py = self.w2p(pose.x, pose.y)
        radius = max(6, int(round(0.24 * self.scale)))
        cv2.circle(frame, (px, py), radius + 2, (35, 55, 120), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (90, 150, 240), -1, lineType=cv2.LINE_AA)
        tip = (
            int(px + math.cos(pose.theta) * (radius + 7)),
            int(py - math.sin(pose.theta) * (radius + 7)),
        )
        left = (
            int(px + math.cos(pose.theta + 2.3) * (radius - 1)),
            int(py - math.sin(pose.theta + 2.3) * (radius - 1)),
        )
        right = (
            int(px + math.cos(pose.theta - 2.3) * (radius - 1)),
            int(py - math.sin(pose.theta - 2.3) * (radius - 1)),
        )
        tri = np.array([tip, left, right], dtype=np.int32)
        cv2.fillPoly(frame, [tri], (250, 252, 255))
        cv2.polylines(frame, [tri], True, (25, 35, 65), 1)

    def _draw_captions(self, frame: np.ndarray, captions: list[str]) -> None:
        y = 15
        for caption in captions[:2]:
            cv2.putText(frame, caption, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 33, 40), 1, cv2.LINE_AA)
            y += 14


# ---------------------------------------------------------------------------
# camera / depth panels
# ---------------------------------------------------------------------------
def camera_panel(
    rgb: np.ndarray,
    detections: list | None = None,
    target_track_id: int | None = None,
    scale: int = 2,
) -> np.ndarray:
    """Upscale the RGB camera image and draw detection boxes."""
    img = rgb.copy()
    for det in detections or []:
        x1, y1, x2, y2 = det.bbox
        is_target = det.track_id == target_track_id
        color = (40, 220, 60) if is_target else (250, 190, 60)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"id{det.track_id}"
        cv2.putText(img, label, (x1, max(10, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)


def depth_panel(depth: np.ndarray, scale: int = 2, max_depth: float = 7.0) -> np.ndarray:
    """Colourise the depth image (0 = invalid -> black)."""
    d = np.clip(depth / max_depth * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(d, cv2.COLORMAP_JET)
    colored[depth <= 0] = (0, 0, 0)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.resize(colored, (int(colored.shape[1] * scale), int(colored.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)


def hstack_panels(panels: list[np.ndarray], gap: int = 6, background: int = 24) -> np.ndarray:
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = 0
    for panel in panels:
        canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gap
    return canvas


def vstack_panels(panels: list[np.ndarray], gap: int = 6, background: int = 24) -> np.ndarray:
    width = max(p.shape[1] for p in panels)
    height = sum(p.shape[0] for p in panels) + gap * (len(panels) - 1)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    y = 0
    for panel in panels:
        canvas[y : y + panel.shape[0], : panel.shape[1]] = panel
        y += panel.shape[0] + gap
    return canvas


def add_header(frame: np.ndarray, lines: list[str], height: int = 26) -> np.ndarray:
    header = np.full((height, frame.shape[1], 3), 18, dtype=np.uint8)
    x = 8
    for line in lines:
        cv2.putText(header, line, (x, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 235, 240), 1, cv2.LINE_AA)
        x += int(9 * len(line) * 0.44) + 22
    return np.vstack([header, frame])


def status_panel(
    width: int,
    lines: list[tuple[str, tuple[int, int, int]]],
    height: int = 92,
    background: int = 18,
) -> np.ndarray:
    """Dark status panel with one coloured line per entry (state, metrics)."""
    panel = np.full((height, width, 3), background, dtype=np.uint8)
    y = 20
    for text, color in lines:
        cv2.putText(panel, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22
    return panel


# ---------------------------------------------------------------------------
# GIF writing
# ---------------------------------------------------------------------------
def save_gif(
    frames: list[np.ndarray],
    path: str,
    fps: int = 8,
    max_bytes: int = 3_500_000,
    colors: int = 96,
    verbose: bool = True,
) -> dict:
    """Save frames as an optimised GIF, shrinking until it fits ``max_bytes``.

    Returns a dict with the final size, frame count, scale and colour count.
    """
    import os

    if not frames:
        raise ValueError("no frames to save")
    duration = max(20, int(round(1000.0 / fps)))
    attempts = [
        (1.0, colors),
        (1.0, 64),
        (0.85, 64),
        (0.7, 48),
        (0.6, 32),
    ]
    last_size = None
    for scale, n_colors in attempts:
        out_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if scale < 1.0:
                new_size = (max(2, int(frame.shape[1] * scale)), max(2, int(frame.shape[0] * scale)))
                img = img.resize(new_size, Image.BILINEAR)
            out_frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=n_colors))
        out_frames[0].save(
            path,
            save_all=True,
            append_images=out_frames[1:],
            duration=duration,
            loop=0,
            optimize=True,
            disposal=2,
        )
        size = os.path.getsize(path)
        last_size = size
        if verbose:
            print(f"  gif {os.path.basename(path)}: scale={scale} colors={n_colors} "
                  f"frames={len(frames)} size={size/1e6:.2f} MB")
        if size <= max_bytes:
            return {"path": path, "bytes": size, "frames": len(frames), "scale": scale, "colors": n_colors}
    if verbose:
        print(f"  WARNING: {path} still {last_size/1e6:.2f} MB after all attempts")
    return {"path": path, "bytes": last_size, "frames": len(frames), "scale": attempts[-1][0], "colors": attempts[-1][1]}
