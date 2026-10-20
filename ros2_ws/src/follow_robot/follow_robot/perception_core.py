"""Perception algorithms used by ``perception_node``.

This module is the ROS2-side port of ``navfollow.perception`` from the offline
simulator: the same detector interface, the same ByteTrack-style two-stage IoU
tracker, the same colour-histogram ReID gallery and the same pinhole
deprojection.  Keeping the algorithm code ROS-free makes it unit-testable and
lets the simulator and the robot share one implementation.

Units: metres, seconds, pixels (u right, v down).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# detections
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 inclusive
    score: float
    label: str = "person"
    color_name: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0] + 1) * max(0, self.bbox[3] - self.bbox[1] + 1)


class YoloV8PersonDetector:
    """Ultralytics YOLOv8n person detector (COCO class 0)."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, device: str = "cpu") -> None:
        from ultralytics import YOLO  # optional dependency

        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        results = self.model.predict(rgb, conf=self.conf, classes=[0], device=self.device, verbose=False)
        detections: list[Detection] = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(round(x2)), int(round(y2))),
                    score=float(box.conf[0]),
                )
            )
        return detections


class ColorPersonDetector:
    """HSV fallback detector (used on the Gazebo test world without weights)."""

    COLORS = {
        "red": (205, 45, 45),
        "blue": (40, 90, 200),
        "green": (40, 165, 70),
        "orange": (225, 135, 35),
    }

    def __init__(self, min_area: int = 120) -> None:
        self.min_area = min_area
        self._ranges: dict[str, list] = {}
        for name, rgb in self.COLORS.items():
            h, s, v = cv2.cvtColor(np.uint8([[list(rgb)]]), cv2.COLOR_RGB2HSV)[0, 0]
            lo = int(h) - 10
            hi = int(h) + 10
            sat = max(60, int(s) - 60)
            ranges = []
            if lo < 0:
                ranges.append(((0, sat, 45), (hi, 255, 255)))
                ranges.append(((180 + lo, sat, 45), (179, 255, 255)))
            else:
                ranges.append(((lo, sat, 45), (hi, 255, 255)))
            self._ranges[name] = ranges

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        out: list[Detection] = []
        for name, ranges in self._ranges.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                out.append(
                    Detection(
                        bbox=(x, y, x + w - 1, y + h - 1),
                        score=float(min(0.99, 0.35 + cv2.contourArea(contour) / 1600.0)),
                        color_name=name,
                    )
                )
        out.sort(key=lambda d: d.area, reverse=True)
        return out


# ---------------------------------------------------------------------------
# tracking + ReID
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    hits: int = 1
    age: float = 0.0
    time_since_update: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    feature: np.ndarray | None = None


class ByteTrackTracker:
    """Two-stage IoU association (high-score detections first, then low-score)."""

    def __init__(self, max_age: float = 0.5, high: float = 0.55, low: float = 0.40) -> None:
        self.max_age = max_age
        self.high = high
        self.low = low
        self.tracks: list[Track] = []
        self._next_id = 1

    def predict(self, dt: float) -> None:
        for t in self.tracks:
            t.age += dt
            t.time_since_update += dt
            t.bbox = t.bbox + t.velocity * dt

    def _match(self, dets: list[Detection], thr: float) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        if not self.tracks or not dets:
            return [], list(self.tracks), list(dets)
        matched: list[tuple[Track, Detection]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        # greedy match by IoU (Hungarian would be used for > 3 people)
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(dets):
                iou = _iou(track.bbox, det.bbox)
                if iou >= thr:
                    pairs.append((iou, ti, di))
        for iou, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matched.append((self.tracks[ti], dets[di]))
        rem_t = [t for i, t in enumerate(self.tracks) if i not in used_t]
        rem_d = [d for i, d in enumerate(dets) if i not in used_d]
        return matched, rem_t, rem_d

    def update(self, detections: list[Detection], dt: float) -> list[Track]:
        self.predict(dt)
        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low <= d.score < self.high]
        matched_high, rem_tracks, rem_high = self._match(high, 0.2)
        for track, det in matched_high:
            self._apply(track, det, dt)
        matched_low, rem_tracks, _rem_low = self._match(low, 0.1)
        for track, det in matched_low:
            self._apply(track, det, dt)
        for det in rem_high:
            track = Track(track_id=self._next_id, bbox=np.array(det.bbox, float), score=det.score)
            self._next_id += 1
            self.tracks.append(track)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.time_since_update == 0.0]

    def _apply(self, track: Track, det: Detection, dt: float) -> None:
        new_bbox = np.array(det.bbox, float)
        if dt > 1e-9:
            track.velocity = 0.5 * track.velocity + 0.5 * (new_bbox - track.bbox) / dt
        track.bbox = new_bbox
        track.score = det.score
        track.hits += 1
        track.time_since_update = 0.0


class ColorHistogramReID:
    """HSV jacket-histogram appearance descriptor and dual-threshold gallery."""

    def __init__(self, high: float = 0.78, low: float = 0.52, consecutive: int = 3) -> None:
        self.high = high
        self.low = low
        self.consecutive = consecutive
        self.features: list[np.ndarray] = []

    def extract(self, rgb: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = rgb.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        hsv = cv2.cvtColor(rgb[y1 : y2 + 1, x1 : x2 + 1], cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1).astype(np.float32).reshape(-1)

    def similarity(self, feature) -> float:
        """Best similarity of ``feature`` against the target gallery."""
        if feature is None or not self.features:
            return 0.0
        return max(
            1.0 - float(cv2.compareHist(np.asarray(feature, np.float32), f, cv2.HISTCMP_BHATTACHARYYA))
            for f in self.features
        )

    def add(self, feature) -> None:
        if feature is None:
            return
        self.features.append(np.asarray(feature, np.float32))
        if len(self.features) > 24:
            self.features = self.features[:4] + self.features[-20:]


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
def bbox_depth(depth: np.ndarray, bbox, valid_max: float = 10.0) -> float | None:
    """Median valid depth inside the central patch of a bbox (metres)."""
    x1, y1, x2, y2 = bbox
    h, w = depth.shape[:2]
    x1, x2 = max(0, min(int(x1), w - 1)), max(0, min(int(x2), w - 1))
    y1, y2 = max(0, min(int(y1), h - 1)), max(0, min(int(y2), h - 1))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    xa, xb = int(x1 + 0.25 * bw), int(x1 + 0.75 * bw)
    ya, yb = int(y1 + 0.25 * bh), int(y1 + 0.75 * bh)
    patch = depth[ya : yb + 1, xa : xb + 1].astype(np.float32)
    valid = patch[(patch > 0.2) & (patch < valid_max)]
    if valid.size < max(3, int(0.3 * patch.size)):
        return None
    return float(np.median(valid))


def deproject_pixel(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """pixel + Z depth -> 3D point in the camera optical frame."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=float)


def bearing_range_from_base_point(point_base: np.ndarray) -> tuple[float, float]:
    """base_link point (x forward, y left) -> (distance, bearing)."""
    forward, left = float(point_base[0]), float(point_base[1])
    return math.hypot(forward, left), math.atan2(left, forward)
