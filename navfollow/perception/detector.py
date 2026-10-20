"""Person detection interfaces.

Two detectors are provided:

* :class:`ColorPersonDetector` - the offline detector used by the simulator and
  the test-suite.  It segments the jacket colour of a rendered pedestrian in
  HSV space, cleans the mask with morphology and returns bounding boxes.  This
  keeps the perception pipeline fully runnable without model weights or a GPU.
* :class:`UltralyticsYOLOv8Detector` - the adapter used on the real robot /
  Gazebo.  ``ultralytics`` is an *optional* dependency: importing this module
  never requires it, and the class raises a clear error if the package is not
  installed.  It is intentionally excluded from the automated evaluation so
  that the reported numbers are reproducible on a CPU-only machine.

Both implement the same :class:`Detector` interface, so the tracker / ReID /
follow stack is detector-agnostic (the ROS2 node in
``ros2_ws/src/follow_robot/follow_robot/perception_node.py`` swaps in YOLOv8).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from ..persons import PERSON_COLORS


@dataclass
class Detection:
    """One detected person bounding box."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (inclusive)
    score: float
    class_name: str = "person"
    color_name: str | None = None

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


class Detector(ABC):
    """Common detector interface used by the perception pipeline."""

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Return person detections for an RGB uint8 image (H, W, 3)."""


def _hsv_of(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    px = np.uint8([[list(rgb)]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    return int(h), int(s), int(v)


class ColorPersonDetector(Detector):
    """HSV colour-blob person detector for the synthetic camera images."""

    def __init__(
        self,
        colors: list[str] | None = None,
        min_area: int = 110,
        max_area_ratio: float = 0.55,
        min_aspect: float = 0.45,
        hue_tolerance: int = 10,
        sat_threshold: int = 70,
        val_threshold: int = 40,
    ) -> None:
        self.color_names = colors or list(PERSON_COLORS.keys())
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_aspect = min_aspect
        self.hue_tolerance = hue_tolerance
        self.sat_threshold = sat_threshold
        self.val_threshold = val_threshold
        self._ranges: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
        for name in self.color_names:
            self._ranges[name] = self._make_ranges(PERSON_COLORS[name])
        self._kernel = np.ones((3, 3), np.uint8)

    def _make_ranges(
        self, rgb: tuple[int, int, int]
    ) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
        h, s, _v = _hsv_of(rgb)
        sat_lo = max(55, min(self.sat_threshold, s - 60))
        val_lo = self.val_threshold
        lo = h - self.hue_tolerance
        hi = h + self.hue_tolerance
        if lo < 0:
            return [
                ((0, sat_lo, val_lo), (hi, 255, 255)),
                ((180 + lo, sat_lo, val_lo), (179, 255, 255)),
            ]
        if hi > 179:
            return [
                ((lo, sat_lo, val_lo), (179, 255, 255)),
                ((0, sat_lo, val_lo), (hi - 180, 255, 255)),
            ]
        return [((lo, sat_lo, val_lo), (hi, 255, 255))]

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        if rgb is None or rgb.size == 0:
            return []
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h_img, w_img = rgb.shape[:2]
        detections: list[Detection] = []
        for name, ranges in self._ranges.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            if mask is None:
                continue
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if area > self.max_area_ratio * h_img * w_img:
                    continue
                if h <= 0 or w / h < self.min_aspect:
                    continue
                # a jacket blob that is far away has few pixels: score reflects
                # apparent size, similar to a detector confidence
                score = float(min(0.99, 0.35 + area / 1600.0))
                detections.append(
                    Detection(
                        bbox=(int(x), int(y), int(x + w - 1), int(y + h - 1)),
                        score=score,
                        class_name="person",
                        color_name=name,
                    )
                )
        # remove near-duplicate boxes of the same person (e.g. split blobs)
        detections.sort(key=lambda d: d.area, reverse=True)
        kept: list[Detection] = []
        for det in detections:
            duplicate = False
            for other in kept:
                if _iou(det.bbox, other.bbox) > 0.35:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class UltralyticsYOLOv8Detector(Detector):
    """YOLOv8n person detector adapter (optional dependency).

    Only ``class_id == 0`` (person, COCO) detections are returned.  This is the
    detector used by the ROS2 node on Gazebo / the real robot; it is not used in
    the CPU-only evaluation so that results stay reproducible without weights.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ultralytics is not installed.  Install it with "
                "`pip install ultralytics` or use ColorPersonDetector for the "
                "offline pipeline.  See docs/ros2_integration.md."
            ) from exc
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, rgb: np.ndarray) -> list[Detection]:  # pragma: no cover - optional
        results = self.model.predict(rgb, conf=self.conf, classes=[0], device=self.device, verbose=False)
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            score = float(box.conf[0]) if box.conf is not None else 0.0
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(round(x2)), int(round(y2))),
                    score=score,
                    class_name="person",
                )
            )
        return detections
