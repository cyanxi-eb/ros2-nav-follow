"""Perception subpackage: detector, tracker, ReID, projection, pipeline."""

from .detector import ColorPersonDetector, Detection, Detector, UltralyticsYOLOv8Detector
from .pipeline import FrameResult, PerceptionPipeline, TrackedDetection
from .projection import base_observation_from_bbox, bbox_depth, deproject_pixel
from .reid import ColorHistogramReID, ReIDConfig, ReIDGallery, motion_gate
from .tracker import IoUTracker, Track, TrackerOutput, hungarian, iou_matrix

__all__ = [
    "Detection",
    "Detector",
    "ColorPersonDetector",
    "UltralyticsYOLOv8Detector",
    "IoUTracker",
    "Track",
    "TrackerOutput",
    "hungarian",
    "iou_matrix",
    "ColorHistogramReID",
    "ReIDConfig",
    "ReIDGallery",
    "motion_gate",
    "deproject_pixel",
    "bbox_depth",
    "base_observation_from_bbox",
    "PerceptionPipeline",
    "FrameResult",
    "TrackedDetection",
]
