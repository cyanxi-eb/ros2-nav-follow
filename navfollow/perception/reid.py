"""Colour-histogram re-identification (ReID) for the person-following task.

The evaluator's target can be hidden for 1-3 s.  When it reappears, the tracker
has already dropped the old track, so the follower re-associates detections with
the recorded appearance model of the target using two thresholds:

* ``high`` similarity + motion gate -> immediate re-acquisition,
* ``low`` similarity + motion gate  -> candidate, requires N consecutive frames.

The appearance model is an HSV (H,S) histogram of the jacket crop.  It is cheap,
needs no neural weights, and is robust to the distance-dependent brightness
shading of the synthetic camera because V is not part of the descriptor (a real
deployment can swap in OSNet embeddings behind the same ``extract``/
``similarity`` interface).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ReIDConfig:
    bins_h: int = 16
    bins_s: int = 8
    min_crop: int = 8
    high_threshold: float = 0.78
    low_threshold: float = 0.52
    consecutive_frames: int = 3
    gate_radius: float = 1.6  # metres, grows with time since last seen
    gate_growth: float = 0.6  # metres per second


class ColorHistogramReID:
    """Appearance descriptor + similarity for jacket crops."""

    def __init__(self, config: ReIDConfig | None = None) -> None:
        self.cfg = config or ReIDConfig()

    def extract(self, rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h, w = rgb.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 - x1 + 1 < self.cfg.min_crop or y2 - y1 + 1 < self.cfg.min_crop:
            return None
        crop = rgb[y1 : y2 + 1, x1 : x2 + 1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [self.cfg.bins_h, self.cfg.bins_s],
            [0, 180, 0, 256],
        )
        hist = cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1).astype(np.float32)
        return hist.reshape(-1)

    def similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        # OpenCV's Bhattacharyya distance is in [0, 1]; 1 - d is a similarity.
        dist = float(
            cv2.compareHist(
                np.asarray(a, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        return float(np.clip(1.0 - dist, 0.0, 1.0))


@dataclass
class ReIDGallery:
    """Bounded gallery of target appearance descriptors."""

    reid: ColorHistogramReID = field(default_factory=ColorHistogramReID)
    max_features: int = 24

    def __post_init__(self) -> None:
        self.features: list[np.ndarray] = []

    def add(self, feature: np.ndarray | None) -> None:
        if feature is None:
            return
        self.features.append(np.asarray(feature, dtype=np.float32))
        if len(self.features) > self.max_features:
            # keep the most recent half plus a few early ones for diversity
            early = self.features[:4]
            recent = self.features[-(self.max_features - 4) :]
            self.features = early + recent

    def best_similarity(self, feature: np.ndarray | None) -> float:
        if feature is None or not self.features:
            return 0.0
        return max(self.reid.similarity(f, feature) for f in self.features)

    def clear(self) -> None:
        self.features = []

    def __len__(self) -> int:
        return len(self.features)


def motion_gate(
    predicted_world: tuple[float, float] | None,
    detection_world: tuple[float, float] | None,
    seconds_since_seen: float,
    config: ReIDConfig | None = None,
) -> bool:
    """True when the detected person is close enough to the Kalman prediction."""
    cfg = config or ReIDConfig()
    if predicted_world is None or detection_world is None:
        return False
    radius = cfg.gate_radius + cfg.gate_growth * max(0.0, seconds_since_seen)
    return float(np.hypot(predicted_world[0] - detection_world[0], predicted_world[1] - detection_world[1])) <= radius
