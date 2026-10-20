"""Multi-object tracking with a ByteTrack-style two-stage association.

The tracker keeps a small Kalman-free constant-velocity box predictor (the
follow controller has its own Kalman filter for the target's *world* position),
matches high-confidence detections first and low-confidence ones second, and
uses the Hungarian algorithm for the assignment.  ``hungarian`` is implemented
locally so the project has no scipy dependency.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular matrix.

    Returns ``(row, col)`` pairs covering ``min(n_rows, n_cols)`` matches.
    Classic O(n^3) shortest augmenting path implementation.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transposed = False
    if n > m:
        cost = cost.T
        n, m = m, n
        transposed = True
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment.append((int(p[j]) - 1, j - 1))
    if transposed:
        assignment = [(b, a) for a, b in assignment]
    return assignment


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy arrays."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1 + 1, 0, None)
    ih = np.clip(iy2 - iy1 + 1, 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0] + 1) * (a[..., 3] - a[..., 1] + 1)
    area_b = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


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
    last_detection: Detection | None = None

    @property
    def center(self) -> tuple[float, float]:
        return float((self.bbox[0] + self.bbox[2]) / 2.0), float((self.bbox[1] + self.bbox[3]) / 2.0)

    def predict(self, dt: float) -> np.ndarray:
        return self.bbox + self.velocity * dt


@dataclass
class TrackerOutput:
    tracks: list[Track]
    matched: list[tuple[Track, Detection]]
    new_tracks: list[Track]


class IoUTracker:
    """ByteTrack-inspired two-stage IoU tracker."""

    def __init__(
        self,
        max_age: float = 0.55,
        iou_threshold: float = 0.20,
        low_score: float = 0.45,
        high_score: float = 0.55,
        new_track_score: float = 0.40,
        center_gate: float = 90.0,
    ) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.low_score = low_score
        self.high_score = high_score
        self.new_track_score = new_track_score
        self.center_gate = center_gate
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def _associate(
        self,
        tracks: list[Track],
        detections: list[Detection],
        dt: float,
        iou_thr: float,
    ) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        if not tracks or not detections:
            return [], tracks, detections
        pred_boxes = np.stack([t.predict(dt) for t in tracks])
        det_boxes = np.array([d.bbox for d in detections], dtype=float)
        ious = iou_matrix(pred_boxes, det_boxes)
        # add a centre-distance gate on top of IoU (small boxes move a lot)
        centers_t = pred_boxes[:, :2] * 0.5 + pred_boxes[:, 2:] * 0.5
        centers_d = det_boxes[:, :2] * 0.5 + det_boxes[:, 2:] * 0.5
        dist = np.linalg.norm(centers_t[:, None, :] - centers_d[None, :, :], axis=2)
        cost = 1.0 - ious
        cost[dist > self.center_gate] = 1e6
        assignment = hungarian(cost)
        matched: list[tuple[Track, Detection]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for ti, di in assignment:
            if ious[ti, di] >= iou_thr and cost[ti, di] < 1e6:
                matched.append((tracks[ti], detections[di]))
                used_t.add(ti)
                used_d.add(di)
        rem_tracks = [t for i, t in enumerate(tracks) if i not in used_t]
        rem_dets = [d for i, d in enumerate(detections) if i not in used_d]
        return matched, rem_tracks, rem_dets

    # ------------------------------------------------------------------
    def update(self, detections: list[Detection], dt: float) -> TrackerOutput:
        for t in self.tracks:
            t.age += dt
            t.time_since_update += dt
            t.bbox = t.predict(dt)

        high = [d for d in detections if d.score >= self.high_score]
        low = [d for d in detections if d.score < self.high_score]

        matched_high, rem_tracks, rem_high = self._associate(self.tracks, high, dt, self.iou_threshold)
        for track, det in matched_high:
            self._update_track(track, det, dt)

        # stage 2: low-score detections matched to still-unmatched tracks
        matched_low, rem_tracks, rem_low = self._associate(
            rem_tracks, low, dt, self.iou_threshold * 0.5
        )
        for track, det in matched_low:
            self._update_track(track, det, dt)

        new_tracks: list[Track] = []
        # unmatched *high* detections spawn new tracks; unmatched low ones are
        # dropped (ByteTrack behaviour)
        for det in rem_high:
            if det.score < self.new_track_score:
                continue
            track = Track(
                track_id=self._next_id,
                bbox=np.array(det.bbox, dtype=float),
                score=det.score,
                last_detection=det,
            )
            self._next_id += 1
            self.tracks.append(track)
            new_tracks.append(track)

        # drop stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return TrackerOutput(tracks=list(self.tracks), matched=matched_high + matched_low, new_tracks=new_tracks)

    def _update_track(self, track: Track, det: Detection, dt: float) -> None:
        new_bbox = np.array(det.bbox, dtype=float)
        if dt > 1e-9:
            track.velocity = 0.5 * track.velocity + 0.5 * ((new_bbox - track.bbox) / dt)
        track.bbox = new_bbox
        track.score = det.score
        track.hits += 1
        track.time_since_update = 0.0
        track.last_detection = det

    def get(self, track_id: int) -> Track | None:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None
