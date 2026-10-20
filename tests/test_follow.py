"""Follower tests: Kalman, PID, state machine transitions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from navfollow.follow.controller import FollowConfig, FollowController, FollowState
from navfollow.follow.kalman import KalmanCV2D
from navfollow.follow.pid import FollowPID, FollowPIDConfig, PID, PIDConfig
from navfollow.geometry import Pose2D
from navfollow.maps import load_map
from navfollow.perception.detector import Detection
from navfollow.perception.pipeline import FrameResult, TrackedDetection
from navfollow.perception.reid import ColorHistogramReID, ReIDConfig


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------
def test_kalman_estimates_constant_velocity():
    kf = KalmanCV2D(accel_std=0.8, meas_std=0.1)
    kf.reset(0.0, 0.0)
    rng = np.random.default_rng(0)
    for i in range(1, 60):
        t = i * 0.1
        kf.predict(0.1)
        kf.update(0.5 * t + rng.normal(0, 0.05), 0.25 * t + rng.normal(0, 0.05))
    assert kf.velocity[0] == pytest.approx(0.5, abs=0.12)
    assert kf.velocity[1] == pytest.approx(0.25, abs=0.12)
    pred_x, pred_y = kf.predict_position(1.0)
    assert pred_x == pytest.approx(0.5 * 6.0 + 0.5, abs=0.3)
    assert pred_y == pytest.approx(0.25 * 6.0 + 0.25, abs=0.3)


def test_kalman_extrapolates_through_occlusion():
    kf = KalmanCV2D(accel_std=0.5, meas_std=0.05)
    kf.reset(0.0, 0.0)
    for i in range(1, 20):
        kf.predict(0.1)
        kf.update(0.6 * i * 0.1, 0.0)
    # 1.5 s without measurements: predicted position keeps advancing
    kf.predict(1.5)
    assert kf.position[0] == pytest.approx(1.14 + 0.6 * 1.5, abs=0.25)


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------
def test_pid_clamps_and_zeroes_integral_windup():
    pid = PID(PIDConfig(kp=10.0, ki=5.0, kd=0.0, out_min=-0.5, out_max=0.5, integral_limit=1.0))
    outputs = [pid.update(1.0, 0.1) for _ in range(30)]
    assert max(outputs) <= 0.5 + 1e-9
    assert abs(pid.integral) <= 1.0 + 1e-9
    # negative error should drive output negative
    assert pid.update(-1.0, 0.1) < 0.0


def test_follow_pid_drives_toward_desired_distance():
    def step(distance: float, bearing: float = 0.0):
        pid = FollowPID(FollowPIDConfig(desired_distance=1.4, kp_dist=1.3))
        return pid.compute(distance, bearing, 0.1)

    v_far, _ = step(2.4)
    v_near, _ = step(0.9)
    v_ok, _ = step(1.4)
    assert v_far > 0.4
    assert v_near < 0.0
    assert abs(v_ok) < 0.1
    # large bearing -> turn in place, linear velocity reduced
    v_turn, w_turn = step(2.4, 1.2)
    assert v_turn < v_far
    assert w_turn > 0.0


# ---------------------------------------------------------------------------
# controller helpers
# ---------------------------------------------------------------------------
class FakeMatcher:
    """Pipeline-compatible matcher with a controllable similarity."""

    def __init__(self, similarity: float = 0.95, gate: bool = True) -> None:
        self.similarity = similarity
        self.gate = gate
        self.locked: list[int] = []

    def similarity_to_target(self, feature):
        return self.similarity

    def motion_gate(self, predicted_world, det, seconds_since_seen):
        return self.gate

    def lock_track(self, track_id, feature=None):
        self.locked.append(track_id)


def make_observation(world_xy=(2.0, 2.0), distance=1.5, bearing=0.0, track_id=1) -> TrackedDetection:
    return TrackedDetection(
        detection=Detection(bbox=(60, 40, 100, 90), score=0.9),
        track_id=track_id,
        feature=np.ones(8, dtype=np.float32),
        depth=distance,
        base_xy=np.array([distance * math.cos(bearing), distance * math.sin(bearing), 0.0]),
        world_xy=world_xy,
        distance=distance,
        bearing=bearing,
    )


def make_frame(observation: TrackedDetection | None, target_track_id: int | None = 1) -> FrameResult:
    return FrameResult(
        stamp=0.0,
        detections=[] if observation is None else [observation],
        target=observation,
        target_track_id=target_track_id,
    )


def make_detection_frame(observation: TrackedDetection) -> FrameResult:
    """A frame where the locked track is gone but a candidate detection exists."""
    return FrameResult(
        stamp=0.0,
        detections=[observation],
        target=None,
        target_track_id=1,
    )


def _controller(**overrides) -> FollowController:
    cfg = FollowConfig(
        lost_timeout=0.3,
        occluded_timeout=1.0,
        reacquire_timeout=2.0,
        arrival_hold=0.5,
        search_timeout=5.0,
        **overrides,
    )
    return FollowController(config=cfg, matcher=FakeMatcher(), grid=load_map("warehouse_a"))


def _feed_visible(controller: FollowController, steps: int, dt: float = 0.1, world=(2.0, 2.0), start=0.0):
    now = start
    for _ in range(steps):
        now += dt
        controller.update(make_frame(make_observation(world_xy=world)), dt, now, Pose2D(1.0, 2.0, 0.0))
    return now


# ---------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------
def test_controller_tracks_visible_target():
    controller = _controller()
    cmd = controller.update(make_frame(make_observation()), 0.1, 0.0, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.TRACK
    assert controller.transitions[0][1] == FollowState.SEARCH
    assert controller.transitions[0][2] == FollowState.TRACK


def test_controller_enters_occluded_then_reacquires_strong_match():
    controller = _controller()
    now = _feed_visible(controller, 4)
    # target disappears
    for _ in range(6):
        now += 0.1
        cmd = controller.update(make_frame(None), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state in (FollowState.OCCLUDED, FollowState.REACQUIRE)
    assert any(ev["type"] == "occluded" for ev in controller.events)
    # after occluded_timeout the controller starts a scan
    for _ in range(12):
        now += 0.1
        cmd = controller.update(make_frame(None), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.REACQUIRE
    # ReID strong match -> back to TRACK
    fresh = make_observation(track_id=7)
    cmd = controller.update(make_detection_frame(fresh), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.TRACK
    assert controller.reacquire_success >= 1
    assert 7 in controller.matcher.locked


def test_controller_stops_after_reacquire_timeout():
    controller = _controller()
    now = _feed_visible(controller, 5)
    for _ in range(80):
        now += 0.1
        cmd = controller.update(make_frame(None), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.STOP
    assert cmd.v == 0.0 and cmd.w == 0.0


def test_controller_requires_consecutive_low_similarity_matches():
    controller = _controller()
    controller.cfg.reid_high = 0.99
    controller.cfg.reid_low = 0.5
    controller.cfg.reid_consecutive = 3
    controller.matcher = FakeMatcher(similarity=0.7)
    now = _feed_visible(controller, 3)
    for _ in range(5):
        now += 0.1
        cmd = controller.update(make_frame(None), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state in (FollowState.OCCLUDED, FollowState.REACQUIRE)
    # one weak match is not enough
    obs = make_observation(track_id=9)
    controller.update(make_detection_frame(obs), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert controller.state in (FollowState.OCCLUDED, FollowState.REACQUIRE)
    # three consecutive weak matches are accepted
    for _ in range(3):
        now += 0.1
        cmd = controller.update(make_detection_frame(make_observation(track_id=9)), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.TRACK


def test_controller_arrives_when_target_stops():
    controller = _controller()
    now = _feed_visible(controller, 30, world=(2.0, 2.0))
    assert controller.state == FollowState.ARRIVED
    cmd = controller.update(make_frame(make_observation()), 0.1, now, Pose2D(1.0, 2.0, 0.0))
    assert cmd.v == 0.0


def test_controller_search_spins_without_target():
    controller = _controller()
    cmd = controller.update(None, 0.1, 0.0, Pose2D(1.0, 2.0, 0.0))
    assert cmd.state == FollowState.SEARCH
    assert abs(cmd.w) > 0.0
