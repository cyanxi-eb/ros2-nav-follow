"""End-to-end simulation + evaluation aggregation tests (fast smoke runs)."""
from __future__ import annotations

import numpy as np
import pytest

from navfollow.evaluate import run_navigation_suite, run_occlusion_suite
from navfollow.maps import load_map
from navfollow.simulation import run_follow_episode, run_navigation_trial


def test_navigation_trial_returns_consistent_metrics():
    result = run_navigation_trial(
        "warehouse_a",
        seed=1,
        n_pedestrians=0,
        max_time=45.0,
        start=(2.0, 6.5, 0.0),
        goal=(17.5, 6.5),
    )
    assert result.map_name == "warehouse_a"
    assert result.success in (True, False)
    assert result.time_s > 0.0
    assert result.path_length > 1.0
    assert result.trajectory_length > 0.0
    assert 0.0 < result.path_ratio < 70.0
    assert result.collision in (True, False)
    assert len(result.trajectory) > 10
    if result.success:
        assert result.path_ratio < 2.0
        assert abs(result.final_pose[0] - result.goal[0]) < 0.6
        assert abs(result.final_pose[1] - result.goal[1]) < 0.6


def test_navigation_trial_fails_for_unreachable_goal():
    grid = load_map("warehouse_a")
    # goal inside a shelf block -> the planner cannot reach it
    result = run_navigation_trial(
        "warehouse_a",
        seed=2,
        n_pedestrians=0,
        max_time=25.0,
        start=(2.0, 6.5, 0.0),
        goal=(6.55, 3.5),
        grid=grid,
    )
    assert not result.success
    assert result.recoveries > 0


def test_follow_episode_tracks_and_reports_distance_error():
    result = run_follow_episode(duration=6.0, collect_frames=True)
    assert result.collisions == 0
    assert result.tracking_ratio > 0.4
    assert len(result.distance_errors) > 10
    errors = np.array(result.distance_errors)
    assert float(np.mean(errors)) < 0.6
    assert result.events == []
    assert result.to_dict()["occlusion_events"] == 0


def test_follow_episode_reacquires_after_occlusion():
    result = run_follow_episode(
        duration=12.0,
        occlusion_script=[(4.0, 1.5)],
        collect_frames=True,
    )
    assert len(result.events) == 1
    event = result.events[0]
    assert event.success
    assert event.identity_correct
    assert event.recovery_time is not None and 0.0 <= event.recovery_time <= 5.0
    assert result.collisions == 0


def test_evaluation_suite_aggregation_smoke():
    nav = run_navigation_suite(maps=("warehouse_a",), trials_per_map=1, verbose=False)
    assert nav["trials"] == 1
    assert 0.0 <= nav["success_rate"] <= 1.0
    assert nav["collisions"] in (0, 1)
    assert "warehouse_a" in nav["per_map"]

    occ = run_occlusion_suite(episodes=1, events_per_episode=1, episode_duration=8.0, verbose=False)
    assert occ["episodes"] == 1
    assert occ["events"] == 1
    assert 0.0 <= occ["reacquire_rate"] <= 1.0
    if occ["reacquire_successes"]:
        assert occ["distance_error_mean_m"] >= 0.0
        assert occ["recovery_time_mean_s"] >= 0.0
