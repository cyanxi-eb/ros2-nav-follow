"""Perception tests: detector, tracker, ReID, projection, synthetic RGB-D pipeline."""
from __future__ import annotations

import math

import numpy as np
import pytest

from navfollow.camera import CameraIntrinsics, SyntheticCamera, undistort_rgbd
from navfollow.geometry import Pose2D
from navfollow.maps import load_map
from navfollow.perception.detector import ColorPersonDetector, Detection
from navfollow.perception.pipeline import PerceptionPipeline
from navfollow.perception.projection import (
    base_observation_from_bbox,
    bbox_depth,
    deproject_pixel,
    project_point,
)
from navfollow.perception.reid import ColorHistogramReID, ReIDConfig, ReIDGallery, motion_gate
from navfollow.perception.tracker import IoUTracker, hungarian, iou_matrix
from navfollow.persons import Person
from navfollow.simulation import build_follow_scene
from navfollow.tf_tree import build_robot_tree
from navfollow.world import SimWorld


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------
def _paint_box(rgb: np.ndarray, bbox, color) -> np.ndarray:
    out = rgb.copy()
    x1, y1, x2, y2 = bbox
    out[y1 : y2 + 1, x1 : x2 + 1] = color
    return out


def test_detector_finds_painted_jacket():
    rgb = np.full((120, 160, 3), 190, dtype=np.uint8)
    rgb = _paint_box(rgb, (60, 30, 95, 90), (205, 45, 45))  # red jacket
    dets = ColorPersonDetector().detect(rgb)
    assert len(dets) == 1
    det = dets[0]
    assert det.color_name == "red"
    assert det.score > 0.4
    cx, cy = det.center
    assert 70 <= cx <= 85
    assert 50 <= cy <= 70


def test_detector_ignores_background_and_empty_image():
    rgb = np.full((120, 160, 3), 180, dtype=np.uint8)  # grey floor, low saturation
    assert ColorPersonDetector().detect(rgb) == []
    assert ColorPersonDetector().detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_detector_separates_two_colors():
    rgb = np.full((120, 160, 3), 185, dtype=np.uint8)
    rgb = _paint_box(rgb, (20, 30, 55, 95), (40, 90, 200))  # blue
    rgb = _paint_box(rgb, (100, 30, 130, 95), (40, 165, 70))  # green
    dets = ColorPersonDetector().detect(rgb)
    colors = {d.color_name for d in dets}
    assert colors == {"blue", "green"}


# ---------------------------------------------------------------------------
# tracker / association
# ---------------------------------------------------------------------------
def test_hungarian_known_assignment():
    cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    assignment = sorted(hungarian(cost))
    assert assignment == [(0, 1), (1, 0), (2, 2)]
    total = sum(cost[i, j] for i, j in assignment)
    assert total == pytest.approx(1.0 + 2.0 + 2.0)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 15, 15]], dtype=float)
    m = iou_matrix(a, b)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(0.0)
    assert 0.1 < m[0, 2] < 0.2


def test_tracker_keeps_id_across_frames():
    tracker = IoUTracker()
    det = Detection(bbox=(40, 20, 80, 90), score=0.9)
    out1 = tracker.update([det], dt=0.1)
    track_id = out1.tracks[0].track_id
    moved = Detection(bbox=(45, 20, 85, 90), score=0.9)
    out2 = tracker.update([moved], dt=0.1)
    assert len(out2.matched) == 1
    assert out2.matched[0][0].track_id == track_id


def test_tracker_drops_tracks_after_max_age():
    tracker = IoUTracker(max_age=0.3)
    tracker.update([Detection(bbox=(40, 20, 80, 90), score=0.9)], dt=0.1)
    assert len(tracker.tracks) == 1
    for _ in range(5):
        tracker.update([], dt=0.1)
    assert tracker.tracks == []


# ---------------------------------------------------------------------------
# ReID
# ---------------------------------------------------------------------------
def test_reid_same_colour_more_similar_than_different():
    reid = ColorHistogramReID()
    rgb = np.full((100, 100, 3), 40, dtype=np.uint8)
    rgb[:, :] = (200, 40, 40)
    other = np.full((100, 100, 3), (40, 90, 200), dtype=np.uint8)
    crop_bbox = (0, 0, 99, 99)
    f_red = reid.extract(rgb, crop_bbox)
    assert f_red is not None and f_red.shape[0] == ReIDConfig().bins_h * ReIDConfig().bins_s
    # same jacket under different illumination (multiplicative shading)
    brighter = (rgb.astype(np.float32) * 1.15).clip(0, 255).astype(np.uint8)
    f_red_bright = reid.extract(brighter, crop_bbox)
    f_blue = reid.extract(other, crop_bbox)
    assert reid.similarity(f_red, f_red_bright) > 0.75
    assert reid.similarity(f_red, f_blue) < 0.35


def test_reid_gallery_best_match_and_motion_gate():
    reid = ColorHistogramReID()
    gallery = ReIDGallery(reid=reid)
    red = np.full((80, 80, 3), (205, 45, 45), dtype=np.uint8)
    blue = np.full((80, 80, 3), (40, 90, 200), dtype=np.uint8)
    f_red = reid.extract(red, (0, 0, 79, 79))
    f_blue = reid.extract(blue, (0, 0, 79, 79))
    gallery.add(f_red)
    assert gallery.best_similarity(f_red) > gallery.best_similarity(f_blue)
    assert motion_gate((1.0, 1.0), (1.2, 1.1), 0.0)
    assert not motion_gate((1.0, 1.0), (5.0, 5.0), 0.0)
    assert motion_gate((1.0, 1.0), (5.0, 5.0), 8.0)


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------
def test_deproject_project_roundtrip():
    K = CameraIntrinsics(width=160, height=120).K
    point = deproject_pixel(100.0, 40.0, 2.5, K)
    u, v, depth = project_point(point, K)
    assert u == pytest.approx(100.0, abs=1e-6)
    assert v == pytest.approx(40.0, abs=1e-6)
    assert depth == pytest.approx(2.5, abs=1e-9)
    # a pixel at the principal point maps to a point on the optical axis
    K2 = CameraIntrinsics(width=160, height=120)
    centre = deproject_pixel(K2.cx, K2.cy, 3.0, K2.K)
    assert np.allclose(centre, [0.0, 0.0, 3.0], atol=1e-9)


def test_bbox_depth_uses_central_region():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[20:90, 20:90] = 2.0
    depth[20:90, 20:50] = 1.0  # a near occluder on the left half
    value = bbox_depth(depth, (20, 20, 89, 89))
    assert value is not None
    # the central patch is dominated by the 2.0 m region
    assert 1.0 <= value <= 2.0
    assert bbox_depth(np.zeros((100, 100), dtype=np.float32), (10, 10, 50, 50)) is None


def test_camera_render_and_person_geometry():
    grid, world, target, pose = build_follow_scene()
    camera = SyntheticCamera()
    rgb, depth = camera.render(world, pose)
    assert rgb.shape == (120, 160, 3)
    assert depth.shape == (120, 160)
    assert depth[rgb.shape[0] // 2 :, :].max() > 0.5  # floor has valid depth
    # front person should be visible and roughly at the true distance
    visible = np.zeros_like(depth, dtype=bool)
    torso = depth > 0
    assert torso.sum() > 1000
    gt_distance = math.hypot(target.x - pose.x, target.y - pose.y)
    person_pixels = depth[(depth > 0) & (depth < gt_distance + 0.6) & (depth > gt_distance - 0.8)]
    assert person_pixels.size > 100


def test_person_hidden_behind_wall_not_rendered():
    grid = load_map("warehouse_a")
    robot_pose = Pose2D(2.0, 6.5, 0.0)
    # 6.2 m ahead lies behind the 0.7 m pillar at x = 5.4 in the central aisle
    person = Person(
        person_id=1,
        color_name="red",
        waypoints=[(6.2, 6.5)],
        speed=0.0,
        loop=False,
    )
    world = SimWorld(grid, [person])
    assert not world.person_visible(person, robot_pose, half_fov=0.5)
    # in front of the robot with no pillar in between
    person.x, person.y = 3.0, 6.5
    assert world.person_visible(person, robot_pose, half_fov=0.5)
    # behind the robot's field of view
    person.x, person.y = 1.0, 6.5
    assert not world.person_visible(person, robot_pose, half_fov=0.5)


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------
def test_pipeline_locks_and_tracks_target():
    grid, world, target, pose = build_follow_scene()
    camera = SyntheticCamera()
    pipeline = PerceptionPipeline(intrinsics=camera.intrinsics)
    rgb, depth = camera.render(world, pose)
    frame = pipeline.step(rgb, depth, 0.0, 0.0, robot_pose=pose)
    assert not frame.dropped
    assert len(frame.detections) >= 1
    assert pipeline.lock_target_by_world(target.pose, frame.detections)
    locked = pipeline.target_track_id
    assert locked is not None
    world.step(0.1, pose)
    rgb, depth = camera.render(world, pose)
    frame2 = pipeline.step(rgb, depth, 0.1, 0.1, robot_pose=pose)
    assert frame2.target is not None
    assert frame2.target_track_id == locked
    assert frame2.target.distance is not None
    assert abs(frame2.target.distance - 1.22) < 0.6


def test_pipeline_drops_timestamp_mismatch():
    grid, world, target, pose = build_follow_scene()
    camera = SyntheticCamera()
    pipeline = PerceptionPipeline(intrinsics=camera.intrinsics, time_tolerance=0.02)
    rgb, depth = camera.render(world, pose)
    frame = pipeline.step(rgb, depth, 1.0, 1.5, robot_pose=pose)
    assert frame.dropped
    assert frame.drop_reason == "depth_rgb_timestamp_mismatch"
    frame_ok = pipeline.step(rgb, depth, 1.1, 1.11, robot_pose=pose)
    assert not frame_ok.dropped


def test_base_observation_from_bbox_matches_depth():
    from navfollow.camera import CameraMount

    K = CameraIntrinsics().K
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    T_base_optical = CameraMount().optical_matrix()
    obs = base_observation_from_bbox((60, 40, 100, 90), depth, K, T_base_optical)
    assert obs is not None
    distance, bearing, base_xyz = obs
    assert distance == pytest.approx(2.0, abs=0.15)
    assert abs(bearing) < 0.3
    # camera is mounted 1.05 m above base_link and looks slightly down, so the
    # 2 m point sits ~0.7 m above the floor plane
    assert 0.5 < base_xyz[2] < 0.9
    assert base_xyz[0] > 1.9
