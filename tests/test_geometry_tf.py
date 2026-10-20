"""Geometry and TF tree tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from navfollow.geometry import (
    Pose2D,
    angle_diff,
    matrix_to_quaternion,
    quaternion_from_yaw,
    quaternion_slerp,
    quaternion_to_matrix,
    rpy_to_matrix,
    transform_inverse,
    wrap_angle,
)
from navfollow.tf_tree import (
    ExtrapolationError,
    FrameNotFound,
    TransformError,
    TransformTree,
    build_robot_tree,
    optical_rotation,
)


def test_wrap_angle_and_angle_diff():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert abs(wrap_angle(-3 * math.pi)) == pytest.approx(math.pi)
    assert angle_diff(0.1, 6.2) == pytest.approx(0.1 - 6.2 + 2 * math.pi)
    for a in (-3.0, -0.2, 0.0, 1.7, 100.0):
        assert -math.pi < wrap_angle(a) <= math.pi


def test_quaternion_roundtrip():
    for roll, pitch, yaw in [(0, 0, 0), (0.1, -0.2, 0.3), (-0.4, 0.5, -2.9)]:
        R = rpy_to_matrix(roll, pitch, yaw)
        q = matrix_to_quaternion(R)
        assert np.allclose(quaternion_to_matrix(q), R, atol=1e-9)
        assert np.linalg.norm(q) == pytest.approx(1.0)


def test_quaternion_yaw_and_slerp():
    q0 = quaternion_from_yaw(0.0)
    q1 = quaternion_from_yaw(math.pi / 2)
    q_mid = quaternion_slerp(q0, q1, 0.5)
    R = quaternion_to_matrix(q_mid)
    assert math.atan2(R[1, 0], R[0, 0]) == pytest.approx(math.pi / 4, abs=1e-6)
    # antipodal quaternions must not blow up
    assert np.linalg.norm(quaternion_slerp(q0, -q1, 0.5)) == pytest.approx(1.0)


def test_pose2d_compose_and_transform_point():
    parent = Pose2D(1.0, 2.0, math.pi / 2)
    child = Pose2D(0.5, 0.0, 0.0)
    composed = parent.compose(child)
    # child at 0.5 m forward of the parent: world (1.0, 2.5)
    assert composed.x == pytest.approx(1.0)
    assert composed.y == pytest.approx(2.5)
    local = parent.transform_point(1.0, 2.5)
    assert local[0] == pytest.approx(0.5)
    assert local[1] == pytest.approx(0.0, abs=1e-9)
    inverse = parent.inverse()
    world = inverse.compose(parent)
    assert world.x == pytest.approx(0.0, abs=1e-9)
    assert world.y == pytest.approx(0.0, abs=1e-9)
    assert wrap_angle(world.theta) == pytest.approx(0.0, abs=1e-9)


def test_transform_inverse_roundtrip():
    R = rpy_to_matrix(0.1, 0.2, 0.3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (1.0, -2.0, 0.5)
    assert np.allclose(T @ transform_inverse(T), np.eye(4), atol=1e-9)


def test_tf_chain_lookup_roundtrip():
    tf = build_robot_tree()
    tf.set_transform(
        "odom",
        "base_link",
        translation=(2.0, 1.0, 0.0),
        quaternion=quaternion_from_yaw(0.5),
        stamp=10.0,
    )
    T_base_cam = tf.lookup("base_link", "camera_optical", 10.0)
    T_cam_base = tf.lookup("camera_optical", "base_link", 10.0)
    assert np.allclose(T_base_cam @ T_cam_base, np.eye(4), atol=1e-9)
    T_map_base = tf.lookup("map", "base_link", 10.0)
    assert T_map_base[0, 3] == pytest.approx(2.0)
    assert T_map_base[1, 3] == pytest.approx(1.0)
    # camera optical z axis (forward) must point along the robot heading
    forward = T_map_base[:3, :3] @ T_base_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])
    horizontal = math.hypot(forward[0], forward[1])
    assert forward[0] / horizontal == pytest.approx(math.cos(0.5), abs=1e-6)
    assert forward[1] / horizontal == pytest.approx(math.sin(0.5), abs=1e-6)
    # ... and 8 degrees downward (the camera is mounted with a downward tilt)
    assert forward[2] == pytest.approx(-math.sin(0.14), abs=1e-6)


def test_tf_optical_rotation_is_rep103():
    R = optical_rotation()
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_tf_temporal_interpolation():
    tf = TransformTree()
    tf.add_frame("map")
    tf.set_transform("map", "base_link", translation=(0.0, 0.0, 0.0), stamp=0.0)
    tf.set_transform("map", "base_link", translation=(1.0, 0.0, 0.0), stamp=1.0)
    T = tf.lookup("map", "base_link", 0.25)
    assert T[0, 3] == pytest.approx(0.25)
    T2 = tf.lookup("map", "base_link", 0.75)
    assert T2[0, 3] == pytest.approx(0.75)


def test_tf_stale_and_unknown_frames():
    tf = TransformTree(tolerance=0.1)
    tf.add_frame("map")
    tf.set_transform("map", "base_link", translation=(0.0, 0.0, 0.0), stamp=5.0)
    with pytest.raises(ExtrapolationError):
        tf.lookup("map", "base_link", 3.0)
    with pytest.raises(FrameNotFound):
        tf.lookup("map", "nope", 5.0)
    # a second tree branch is not connected
    tf.set_transform("world", "other", translation=(0.0, 0.0, 0.0), stamp=5.0)
    with pytest.raises(TransformError):
        tf.lookup("map", "other", 5.0)


def test_tf_cannot_reparent():
    tf = TransformTree()
    tf.set_transform("map", "base", stamp=0.0)
    with pytest.raises(TransformError):
        tf.set_transform("odom", "base", stamp=1.0)
