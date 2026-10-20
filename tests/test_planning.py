"""Map, A* global planner and DWA local planner tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from navfollow.dwa import DWAConfig, DWAPlanner
from navfollow.geometry import Pose2D
from navfollow.maps import MAP_NAMES, GridMap, load_map
from navfollow.planning import (
    astar,
    build_planning_grid,
    line_of_sight_plan,
    path_length,
    point_at_distance,
    project_to_path,
    smooth_path,
)


def _empty_grid(width_m: float = 8.0, height_m: float = 6.0, res: float = 0.05) -> GridMap:
    grid = GridMap(
        name="test",
        occupancy=np.zeros((int(height_m / res), int(width_m / res)), dtype=bool),
        resolution=res,
    )
    grid.build_derived()
    return grid


@pytest.mark.parametrize("map_name", MAP_NAMES)
def test_maps_have_free_start_and_goal(map_name):
    grid = load_map(map_name)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x, y, _theta = grid.sample_free_pose(rng)
        assert grid.is_free(x, y, 0.3)
        assert not grid.robot_blocked(x, y)


@pytest.mark.parametrize("map_name", MAP_NAMES)
def test_maps_free_space_is_connected(map_name):
    """Every sampled free cell must be reachable from one seed free cell."""
    grid = load_map(map_name)
    free = ~grid.occupancy
    from collections import deque

    start = None
    for row in range(grid.height):
        for col in range(grid.width):
            if free[row, col]:
                start = (row, col)
                break
        if start:
            break
    assert start is not None
    seen = np.zeros_like(grid.occupancy)
    queue = deque([start])
    seen[start] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width and not seen[nr, nc] and free[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    assert seen.sum() == free.sum(), f"{map_name}: free space is not fully connected"


@pytest.mark.parametrize("map_name", MAP_NAMES)
def test_astar_finds_path_and_path_is_free(map_name):
    grid = load_map(map_name)
    pgrid = build_planning_grid(grid)
    rng = np.random.default_rng(3)
    for _ in range(3):
        sx, sy, _ = grid.sample_free_pose(rng)
        gx, gy, _ = grid.sample_free_pose(rng)
        raw = astar(pgrid, (sx, sy), (gx, gy))
        assert raw is not None, f"{map_name}: no plan from {(sx, sy)} to {(gx, gy)}"
        assert raw[0] == pytest.approx((sx, sy))
        assert raw[-1] == pytest.approx((gx, gy))
        path = smooth_path(pgrid, raw)
        assert len(path) >= 2
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            assert line_of_sight_plan(pgrid, (x0, y0), (x1, y1))
        assert path_length(path) <= path_length(raw) + 1e-6


def test_astar_returns_none_for_disconnected_map():
    grid = _empty_grid(6.0, 4.0)
    # vertical wall splitting the map in two
    r0, c0 = grid.world_to_grid(3.0, 0.2)
    r1, c1 = grid.world_to_grid(3.1, 3.8)
    grid.occupancy[r0:r1, c0:c1] = True
    grid.build_derived()
    pgrid = build_planning_grid(grid)
    assert astar(pgrid, (1.0, 2.0), (5.0, 2.0)) is None
    assert astar(pgrid, (1.0, 1.0), (2.0, 2.0)) is not None


def test_path_projection_and_arclength():
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    d, s, proj = project_to_path(path, (1.0, 0.5))
    assert d == pytest.approx(0.5)
    assert s == pytest.approx(1.0)
    assert proj[0] == pytest.approx(1.0)
    p = point_at_distance(path, 3.0)
    assert p[0] == pytest.approx(2.0)
    assert p[1] == pytest.approx(1.0)
    assert path_length(path) == pytest.approx(4.0)


def test_dwa_returns_forward_command_on_clear_path():
    grid = _empty_grid()
    dwa = DWAPlanner(grid, DWAConfig())
    pose = Pose2D(1.0, 2.0, 0.0)
    path = [(1.0, 2.0), (5.0, 2.0)]
    cmd = dwa.compute(pose, (0.0, 0.0), path)
    assert cmd is not None
    assert cmd.v > 0.0
    assert abs(cmd.w) < 0.5
    assert cmd.v <= dwa.cfg.max_v + 1e-9
    assert cmd.admissible >= 1


def test_dwa_rejects_colliding_forward_motion():
    grid = _empty_grid()
    # wall directly in front of the robot at x = 1.8
    r0, c0 = grid.world_to_grid(1.8, 0.5)
    r1, c1 = grid.world_to_grid(2.0, 3.0)
    grid.occupancy[r0:r1, c0:c1] = True
    grid.build_derived()
    dwa = DWAPlanner(grid, DWAConfig())
    pose = Pose2D(1.0, 1.8, 0.0)
    path = [(1.0, 1.8), (4.0, 1.8)]
    cmd = dwa.compute(pose, (0.0, 0.0), path)
    # it may turn in place, but it must not command a fast forward run into the wall
    if cmd is not None:
        assert cmd.v < 0.45


def test_dwa_respects_dynamic_persons():
    grid = _empty_grid()
    dwa = DWAPlanner(grid, DWAConfig())
    pose = Pose2D(1.0, 2.0, 0.0)
    path = [(1.0, 2.0), (5.0, 2.0)]
    clear = dwa.compute(pose, (0.0, 0.0), path, persons=[])
    blocked = dwa.compute(pose, (0.0, 0.0), path, persons=[(2.0, 2.0, 0.3)])
    assert clear is not None and blocked is not None
    assert blocked.v <= clear.v + 1e-6
