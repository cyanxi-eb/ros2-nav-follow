"""A* global planner with a Nav2-like layered cost interpretation.

The planners work on a down-sampled *planning grid* (default 10 cm) so that A*
stays cheap in pure Python.  Three Nav2 costmap ideas are reproduced in a
simplified form:

* ``free``   - cells that the robot footprint can occupy (static obstacle layer
  + inflation layer at the inscribed radius),
* ``cost``   - traversal cost, higher close to obstacles (inflation gradient),
  which keeps the path centred in aisles,
* replanning with a relaxed inflation radius when the strict plan fails
  (the offline analogue of Nav2 clearing/replanning the costmap).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle
from .maps import GridMap, dilate

SQRT2 = math.sqrt(2.0)


@dataclass
class PlanningGrid:
    """Down-sampled occupancy grid used by A*."""

    name: str
    free: np.ndarray  # True == navigable
    cost: np.ndarray  # traversal multiplier (>= 1.0)
    resolution: float
    origin: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def world_to_plan(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        return row, col

    def plan_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]


def build_planning_grid(
    grid: GridMap,
    resolution: float = 0.1,
    inflation_radius: float = 0.34,
    soft_radius: float = 0.62,
) -> PlanningGrid:
    """Down-sample a fine occupancy grid and build free/cost layers."""
    fine = grid.resolution
    factor = max(1, int(round(resolution / fine)))
    occ = grid.occupancy
    h, w = occ.shape
    h2 = (h + factor - 1) // factor
    w2 = (w + factor - 1) // factor
    padded = np.zeros((h2 * factor, w2 * factor), dtype=bool)
    padded[:h, :w] = occ
    coarse = padded.reshape(h2, factor, w2, factor).any(axis=(1, 3))

    core = dilate(coarse, inflation_radius / resolution)
    soft = dilate(coarse, soft_radius / resolution)
    free = ~core
    cost = np.where(soft, 4.0, 1.0).astype(np.float64)
    return PlanningGrid(
        name=grid.name,
        free=free,
        cost=cost,
        resolution=resolution,
        origin=(grid.origin[0], grid.origin[1]),
    )


def _nearest_free(pgrid: PlanningGrid, row: int, col: int, max_radius: int = 25) -> tuple[int, int] | None:
    if pgrid.inside(row, col) and pgrid.free[row, col]:
        return row, col
    h, w = pgrid.shape
    for r in range(1, max_radius + 1):
        row0, row1 = max(0, row - r), min(h, row + r + 1)
        col0, col1 = max(0, col - r), min(w, col + r + 1)
        best = None
        best_d = 1e18
        for rr in range(row0, row1):
            for cc in range(col0, col1):
                if pgrid.free[rr, cc]:
                    d = (rr - row) ** 2 + (cc - col) ** 2
                    if d < best_d:
                        best_d = d
                        best = (rr, cc)
        if best is not None:
            return best
    return None


def astar(
    pgrid: PlanningGrid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_expansions: int = 60000,
) -> list[tuple[float, float]] | None:
    """A* on the planning grid.  Returns a world-coordinate path or ``None``."""
    start = pgrid.world_to_plan(*start_xy)
    goal = pgrid.world_to_plan(*goal_xy)
    start = _nearest_free(pgrid, *start)
    goal = _nearest_free(pgrid, *goal)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start_xy, goal_xy]

    h, w = pgrid.shape
    free = pgrid.free
    cost = pgrid.cost
    n_cells = h * w
    start_idx = start[0] * w + start[1]
    goal_idx = goal[0] * w + goal[1]

    g_score = np.full(n_cells, np.inf, dtype=np.float64)
    parent = np.full(n_cells, -1, dtype=np.int64)
    closed = np.zeros(n_cells, dtype=bool)
    g_score[start_idx] = 0.0

    def heuristic(idx: int) -> float:
        r, c = divmod(int(idx), w)
        dr, dc = abs(r - goal[0]), abs(c - goal[1])
        return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)

    open_heap: list[tuple[float, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start_idx), counter, start_idx))
    expansions = 0
    neighbours = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, SQRT2),
        (-1, 1, SQRT2),
        (1, -1, SQRT2),
        (1, 1, SQRT2),
    )
    while open_heap:
        _f, _counter, idx = heapq.heappop(open_heap)
        if closed[idx]:
            continue
        if idx == goal_idx:
            break
        closed[idx] = True
        expansions += 1
        if expansions > max_expansions:
            return None
        r, c = divmod(idx, w)
        for dr, dc, step_cost in neighbours:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nidx = nr * w + nc
            if not free[nr, nc] or closed[nidx]:
                continue
            tentative = g_score[idx] + step_cost * cost[nr, nc]
            if tentative < g_score[nidx] - 1e-12:
                g_score[nidx] = tentative
                parent[nidx] = idx
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(nidx), counter, nidx))
    if parent[goal_idx] < 0 and goal_idx != start_idx:
        return None

    # reconstruct
    cells = [goal_idx]
    cur = goal_idx
    while cur != start_idx:
        cur = int(parent[cur])
        if cur < 0:
            return None
        cells.append(cur)
    cells.reverse()
    path = [pgrid.plan_to_world(*divmod(c, w)) for c in cells]
    # anchor exact endpoints so the controller converges precisely
    path[0] = tuple(start_xy)
    path[-1] = tuple(goal_xy)
    return path


def line_of_sight_plan(pgrid: PlanningGrid, a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Check a straight segment against the free layer using DDA grid traversal."""
    r0, c0 = pgrid.world_to_plan(*a)
    r1, c1 = pgrid.world_to_plan(*b)
    if not (pgrid.inside(r0, c0) and pgrid.inside(r1, c1)):
        return False
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return pgrid.free[r0, c0]
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r0 + t * dr))
        c = int(round(c0 + t * dc))
        if not pgrid.free[r, c]:
            return False
    return True


def smooth_path(
    pgrid: PlanningGrid, path: list[tuple[float, float]], max_iterations: int = 200
) -> list[tuple[float, float]]:
    """String-pulling shortcutting (greedy farthest visible point)."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    n = len(path)
    iterations = 0
    while i < n - 1 and iterations < max_iterations:
        iterations += 1
        j = n - 1
        while j > i + 1:
            if line_of_sight_plan(pgrid, path[i], path[j]):
                break
            j -= 1
        out.append(path[j])
        i = j
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def path_length(path: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def project_to_path(path: list[tuple[float, float]], point: tuple[float, float]) -> tuple[float, float, tuple[float, float]]:
    """Project a point onto the polyline.

    Returns ``(distance, arc_length_at_projection, projected_point)``.  This is
    what a path-following controller should use instead of the raw closest
    vertex: the closest *vertex* jumps when the robot cuts a corner or
    overshoots, which makes the lookahead carrot flip behind the robot.
    """
    px, py = point
    best_d = float("inf")
    best_s = 0.0
    best_xy = path[0]
    s_acc = 0.0
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        seg_len = math.sqrt(seg_len2)
        if seg_len2 < 1e-12:
            s_acc += seg_len
            continue
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        cx, cy = ax + t * dx, ay + t * dy
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d = d
            best_s = s_acc + t * seg_len
            best_xy = (cx, cy)
        s_acc += seg_len
    return best_d, best_s, best_xy


def point_at_distance(path: list[tuple[float, float]], distance: float) -> tuple[float, float]:
    """Point ``distance`` metres along the polyline from the start."""
    if distance <= 0.0:
        return path[0]
    remaining = distance
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg >= remaining:
            t = remaining / max(seg, 1e-9)
            return (ax + t * (bx - ax), ay + t * (by - ay))
        remaining -= seg
    return path[-1]


def point_to_path_distance(path: list[tuple[float, float]], point: tuple[float, float]) -> tuple[float, int]:
    """Return (distance, index of the closest segment start)."""
    if not path:
        return float("inf"), 0
    best_d = float("inf")
    best_i = 0
    px, py = point
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        cx, cy = ax + t * dx, ay + t * dy
        d = math.hypot(px - cx, py - cy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_d, best_i


def closest_path_index(
    path: list[tuple[float, float]], point: tuple[float, float], start_index: int = 0
) -> tuple[int, float]:
    """Closest path vertex at/after ``start_index`` (monotonic following)."""
    best_i = start_index
    best_d = float("inf")
    for i in range(start_index, len(path)):
        d = math.hypot(path[i][0] - point[0], path[i][1] - point[1])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def lookahead_point(
    path: list[tuple[float, float]], index: int, distance: float
) -> tuple[float, float]:
    """Walk ``distance`` metres along the path from vertex ``index``."""
    if index >= len(path) - 1:
        return path[-1]
    remaining = distance
    i = index
    while i < len(path) - 1:
        ax, ay = path[i]
        bx, by = path[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg >= remaining:
            t = remaining / max(seg, 1e-9)
            return (ax + t * (bx - ax), ay + t * (by - ay))
        remaining -= seg
        i += 1
    return path[-1]


def heading_of_segment(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def rotate_xy(x: float, y: float, theta: float) -> tuple[float, float]:
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


def wrap_pi(a: float) -> float:
    return wrap_angle(a)
