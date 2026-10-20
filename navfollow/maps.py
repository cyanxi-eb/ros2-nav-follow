"""Occupancy-grid maps of three indoor / warehouse environments.

Maps are generated procedurally with fixed parameters (deterministic, no data
files needed).  A map is a :class:`GridMap` holding a boolean occupancy array at
``resolution`` metres per cell.  We also derive:

* ``blocked_robot``  - obstacles dilated by the robot radius (collision test),
* ``blocked_margin`` - obstacles dilated by a safety margin (DWA clearance cost),
* ``plan_free``      - free space at planning resolution for A*.

The three scenes exercise different failure modes:

* ``warehouse_a`` - wide aisles, shelves and pillars; main scene for nav + follow,
* ``warehouse_b`` - denser shelves / dead ends; harder global planning,
* ``corridor_c``  - narrow 1.2 m corridors and doorways; recovery behaviour tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridMap:
    """2D occupancy grid.

    ``occupancy`` is ``True`` for occupied cells.  The origin is the centre of
    cell (0, 0) in the world frame, i.e. ``world_x = (col + 0.5) * res + origin_x``
    is *not* used; instead cell (row, col) covers
    ``[origin_x + col*res, origin_x + (col+1)*res)``.
    """

    name: str
    occupancy: np.ndarray
    resolution: float = 0.05
    origin: tuple[float, float] = (0.0, 0.0)
    # cached derived grids
    blocked_robot: np.ndarray | None = field(default=None, repr=False)
    blocked_margin: np.ndarray | None = field(default=None, repr=False)
    robot_radius: float = 0.18
    safety_margin: float = 0.30

    # -- geometry ---------------------------------------------------------
    @property
    def height(self) -> int:
        return int(self.occupancy.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def size_m(self) -> tuple[float, float]:
        return self.width * self.resolution, self.height * self.resolution

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = int(math.floor((y - self.origin[1]) / self.resolution))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def cell_occupied(self, x: float, y: float, grid: np.ndarray | None = None) -> bool:
        g = self.occupancy if grid is None else grid
        row, col = self.world_to_grid(x, y)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return True
        return bool(g[row, col])

    def occupied_points(self) -> np.ndarray:
        rows, cols = np.nonzero(self.occupancy)
        xs = self.origin[0] + (cols + 0.5) * self.resolution
        ys = self.origin[1] + (rows + 0.5) * self.resolution
        return np.stack([xs, ys], axis=1)

    # -- derived grids ----------------------------------------------------
    def build_derived(self) -> None:
        self.blocked_robot = dilate(self.occupancy, self.robot_radius / self.resolution)
        self.blocked_margin = dilate(self.occupancy, self.safety_margin / self.resolution)

    def ensure_derived(self) -> None:
        if self.blocked_robot is None or self.blocked_margin is None:
            self.build_derived()

    def robot_blocked(self, x: float, y: float) -> bool:
        self.ensure_derived()
        return self.cell_occupied(x, y, self.blocked_robot)

    def margin_blocked(self, x: float, y: float) -> bool:
        self.ensure_derived()
        return self.cell_occupied(x, y, self.blocked_margin)

    # -- sampling ---------------------------------------------------------
    def is_free(self, x: float, y: float, clearance_radius: float | None = None) -> bool:
        if clearance_radius is None or clearance_radius <= 0:
            return not self.cell_occupied(x, y)
        r = int(math.ceil(clearance_radius / self.resolution))
        row, col = self.world_to_grid(x, y)
        r0, r1 = max(0, row - r), min(self.height, row + r + 1)
        c0, c1 = max(0, col - r), min(self.width, col + r + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        return not bool(self.occupancy[r0:r1, c0:c1].any())

    def sample_free_pose(
        self, rng: np.random.Generator, clearance: float = 0.45, max_tries: int = 4000
    ) -> tuple[float, float, float]:
        for _ in range(max_tries):
            x = rng.uniform(self.origin[0], self.origin[0] + self.size_m[0])
            y = rng.uniform(self.origin[1], self.origin[1] + self.size_m[1])
            if self.is_free(x, y, clearance) and not self.robot_blocked(x, y):
                return x, y, float(rng.uniform(-math.pi, math.pi))
        raise RuntimeError(f"could not sample free pose on map {self.name}")


def dilate(occ: np.ndarray, radius_cells: float) -> np.ndarray:
    """Binary dilation with a square structuring element (L-infinity, conservative)."""
    r = int(math.ceil(max(0.0, radius_cells)))
    if r <= 0:
        return occ.copy()
    out = occ.copy()
    for _ in range(r):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        acc = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = acc
    return out


# ---------------------------------------------------------------------------
# map construction helpers
# ---------------------------------------------------------------------------
def _blank_map(name: str, width_m: float, height_m: float, res: float = 0.05) -> GridMap:
    w = int(round(width_m / res))
    h = int(round(height_m / res))
    return GridMap(name=name, occupancy=np.zeros((h, w), dtype=bool), resolution=res)


def _rect(grid: GridMap, x0: float, y0: float, x1: float, y1: float) -> None:
    r0, c0 = grid.world_to_grid(x0, y0)
    r1, c1 = grid.world_to_grid(x1, y1)
    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))
    grid.occupancy[max(0, r0) : min(grid.height, r1 + 1), max(0, c0) : min(grid.width, c1 + 1)] = True


def _wall(grid: GridMap, x0: float, y0: float, x1: float, y1: float, thickness: float = 0.2) -> None:
    if abs(x1 - x0) >= abs(y1 - y0):
        lo, hi = sorted((x0, x1))
        _rect(grid, lo, y0 - thickness / 2, hi, y0 + thickness / 2)
    else:
        lo, hi = sorted((y0, y1))
        _rect(grid, x0 - thickness / 2, lo, x0 + thickness / 2, hi)


def _border(grid: GridMap, thickness: float = 0.2) -> None:
    """Closed outer wall - guarantees no free pockets in the map corners."""
    w, h = grid.size_m
    _rect(grid, 0.0, 0.0, w, thickness)
    _rect(grid, 0.0, h - thickness, w, h)
    _rect(grid, 0.0, 0.0, thickness, h)
    _rect(grid, w - thickness, 0.0, w, h)


def make_warehouse_a() -> GridMap:
    """20 x 13 m warehouse: outer wall, shelf blocks, 3 pillars, wide central aisle.

    The central aisle is 3 m wide and contains three 0.7 m pillars, so a person
    can walk "behind" a pillar relative to the robot - this is what produces the
    natural occlusion in the follow demo.
    """
    g = _blank_map("warehouse_a", 20.0, 13.0)
    w, h = g.size_m
    _border(g, 0.2)
    # shelf rows (bottom row y 2.2-5.0, top row y 8.0-10.8, central aisle 5.0-8.0)
    shelves = [
        (3.2, 2.2, 3.9, 5.0),
        (6.2, 2.2, 6.9, 5.0),
        (9.2, 2.2, 9.9, 5.0),
        (12.2, 2.2, 12.9, 5.0),
        (15.2, 2.4, 15.9, 5.0),
        (3.2, 8.0, 3.9, 10.8),
        (6.2, 8.0, 6.9, 10.8),
        (9.2, 8.0, 9.9, 10.8),
        (12.2, 8.0, 12.9, 10.8),
        (15.2, 8.0, 15.9, 10.8),
    ]
    for x0, y0, x1, y1 in shelves:
        _rect(g, x0, y0, x1, y1)
    # pillars in the central aisle (natural occlusion for the follow scenario)
    for px, py in [(5.4, 6.5), (10.6, 6.5), (16.8, 6.5)]:
        _rect(g, px - 0.35, py - 0.35, px + 0.35, py + 0.35)
    g.build_derived()
    return g


def make_warehouse_b() -> GridMap:
    """18 x 12 m warehouse with a dead-end bay and staggered shelves."""
    g = _blank_map("warehouse_b", 18.0, 12.0)
    w, h = g.size_m
    _border(g, 0.2)
    # long horizontal shelf walls, staggered to create a maze
    _rect(g, 2.5, 2.0, 12.0, 2.7)
    _rect(g, 6.0, 4.4, 16.0, 5.1)
    _rect(g, 2.5, 7.0, 11.5, 7.7)
    _rect(g, 6.5, 9.4, 15.5, 10.1)
    # dead-end bay: three walls around a 1.4 m pocket
    _rect(g, 13.2, 2.9, 14.0, 4.2)
    _rect(g, 13.2, 4.2, 16.6, 5.0)
    # tall pillars for occlusion
    for px, py in [(4.2, 5.7), (9.0, 5.9), (15.0, 7.9)]:
        _rect(g, px - 0.3, py - 0.3, px + 0.3, py + 0.3)
    g.build_derived()
    return g


def make_corridor_c() -> GridMap:
    """16 x 11 m office/lab with one narrow 1.6 m corridor and 1.2 m doorways.

    The corridor is deliberately narrow: the robot has ~0.9 m of inflated free
    space, so local planning has to stay centred, which is what makes this map a
    good recovery-behaviour test.
    """
    g = _blank_map("corridor_c", 16.0, 11.0)
    w, h = g.size_m
    _border(g, 0.2)

    # horizontal corridor between y = 4.2 and y = 5.8 (1.6 m wide)
    # upper wall: doorways at x = 2.0-3.2, 7.0-8.2, 12.0-13.2
    _wall(g, 0.4, 5.8, 2.0, 5.8, 0.2)
    _wall(g, 3.2, 5.8, 7.0, 5.8, 0.2)
    _wall(g, 8.2, 5.8, 12.0, 5.8, 0.2)
    _wall(g, 13.2, 5.8, 15.6, 5.8, 0.2)
    # lower wall: doorways at x = 3.6-4.8, 8.6-9.8, 13.0-14.2
    _wall(g, 0.4, 4.2, 3.6, 4.2, 0.2)
    _wall(g, 4.8, 4.2, 8.6, 4.2, 0.2)
    _wall(g, 9.8, 4.2, 13.0, 4.2, 0.2)
    _wall(g, 14.2, 4.2, 15.6, 4.2, 0.2)

    # room dividers (top side and bottom side)
    _wall(g, 5.2, 5.8, 5.2, 10.85, 0.2)
    _wall(g, 10.4, 5.8, 10.4, 10.85, 0.2)
    _wall(g, 5.2, 0.4, 5.2, 4.2, 0.2)
    _wall(g, 10.4, 0.4, 10.4, 4.2, 0.2)

    # furniture (leaves >= 1 m around each piece)
    _rect(g, 1.2, 1.6, 3.0, 2.2)
    _rect(g, 6.6, 1.5, 8.2, 2.1)
    _rect(g, 12.0, 1.4, 14.2, 2.0)
    _rect(g, 1.4, 7.6, 3.0, 8.2)
    _rect(g, 6.8, 7.8, 8.0, 8.4)
    _rect(g, 12.2, 7.4, 14.4, 8.0)
    # a pillar in the narrow corridor: leaves a 1.2 m passage above it
    _rect(g, 6.4, 4.2, 7.0, 4.6)
    g.build_derived()
    return g


MAP_BUILDERS = {
    "warehouse_a": make_warehouse_a,
    "warehouse_b": make_warehouse_b,
    "corridor_c": make_corridor_c,
}

MAP_NAMES = tuple(MAP_BUILDERS.keys())


def load_map(name: str) -> GridMap:
    if name not in MAP_BUILDERS:
        raise KeyError(f"unknown map {name!r}; choose from {MAP_NAMES}")
    return MAP_BUILDERS[name]()
