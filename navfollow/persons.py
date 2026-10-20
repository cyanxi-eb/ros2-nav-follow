"""Scripted persons (dynamic obstacles / follow targets) for the offline sim.

A person walks a polyline route at constant speed.  Routes are used for

* dynamic obstacles during navigation trials (pedestrians crossing aisles),
* the follow target in the person-following scenario,
* the follow target in the occlusion/re-identification evaluation, where the
  scenario can additionally hide the person for 1-3 s to reproduce a
  "walked behind a shelf" occlusion deterministically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# jacket colours used by the colour/HSV detector; mutually distinct in hue
PERSON_COLORS = {
    "red": (205, 45, 45),
    "blue": (40, 90, 200),
    "green": (40, 165, 70),
    "orange": (225, 135, 35),
    "magenta": (180, 55, 170),
}


@dataclass
class Person:
    """A pedestrian walking a polyline route."""

    person_id: int
    color_name: str = "red"
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed: float = 0.65
    radius: float = 0.28
    height: float = 1.70
    loop: bool = True
    start_index: int = 0
    # scripted occlusion (used by the re-identification evaluation)
    hidden: bool = False
    # runtime state
    x: float = 0.0
    y: float = 0.0
    waypoint_index: int = 0
    arrived: bool = False
    distance_walked: float = 0.0
    paused_remaining: float = 0.0
    # pedestrians yield to the robot when it gets very close (avoids scripted
    # walkers "ramming" a stopped robot, which would pollute collision metrics)
    yield_radius: float = 0.75

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("person needs at least one waypoint")
        self.waypoint_index = min(self.start_index, len(self.waypoints) - 1)
        self.x, self.y = self.waypoints[self.waypoint_index]

    @property
    def color_rgb(self) -> tuple[int, int, int]:
        return PERSON_COLORS[self.color_name]

    @property
    def pose(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def yaw(self) -> float:
        if self.waypoint_index + 1 < len(self.waypoints):
            tx, ty = self.waypoints[self.waypoint_index + 1]
        elif len(self.waypoints) > 1:
            tx, ty = self.waypoints[0]
        else:
            tx, ty = self.x + 1.0, self.y
        return math.atan2(ty - self.y, tx - self.x)

    # ------------------------------------------------------------------
    def step(self, dt: float) -> None:
        if self.paused_remaining > 0.0:
            self.paused_remaining = max(0.0, self.paused_remaining - dt)
            return
        if self.arrived:
            return
        remaining = self.speed * dt
        guard = 0
        while remaining > 1e-9 and guard < 10:
            guard += 1
            if self.waypoint_index + 1 >= len(self.waypoints):
                if self.loop:
                    self.waypoint_index = 0
                else:
                    self.arrived = True
                    return
            tx, ty = self.waypoints[self.waypoint_index + 1]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                self.waypoint_index += 1
                continue
            if dist <= remaining:
                self.x, self.y = tx, ty
                self.distance_walked += dist
                remaining -= dist
                self.waypoint_index += 1
                if self.waypoint_index + 1 >= len(self.waypoints) and not self.loop:
                    self.arrived = True
                    return
            else:
                self.x += dx / dist * remaining
                self.y += dy / dist * remaining
                self.distance_walked += remaining
                remaining = 0.0

    def pause(self, duration: float) -> None:
        self.paused_remaining = max(self.paused_remaining, float(duration))


def make_target_person(route: list[tuple[float, float]], color: str = "red") -> Person:
    return Person(person_id=1, color_name=color, waypoints=route, speed=0.62, loop=True)


def make_pedestrians(rng: np.random.Generator, grid, n: int = 2, seed_route: int | None = None) -> list[Person]:
    """Create loop pedestrians for navigation trials.

    Routes are straight back-and-forth lines through free space, chosen from a
    fixed candidate list so that trials stay reproducible.
    """
    from .maps import GridMap

    assert isinstance(grid, GridMap)
    free = grid.is_free
    w, h = grid.size_m
    candidates = [
        [(1.5, 6.5), (w - 1.8, 6.5)],
        [(1.5, 5.9), (w - 1.8, 5.9)],
        [(10.2, 1.4), (10.2, h - 1.4)],
        [(15.2, 1.4), (15.2, h - 1.4)],
        [(2.2, 1.4), (2.2, h - 1.4)],
        [(1.4, 10.5), (w - 1.8, 10.5)],
        [(8.0, 1.5), (8.0, 5.0)],
        [(1.6, 3.0), (w - 2.2, 3.0)],
    ]
    usable = []
    for route in candidates:
        ok = True
        for i in range(len(route) - 1):
            (x0, y0), (x1, y1) = route[i], route[i + 1]
            steps = int(math.hypot(x1 - x0, y1 - y0) / 0.15) + 2
            for s in range(steps + 1):
                t = s / steps
                if not free(x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.30):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            usable.append(route)
    persons: list[Person] = []
    rng_local = np.random.default_rng(seed_route) if seed_route is not None else rng
    for i in range(min(n, len(usable))):
        route = usable[int(rng_local.integers(0, len(usable)))]
        color = ["blue", "green", "orange", "magenta"][i % 4]
        person = Person(
            person_id=100 + i,
            color_name=color,
            waypoints=list(route),
            speed=float(rng_local.uniform(0.35, 0.6)),
            loop=True,
            start_index=int(rng_local.integers(0, 2)),
        )
        # start somewhere along the first segment
        person.waypoint_index = 0
        person.x, person.y = route[0]
        persons.append(person)
    return persons
