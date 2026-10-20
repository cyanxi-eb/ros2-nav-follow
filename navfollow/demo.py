"""Generate the three demo GIFs from real simulation runs.

    python -m navfollow.demo            # all demos
    python -m navfollow.demo --quick    # shorter versions for CI

Outputs (each <= 3.5 MB):

* ``demos/01_nav_path.gif``      - occupancy/costmap, A* plan, DWA tracking,
* ``demos/02_follow_occlusion.gif`` - follow -> occlusion -> Kalman + ReID re-acquire,
* ``demos/03_recovery.gif``      - blocked aisle -> Nav2-style recovery -> replan around.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .geometry import Pose2D
from .maps import load_map
from .planning import point_at_distance, project_to_path
from .render import (
    TopDownRenderer,
    add_header,
    camera_panel,
    depth_panel,
    hstack_panels,
    save_gif,
    status_panel,
    vstack_panels,
)
from .robot import integrate_pose
from .simulation import FOLLOW_START, build_follow_scene, run_follow_episode, run_navigation_trial
from .world import SimWorld

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demos"
LIDAR_ANGLES = (-2.35, 2.35)


# ---------------------------------------------------------------------------
def demo_navigation(out_path: Path, quick: bool = False) -> dict:
    """Demo 1: map + costmap inflation + A* global plan + DWA tracking."""
    grid = load_map("warehouse_a")
    renderer = TopDownRenderer(grid)
    frames: list[np.ndarray] = []
    state = {"i": 0}
    skip = 4

    def on_frame(info: dict) -> None:
        state["i"] += 1
        if state["i"] % skip:
            return
        pose = info["pose"]
        t = info["t"]
        captions = [
            f"t={t:5.1f}s  pose=({pose.x:4.1f},{pose.y:4.1f})  v={info['cmd'][0]:.2f} m/s",
            f"A* waypoints={len(info['path']) if info['path'] else 0}  "
            f"replans={info['stack'].replan_count}  recoveries={info['stack'].recoveries}",
        ]
        frame = renderer.render(
            pose=pose,
            persons=_person_objects(info) if info["persons"] else [],
            path=info["path"],
            planned_paths=info["planned_paths"],
            lidar_scan=info["scan"],
            lidar_angles=LIDAR_ANGLES,
            goal=info["goal"],
            captions=captions,
            target_person_id=None,
        )
        frames.append(add_header(frame, ["DEMO 1 | A* global plan + DWA local planner (Nav2-style)"]))

    result = run_navigation_trial(
        "warehouse_a",
        seed=7,
        n_pedestrians=2,
        max_time=70.0,
        start=(1.8, 6.5, 0.0),
        goal=(18.3, 6.5),
        on_frame=on_frame,
        collect_scans=True,
    )
    if quick:
        frames = frames[:60]
    stats = save_gif(frames, str(out_path), fps=10, max_bytes=3_500_000)
    print(f"[demo] 01 nav: success={result.success} time={result.time_s:.1f}s "
          f"ratio={result.path_ratio:.2f} frames={stats['frames']}")
    return stats


def _person_objects(info: dict):
    """Render-time light-weight person objects (colour + id for the renderer)."""

    class _P:
        def __init__(self, x, y, radius, color, person_id: int):
            self.x = x
            self.y = y
            self.radius = radius
            self.color_rgb = color
            self.person_id = person_id
            self.hidden = False

    return [
        _P(x, y, radius, color, idx + 100)
        for idx, (x, y, radius, color) in enumerate(info["persons"])
    ]


# ---------------------------------------------------------------------------
def demo_follow_occlusion(out_path: Path, quick: bool = False) -> dict:
    """Demo 2: tracking, occlusion (Kalman extrapolation), ReID re-acquisition."""
    grid = load_map("warehouse_a")
    renderer = TopDownRenderer(grid)
    frames: list[np.ndarray] = []
    state = {"i": 0}
    skip = 5 if quick else 3

    def on_frame(info: dict) -> None:
        state["i"] += 1
        if state["i"] % skip:
            return
        cmd = info["cmd"]
        frame = info["frame"]
        pose = info["pose"]
        top = renderer.render(
            pose=pose,
            persons=info["world"].persons,
            lidar_scan=info["world"].lidar_scan(pose),
            lidar_angles=LIDAR_ANGLES,
            target_xy=cmd.target_world,
            predicted_xy=cmd.predicted_world,
            captions=["yellow ring = target | orange diamond = Kalman prediction"],
            target_person_id=1,
        )
        state_colors = {
            "TRACK": (120, 235, 140),
            "OCCLUDED": (250, 185, 70),
            "REACQUIRE": (250, 230, 90),
            "SEARCH": (190, 190, 190),
            "ARRIVED": (110, 220, 240),
            "STOP": (250, 110, 110),
        }
        status = status_panel(
            top.shape[1],
            [
                (f"FOLLOW STATE: {cmd.state.value}", state_colors.get(cmd.state.value, (240, 240, 240))),
                (
                    f"t={info['t']:5.1f}s   v={info['vel'][0]:+.2f} m/s   "
                    f"dist={cmd.distance if cmd.distance is None else round(cmd.distance, 2)} m",
                    (220, 220, 225),
                ),
                (
                    f"occluded={int(info['occluded'])}   re-acquired events="
                    f"{sum(1 for ev in info['stack'].events if ev.get('type') == 'reacquired')}",
                    (200, 205, 215),
                ),
            ],
        )
        left = vstack_panels([top, status], gap=4)
        cam = camera_panel(frame.rgb, frame.detections, frame.target_track_id, scale=1.5)
        dep = depth_panel(frame.depth, scale=1.5)
        right = vstack_panels([cam, dep], gap=4)
        frames.append(add_header(hstack_panels([left, right], gap=6),
                                 ["DEMO 2 | follow -> occlusion (Kalman) -> ReID re-acquire"]))

    result = run_follow_episode(
        duration=22.0 if quick else 30.0,
        occlusion_script=[(7.0, 2.0), (17.0, 2.6)] if not quick else [(6.0, 2.0)],
        collect_frames=False,
        on_frame=on_frame,
    )
    stats = save_gif(frames, str(out_path), fps=8, max_bytes=3_500_000)
    print(f"[demo] 02 follow: events={len(result.events)} "
          f"success={sum(1 for e in result.events if e.success)}/{len(result.events)} "
          f"dist_err={result.to_dict()['distance_error_mean']:.3f} m frames={stats['frames']}")
    return stats


# ---------------------------------------------------------------------------
def _add_barrier(grid, x0: float, y0: float, x1: float, y1: float) -> None:
    r0, c0 = grid.world_to_grid(x0, y0)
    r1, c1 = grid.world_to_grid(x1, y1)
    grid.occupancy[min(r0, r1) : max(r0, r1) + 1, min(c0, c1) : max(c0, c1) + 1] = True
    grid.build_derived()


def demo_recovery(out_path: Path, quick: bool = False) -> dict:
    """Demo 3: the aisle is blocked mid-run -> stall -> spin/backup -> replan."""
    grid = load_map("warehouse_a")
    renderer = TopDownRenderer(grid)
    frames: list[np.ndarray] = []
    state = {"i": 0, "barrier": False, "flash_until": -1.0, "pallet": None, "last_rec": 0}
    skip = 4

    def on_frame(info: dict) -> None:
        state["i"] += 1
        pose = info["pose"]
        t = info["t"]
        stack = info["stack"]
        # Insert a pallet on the *tracked* path ~2.2 m ahead of the robot.  The
        # current path is NOT deleted, so the controller really has to stall and
        # run the recovery behaviour before the next global replan.
        if not state["barrier"] and t >= 8.0 and stack.path and len(stack.path) >= 2:
            _dev, s_here, _proj = project_to_path(stack.path, (pose.x, pose.y))
            point = point_at_distance(stack.path, s_here + 2.2)
            _add_barrier(info["grid"], point[0] - 0.6, point[1] - 0.6, point[0] + 0.6, point[1] + 0.6)
            stack.refresh_maps()
            state["barrier"] = True
            state["pallet"] = point
            print(f"[demo] 03: inserted blocking pallet at t={t:.1f}s near "
                  f"({point[0]:.1f},{point[1]:.1f}); path is now blocked")
        if state["i"] % skip:
            return
        if stack.recoveries > state["last_rec"]:
            state["last_rec"] = stack.recoveries
            state["flash_until"] = t + 2.0
        latest = stack.events[-1]["type"] if stack.events else "-"
        captions = [
            f"t={t:5.1f}s  AISLE BLOCKED -> recoveries={stack.recoveries}",
            f"last_event={latest}  (clear costmap -> spin -> back up -> replan)",
        ]
        frame = renderer.render(
            pose=pose,
            persons=_person_objects(info) if info["persons"] else [],
            path=info["path"],
            planned_paths=info["planned_paths"],
            lidar_scan=info["scan"],
            lidar_angles=LIDAR_ANGLES,
            goal=info["goal"],
            captions=captions,
            target_person_id=None,
            flash=t < state["flash_until"],
        )
        if state.get("pallet") is not None:
            px, py = renderer.w2p(*state["pallet"])
            cv2.drawMarker(frame, (px, py), (220, 30, 30), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(frame, "BLOCKED", (px - 30, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 20, 20), 1, cv2.LINE_AA)
        frames.append(add_header(frame, ["DEMO 3 | blocked path -> recovery behaviour -> replan"]))

    result = run_navigation_trial(
        "warehouse_a",
        seed=11,
        n_pedestrians=0,
        max_time=90.0,
        start=(1.8, 6.5, 0.0),
        goal=(18.3, 6.5),
        on_frame=on_frame,
        collect_scans=True,
    )
    if quick:
        frames = frames[:80]
    stats = save_gif(frames, str(out_path), fps=10, max_bytes=3_500_000)
    print(f"[demo] 03 recovery: success={result.success} recoveries={result.recoveries} "
          f"time={result.time_s:.1f}s frames={stats['frames']}")
    return stats


# ---------------------------------------------------------------------------
def generate_demos(out_dir: Path | None = None, quick: bool = False) -> list[dict]:
    out = Path(out_dir) if out_dir is not None else DEMO_DIR
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    results.append(demo_navigation(out / "01_nav_path.gif", quick=quick))
    results.append(demo_follow_occlusion(out / "02_follow_occlusion.gif", quick=quick))
    results.append(demo_recovery(out / "03_recovery.gif", quick=quick))
    print(f"[demo] all demos generated in {time.time() - t0:.1f}s -> {out}")
    for r in results:
        print(f"  {Path(r['path']).name}: {r['bytes'] / 1e6:.2f} MB, {r['frames']} frames")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate demo GIFs from simulation runs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=str, default=str(DEMO_DIR))
    args = parser.parse_args(argv)
    generate_demos(Path(args.out), quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
