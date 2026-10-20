"""Evaluation harness: 30 navigation trials + 20 occlusion re-identification events.

Usage::

    python -m navfollow.evaluate --full        # ~5-8 min on a 2-core CPU
    python -m navfollow.evaluate --quick       # CI smoke run, ~40 s
    python -m navfollow.evaluate --quick --no-demos

Writes ``reports/eval_YYYYMMDD.json`` and ``reports/eval_YYYYMMDD.md``.  Every
number in the report is measured by running the simulator; nothing is hand
filled, and the ROS2 layer is explicitly marked as "not executed here".
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .maps import MAP_NAMES
from .simulation import run_follow_episode, run_navigation_trial

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q)) if values else 0.0


# ---------------------------------------------------------------------------
def run_navigation_suite(
    maps: tuple[str, ...] = MAP_NAMES,
    trials_per_map: int = 10,
    seed_base: int = 1000,
    n_pedestrians: int = 2,
    verbose: bool = True,
) -> dict:
    trials = []
    t0 = time.time()
    for map_name in maps:
        for k in range(trials_per_map):
            seed = seed_base + k
            result = run_navigation_trial(map_name, seed=seed, n_pedestrians=n_pedestrians)
            trials.append(result)
            if verbose:
                print(
                    f"  nav {map_name:12s} seed={seed:5d} success={int(result.success)} "
                    f"collision={int(result.collision)} t={result.time_s:5.1f}s "
                    f"ratio={result.path_ratio:.2f} rec={result.recoveries}"
                )
    total_time = time.time() - t0
    successes = [t for t in trials if t.success]
    collisions = [t for t in trials if t.collision]
    per_map: dict[str, dict] = {}
    for map_name in maps:
        sub = [t for t in trials if t.map_name == map_name]
        s = [t for t in sub if t.success]
        per_map[map_name] = {
            "trials": len(sub),
            "successes": len(s),
            "success_rate": round(len(s) / len(sub), 3) if sub else 0.0,
            "collisions": sum(1 for t in sub if t.collision),
            "mean_time_s": round(float(np.mean([t.time_s for t in sub])), 2) if sub else 0.0,
            "mean_path_ratio_success": round(float(np.mean([t.path_ratio for t in s])), 3) if s else 0.0,
            "recoveries_total": sum(t.recoveries for t in sub),
        }
    return {
        "trials": len(trials),
        "successes": len(successes),
        "success_rate": round(len(successes) / len(trials), 3) if trials else 0.0,
        "collisions": len(collisions),
        "collision_rate": round(len(collisions) / len(trials), 3) if trials else 0.0,
        "timed_out": sum(1 for t in trials if t.timed_out),
        "failed": sum(1 for t in trials if not t.success),
        "mean_time_s": round(float(np.mean([t.time_s for t in trials])), 2) if trials else 0.0,
        "mean_time_success_s": round(float(np.mean([t.time_s for t in successes])), 2) if successes else 0.0,
        "mean_path_ratio_success": round(float(np.mean([t.path_ratio for t in successes])), 3)
        if successes
        else 0.0,
        "median_path_ratio_success": round(float(np.median([t.path_ratio for t in successes])), 3)
        if successes
        else 0.0,
        "max_path_ratio_success": round(float(np.max([t.path_ratio for t in successes])), 3)
        if successes
        else 0.0,
        "recoveries_total": sum(t.recoveries for t in trials),
        "trials_with_recovery": sum(1 for t in trials if t.recoveries > 0),
        "replan_total": sum(t.replan_count for t in trials),
        "per_map": per_map,
        "wall_time_s": round(total_time, 1),
        "trials_detail": [t.to_dict() for t in trials],
    }


def run_occlusion_suite(
    episodes: int = 10,
    events_per_episode: int = 2,
    episode_duration: float = 26.0,
    seed_base: int = 2000,
    verbose: bool = True,
) -> dict:
    results = []
    t0 = time.time()
    for e in range(episodes):
        script = []
        for j in range(events_per_episode):
            duration = 1.0 + 2.0 * ((e * 7 + j * 3) % 21) / 20.0  # deterministic 1.0-3.0 s
            start = 7.0 + j * (episode_duration - 11.0) / max(1, events_per_episode)
            script.append((round(start, 2), round(duration, 2)))
        result = run_follow_episode(
            seed=seed_base + e,
            duration=episode_duration,
            occlusion_script=script,
            collect_frames=True,
        )
        results.append(result)
        if verbose:
            print(
                f"  occ episode {e:2d}: events={len(result.events)} "
                f"success={sum(1 for x in result.events if x.success)}/{len(result.events)} "
                f"track_ratio={result.tracking_ratio:.2f} "
                f"dist_err={result.to_dict()['distance_error_mean']:.3f} "
                f"wall={result.wall_time_s:.1f}s"
            )
    total_time = time.time() - t0
    all_events = [ev for r in results for ev in r.events]
    successes = [ev for ev in all_events if ev.success]
    recovery_times = [ev.recovery_time for ev in successes if ev.recovery_time is not None]
    errors = [e for r in results for e in r.distance_errors]
    durations = [ev.hide_duration for ev in all_events]
    return {
        "episodes": len(results),
        "events": len(all_events),
        "events_per_episode": events_per_episode,
        "occlusion_duration_min_s": round(min(durations), 2) if durations else 0.0,
        "occlusion_duration_max_s": round(max(durations), 2) if durations else 0.0,
        "reacquire_successes": len(successes),
        "reacquire_rate": round(len(successes) / len(all_events), 3) if all_events else 0.0,
        "identity_correct": sum(1 for ev in successes if ev.identity_correct),
        "recovery_time_mean_s": round(float(np.mean(recovery_times)), 2) if recovery_times else 0.0,
        "recovery_time_p95_s": round(_percentile(recovery_times, 95), 2),
        "distance_error_mean_m": round(float(np.mean(errors)), 3) if errors else 0.0,
        "distance_error_p95_m": round(_percentile(errors, 95), 3),
        "distance_error_max_m": round(float(np.max(errors)), 3) if errors else 0.0,
        "distance_samples": len(errors),
        "distance_settling_excluded_s": 2.0,
        "tracking_ratio_mean": round(float(np.mean([r.tracking_ratio for r in results])), 3)
        if results
        else 0.0,
        "collisions": sum(r.collisions for r in results),
        "wall_time_s": round(total_time, 1),
        "events_detail": [ev.to_dict() for ev in all_events],
        "episodes_detail": [r.to_dict() for r in results],
        "episode_transitions": [r.transitions for r in results],
    }


# ---------------------------------------------------------------------------
def _environment() -> dict:
    import cv2

    return {
        "host": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cpu_count": os.cpu_count(),
        "simulation": "pure-python offline simulator (no ROS2 / Gazebo / GPU)",
        "ros2_executed": False,
        "notes": [
            "All navigation / perception / following numbers were produced by the "
            "offline simulator in this repository on the host above.",
            "The ROS2 ament_python package and Gazebo world in ros2_ws/ are provided "
            "as the deployment path and were NOT executed on this machine (no ROS2).",
            "The YOLOv8 detector adapter is optional and not used in these numbers: "
            "the CPU-only evaluation uses the synthetic-camera colour detector.",
        ],
    }


def _markdown_report(report: dict) -> str:
    nav = report["navigation"]
    occ = report["occlusion"]
    env = report["environment"]
    lines = [
        f"# Evaluation report - {report['date']}",
        "",
        f"Mode: **{report['mode']}** | generated {report['generated_at']} | "
        f"total wall time **{report['wall_time_s']} s**",
        "",
        "All numbers below are measured by running `navfollow.simulation` on this machine "
        "(no ROS2, no Gazebo, no GPU). The synthetic-camera perception pipeline is the "
        "same code that runs behind the ROS2 nodes; only the detector (colour blob vs "
        "YOLOv8) and the renderer differ. See `docs/evaluation.md` for methodology.",
        "",
        "## 1. Navigation trials",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| trials | {nav['trials']} ({len(nav['per_map'])} maps x {nav['trials'] // max(len(nav['per_map']), 1)}) |",
        f"| success rate | **{nav['success_rate'] * 100:.1f}%** ({nav['successes']}/{nav['trials']}) |",
        f"| collisions | **{nav['collisions']}** (rate {nav['collision_rate'] * 100:.1f}%) |",
        f"| mean time per trial | {nav['mean_time_s']:.1f} s |",
        f"| mean path length ratio (successes) | **{nav['mean_path_ratio_success']:.2f}** "
        f"(median {nav.get('median_path_ratio_success', 0.0):.2f}, max {nav['max_path_ratio_success']:.2f}) |",
        f"| recovery behaviour activations | {nav['recoveries_total']} "
        f"({nav['trials_with_recovery']} trials needed recovery) |",
        f"| total global plans | {nav['replan_total']} |",
        "",
        "Per map:",
        "",
        "| map | trials | successes | success rate | collisions | mean time | mean path ratio | recoveries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, data in nav["per_map"].items():
        lines.append(
            f"| {name} | {data['trials']} | {data['successes']} | {data['success_rate'] * 100:.0f}% | "
            f"{data['collisions']} | {data['mean_time_s']:.1f} s | {data['mean_path_ratio_success']:.2f} | "
            f"{data['recoveries_total']} |"
        )
    lines += [
        "",
        "## 2. Occlusion / re-identification",
        "",
        f"{occ['events']} scripted occlusions of 1.0-3.0 s across {occ['episodes']} follow episodes.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| occlusion events | {occ['events']} (duration {occ['occlusion_duration_min_s']:.1f}-"
        f"{occ['occlusion_duration_max_s']:.1f} s) |",
        f"| re-identification success | **{occ['reacquire_successes']}/{occ['events']}** "
        f"({occ['reacquire_rate'] * 100:.0f}%) |",
        f"| correct identity on reacquisition | {occ['identity_correct']}/{occ['events']} |",
        f"| recovery time (reappear -> TRACK) | mean **{occ['recovery_time_mean_s']:.2f} s**, "
        f"p95 {occ['recovery_time_p95_s']:.2f} s |",
        f"| follow distance error | mean **{occ['distance_error_mean_m']:.3f} m**, "
        f"p95 **{occ['distance_error_p95_m']:.3f} m**, max {occ['distance_error_max_m']:.3f} m "
        f"({occ['distance_samples']} samples, 2 s settling window after each (re)acquisition excluded) |",
        f"| tracking-ratio (frames in TRACK) | {occ['tracking_ratio_mean'] * 100:.1f}% |",
        f"| collisions during follow | {occ['collisions']} |",
        "",
        "## 3. Environment",
        "",
        f"- Host: `{env['host']}`",
        f"- Python {env['python']}, numpy {env['numpy']}, OpenCV {env['opencv']}, CPU cores {env['cpu_count']}",
        f"- Simulation: {env['simulation']}",
        f"- ROS2 executed: **{env['ros2_executed']}** (see `docs/ros2_integration.md`)",
        "",
        "## 4. Caveats",
        "",
        "- The template numbers from the resume draft (95% navigation success, 85% re-ID "
        "recovery, <0.2 m distance error) are **not** the targets of this evaluation and are "
        "**not** reproduced here; only measured values are reported.",
        "- DWA/A* are simplified stand-ins for Nav2's planner/controller and the recovery "
        "behaviour is a simplified Nav2 behaviour tree.",
        "- The offline detector is HSV colour segmentation on a synthetic renderer; the "
        "YOLOv8 adapter exists but is excluded from the automated numbers.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline evaluation suite")
    parser.add_argument("--quick", action="store_true", help="CI smoke mode (small suites)")
    parser.add_argument("--nav-trials", type=int, default=None, help="trials per map")
    parser.add_argument("--occlusion-episodes", type=int, default=None)
    parser.add_argument("--out", type=str, default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--no-demos", action="store_true", help="skip demo GIF generation")
    parser.add_argument("--seed-base", type=int, default=1000)
    args = parser.parse_args(argv)

    quick = args.quick
    trials_per_map = args.nav_trials if args.nav_trials is not None else (2 if quick else 10)
    episodes = args.occlusion_episodes if args.occlusion_episodes is not None else (2 if quick else 10)
    episode_duration = 20.0 if quick else 26.0

    t_start = time.time()
    print(f"[eval] mode={'quick' if quick else 'full'} maps={MAP_NAMES} "
          f"nav_trials/map={trials_per_map} occlusion_episodes={episodes}")

    print("[eval] navigation suite")
    navigation = run_navigation_suite(trials_per_map=trials_per_map, verbose=True)
    print("[eval] occlusion suite")
    occlusion = run_occlusion_suite(episodes=episodes, episode_duration=episode_duration, verbose=True)

    now = datetime.now()
    report = {
        "date": now.strftime("%Y%m%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "quick" if quick else "full",
        "version": __version__,
        "wall_time_s": round(time.time() - t_start, 1),
        "environment": _environment(),
        "navigation": navigation,
        "occlusion": occlusion,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_{now.strftime('%Y%m%d')}.json"
    md_path = out_dir / f"eval_{now.strftime('%Y%m%d')}.md"
    payload = dict(report)
    payload["navigation"] = {k: v for k, v in navigation.items() if k != "trials_detail"}
    payload["occlusion"] = {k: v for k, v in occlusion.items() if k not in ("episodes_detail", "episode_transitions")}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")

    if not args.no_demos:
        print("[eval] generating demo GIFs")
        try:
            from .demo import generate_demos

            generate_demos(out_dir.parent / "demos", quick=quick)
        except Exception as exc:  # pragma: no cover - demo generation is best effort
            print(f"[eval] WARNING: demo generation failed: {exc}")

    nav = report["navigation"]
    occ = report["occlusion"]
    print("\n===== measured results =====")
    print(f"navigation: {nav['successes']}/{nav['trials']} success "
          f"({nav['success_rate'] * 100:.1f}%), collisions={nav['collisions']}, "
          f"path_ratio={nav['mean_path_ratio_success']:.2f}")
    print(f"occlusion:  {occ['reacquire_successes']}/{occ['events']} reacquired "
          f"({occ['reacquire_rate'] * 100:.0f}%), recovery={occ['recovery_time_mean_s']:.2f}s, "
          f"dist_err mean={occ['distance_error_mean_m']:.3f} m p95={occ['distance_error_p95_m']:.3f} m")
    print(f"total wall time: {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
