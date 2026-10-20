"""Offline trial runners: navigation trials and person-following episodes.

Everything the evaluation reports is produced by these functions running the
real code paths (A*/DWA/nav recovery for navigation, camera -> detector ->
tracker -> ReID -> TF -> Kalman/PID/state machine for following).  No metric is
hand-written; ``navfollow.evaluate`` only aggregates the returned records.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .camera import CameraIntrinsics, CameraMount, SyntheticCamera
from .dwa import DWAConfig, DWAPlanner
from .follow.controller import FollowConfig, FollowController, FollowState
from .geometry import Pose2D, clamp, quaternion_from_yaw
from .maps import GridMap, load_map
from .navstack import Nav2LikeStack, NavConfig, NavStatus
from .perception.pipeline import PerceptionPipeline
from .perception.reid import ReIDConfig
from .persons import Person, make_pedestrians
from .robot import integrate_pose
from .tf_tree import build_robot_tree
from .world import LidarConfig, SimWorld

CONTROL_DT = 0.1
DEFAULT_INTRINSICS = CameraIntrinsics()
DEFAULT_MOUNT = CameraMount()


# ===========================================================================
# navigation trials
# ===========================================================================
@dataclass
class NavigationTrialResult:
    map_name: str
    seed: int
    success: bool
    collision: bool
    timed_out: bool
    time_s: float
    path_length: float
    trajectory_length: float
    path_ratio: float
    recoveries: int
    replan_count: int
    start: tuple[float, float, float]
    goal: tuple[float, float]
    final_pose: tuple[float, float, float]
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    planned_paths: list[list[tuple[float, float]]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        data = {
            "map": self.map_name,
            "seed": self.seed,
            "success": self.success,
            "collision": self.collision,
            "timed_out": self.timed_out,
            "time_s": round(self.time_s, 2),
            "path_length_m": round(self.path_length, 2),
            "trajectory_length_m": round(self.trajectory_length, 2),
            "path_ratio": round(self.path_ratio, 3),
            "recoveries": self.recoveries,
            "replans": self.replan_count,
            "start": [round(v, 2) for v in self.start],
            "goal": [round(v, 2) for v in self.goal],
        }
        return data


def _advance_pedestrians_away(persons: list[Person], start_xy: tuple[float, float], min_dist: float = 1.8) -> None:
    """Make sure no scripted pedestrian starts on top of the robot."""
    for p in persons:
        if math.hypot(p.x - start_xy[0], p.y - start_xy[1]) < min_dist:
            p.x, p.y = p.waypoints[0]
            p.waypoint_index = 0
            # run a partial loop so the pedestrian is somewhere else
            for _ in range(1):
                p.step(1.0)


def run_navigation_trial(
    map_name: str,
    seed: int,
    n_pedestrians: int = 2,
    max_time: float = 90.0,
    grid: GridMap | None = None,
    on_frame: Callable[[dict], None] | None = None,
    collect_scans: bool = False,
    start: tuple[float, float, float] | None = None,
    goal: tuple[float, float] | None = None,
) -> NavigationTrialResult:
    """Run one 30-trial-style navigation episode and return its metrics."""
    import time as _time

    t_wall = _time.time()
    grid = grid if grid is not None else load_map(map_name)
    rng = np.random.default_rng(seed)
    if start is None:
        start = grid.sample_free_pose(rng)
    if goal is None:
        goal_xy = grid.sample_free_pose(rng)
        for _ in range(400):
            if math.hypot(goal_xy[0] - start[0], goal_xy[1] - start[1]) > 6.0:
                break
            goal_xy = grid.sample_free_pose(rng)
        goal = (goal_xy[0], goal_xy[1])
    persons = make_pedestrians(np.random.default_rng(seed + 17), grid, n=n_pedestrians)
    _advance_pedestrians_away(persons, (start[0], start[1]))
    world = SimWorld(grid, persons, lidar=LidarConfig(noise_std=0.01))
    pose = Pose2D(*start)
    stack = Nav2LikeStack(grid, NavConfig(max_time=max_time), DWAConfig())
    stack.set_goal((goal[0], goal[1]))
    dt = CONTROL_DT
    vel = (0.0, 0.0)
    status = NavStatus.TIMEOUT
    steps = 0
    collision = False
    trajectory: list[tuple[float, float]] = [(pose.x, pose.y)]
    for steps in range(int(max_time / dt) + 1):
        world.step(dt, pose)
        step = stack.step(pose, vel, dt, persons=world.dynamic_circles())
        dv = clamp(step.v - vel[0], -1.0 * dt, 1.0 * dt)
        dw = clamp(step.w - vel[1], -4.0 * dt, 4.0 * dt)
        vel = (vel[0] + dv, vel[1] + dw)
        next_pose = integrate_pose(pose, vel[0], vel[1], dt)
        if world.robot_collides(next_pose):
            collision = True
            status = NavStatus.COLLISION
            pose = next_pose
            trajectory.append((pose.x, pose.y))
            break
        pose = next_pose
        trajectory.append((pose.x, pose.y))
        if on_frame is not None:
            on_frame(
                {
                    "t": steps * dt,
                    "pose": pose,
                    "cmd": (vel[0], vel[1]),
                    "path": stack.path,
                    "planned_paths": stack.planned_paths,
                    "persons": [(p.x, p.y, p.radius, p.color_rgb) for p in world.persons],
                    "recovery": step.recovery,
                    "events": step.events,
                    "grid": grid,
                    "stack": stack,
                    "goal": (goal[0], goal[1]),
                    "scan": world.lidar_scan(pose) if collect_scans else None,
                }
            )
        if step.status != NavStatus.RUNNING:
            status = step.status
            break
    success = status == NavStatus.SUCCEEDED
    trajectory_length = sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(trajectory, trajectory[1:])
    )
    planned = stack.path_length
    return NavigationTrialResult(
        map_name=grid.name,
        seed=seed,
        success=success,
        collision=collision,
        timed_out=status == NavStatus.TIMEOUT,
        time_s=steps * dt,
        path_length=float(planned),
        trajectory_length=float(trajectory_length),
        path_ratio=float(trajectory_length / planned) if planned > 1e-6 else 0.0,
        recoveries=stack.recoveries,
        replan_count=stack.replan_count,
        start=(start[0], start[1], start[2]),
        goal=(goal[0], goal[1]),
        final_pose=(pose.x, pose.y, pose.theta),
        trajectory=trajectory,
        planned_paths=stack.planned_paths,
        events=list(stack.events),
        wall_time_s=_time.time() - t_wall,
    )


# ===========================================================================
# person-following episodes
# ===========================================================================
FOLLOW_ROUTE = [(2.6, 7.35), (18.4, 7.35), (18.4, 5.65), (2.6, 5.65)]
DISTRACTOR_ROUTE = [(2.5, 3.0), (2.5, 9.8)]
FOLLOW_START = (1.15, 7.35, 0.0)


def build_follow_scene(
    map_name: str = "warehouse_a",
    target_color: str = "red",
    target_speed: float = 0.45,
    distractor: bool = True,
    grid: GridMap | None = None,
) -> tuple[GridMap, SimWorld, Person, Pose2D]:
    grid = grid if grid is not None else load_map(map_name)
    target = Person(
        person_id=1,
        color_name=target_color,
        waypoints=list(FOLLOW_ROUTE),
        speed=target_speed,
        loop=True,
    )
    persons = [target]
    if distractor:
        persons.append(
            Person(
                person_id=2,
                color_name="blue",
                waypoints=list(DISTRACTOR_ROUTE),
                speed=0.52,
                loop=True,
            )
        )
    world = SimWorld(grid, persons, lidar=LidarConfig(noise_std=0.01))
    pose = Pose2D(*FOLLOW_START)
    return grid, world, target, pose


@dataclass
class OcclusionEventResult:
    event_id: int
    hide_start: float
    hide_duration: float
    reappear_time: float
    reacquired_time: float | None
    success: bool
    recovery_time: float | None
    identity_correct: bool
    distance_error_after: float | None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "hide_start": round(self.hide_start, 2),
            "hide_duration": round(self.hide_duration, 2),
            "reacquired_time": None if self.reacquired_time is None else round(self.reacquired_time, 2),
            "success": self.success,
            "recovery_time_s": None if self.recovery_time is None else round(self.recovery_time, 2),
            "identity_correct": self.identity_correct,
            "distance_error_after_m": None
            if self.distance_error_after is None
            else round(self.distance_error_after, 3),
        }


@dataclass
class FollowEpisodeResult:
    map_name: str
    seed: int
    duration: float
    frames: list[dict] = field(default_factory=list)
    events: list[OcclusionEventResult] = field(default_factory=list)
    distance_errors: list[float] = field(default_factory=list)
    collisions: int = 0
    states: list[str] = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def tracking_ratio(self) -> float:
        if not self.states:
            return 0.0
        return sum(1 for s in self.states if s == FollowState.TRACK.value) / len(self.states)

    def to_dict(self) -> dict:
        errors = np.array(self.distance_errors) if self.distance_errors else np.array([0.0])
        successes = sum(1 for e in self.events if e.success)
        return {
            "map": self.map_name,
            "seed": self.seed,
            "duration_s": round(self.duration, 1),
            "occlusion_events": len(self.events),
            "reacquire_success": successes,
            "reacquire_rate": round(successes / len(self.events), 3) if self.events else 0.0,
            "distance_error_mean": round(float(errors.mean()), 3),
            "distance_error_p95": round(float(np.percentile(errors, 95)), 3),
            "tracking_ratio": round(self.tracking_ratio, 3),
            "collisions": self.collisions,
        }


def _blocked_ahead(grid: GridMap, pose: Pose2D, distance: float, half_width: float = 0.16) -> bool:
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    for offset in (0.0, -half_width, half_width):
        x = pose.x + c * distance - s * offset
        y = pose.y + s * distance + c * offset
        if grid.robot_blocked(x, y):
            return True
    return False


def run_follow_episode(
    map_name: str = "warehouse_a",
    seed: int = 0,
    duration: float = 42.0,
    occlusion_script: list[tuple[float, float]] | None = None,
    target_color: str = "red",
    distractor: bool = True,
    intrinsics: CameraIntrinsics | None = None,
    mount: CameraMount | None = None,
    time_misalign: float = 0.0,
    collect_frames: bool = False,
    on_frame: Callable[[dict], None] | None = None,
    follow_config: FollowConfig | None = None,
    reid_config: ReIDConfig | None = None,
    target_speed: float = 0.45,
) -> FollowEpisodeResult:
    """Run one follow episode, optionally with scripted occlusion windows.

    ``occlusion_script`` is a list of ``(start_seconds, duration_seconds)``.
    During a window the target is removed from the rendered sensor stream (the
    pedestrian is walking behind a shelf); the Kalman filter has to extrapolate
    and the ReID gallery has to re-acquire afterwards.
    """
    import time as _time

    t_wall = _time.time()
    intrinsics = intrinsics or DEFAULT_INTRINSICS
    mount = mount or DEFAULT_MOUNT
    grid, world, target, pose = build_follow_scene(
        map_name, target_color=target_color, distractor=distractor, target_speed=target_speed
    )
    camera = SyntheticCamera(intrinsics=intrinsics, mount=mount)
    pipeline = PerceptionPipeline(
        intrinsics=intrinsics,
        tf=build_robot_tree(
            camera_mount=(mount.x, mount.y, mount.z),
            camera_pitch=mount.pitch,
            camera_yaw=mount.yaw,
            camera_roll=mount.roll,
        ),
        reid_config=reid_config,
        undistort=True,
    )
    controller = FollowController(config=follow_config, matcher=pipeline, grid=grid)
    dwa = DWAPlanner(grid, DWAConfig())
    dt = CONTROL_DT
    vel = (0.0, 0.0)
    result = FollowEpisodeResult(map_name=map_name, seed=seed, duration=duration)
    script = occlusion_script or []
    events = [
        {"start": float(s), "duration": float(d), "done": False, "reappear": None, "records": []}
        for s, d in script
    ]
    # ------------------------------------------------------------------
    # frame 0: target selection (equivalent to a human clicking the person)
    # ------------------------------------------------------------------
    rgb, depth = camera.render(world, pose)
    first = pipeline.step(rgb, depth, 0.0, 0.0, robot_pose=pose)
    if not first.dropped:
        pipeline.lock_target_by_world(target.pose, first.detections)
    previous_gt_visible = world.person_visible(target, pose, half_fov=intrinsics.hfov / 2)
    # distance error is measured in steady-state TRACK: the first 2 s after
    # acquiring the target and 2 s after every re-acquisition are excluded
    # (that transient is reported separately as the recovery metric)
    settle_until = 2.0
    prev_state = FollowState.SEARCH.value
    for step_index in range(int(round(duration / dt))):
        now = step_index * dt
        # ---- scripted occlusion --------------------------------------
        hidden_now = False
        for ev in events:
            if ev["start"] <= now < ev["start"] + ev["duration"]:
                hidden_now = True
            if not ev["done"] and now >= ev["start"] + ev["duration"]:
                ev["done"] = True
                ev["reappear"] = now
        target.hidden = hidden_now
        world.step(dt, pose)
        # ---- synthetic camera + perception ---------------------------
        rgb, depth = camera.render(world, pose)
        frame = pipeline.step(
            rgb, depth, rgb_stamp=now, depth_stamp=now + time_misalign, robot_pose=pose
        )
        gt_visible = world.person_visible(target, pose, half_fov=intrinsics.hfov / 2)
        gt_distance = math.hypot(target.x - pose.x, target.y - pose.y)
        # ---- following control ---------------------------------------
        cmd = controller.update(frame, dt, now, pose)
        v, w = cmd.v, cmd.w
        # local avoidance override: when the straight-ahead path is blocked,
        # hand over to DWA toward the predicted target position
        if v > 0.02 and _blocked_ahead(grid, pose, 0.55):
            predicted = cmd.predicted_world or cmd.target_world
            if predicted is not None:
                dwa_cmd = dwa.compute(
                    pose,
                    vel,
                    [(pose.x, pose.y), predicted],
                    persons=world.dynamic_circles(),
                )
                if dwa_cmd is not None and dwa_cmd.v > -0.05:
                    v, w = 0.85 * dwa_cmd.v, dwa_cmd.w
                else:
                    v = 0.0
            else:
                v = 0.0
        dv = clamp(v - vel[0], -1.2 * dt, 1.2 * dt)
        dw = clamp(w - vel[1], -5.0 * dt, 5.0 * dt)
        vel = (vel[0] + dv, vel[1] + dw)
        next_pose = integrate_pose(pose, vel[0], vel[1], dt)
        if grid.robot_blocked(next_pose.x, next_pose.y):
            result.collisions += 1
            vel = (0.0, 0.0)
        else:
            pose = next_pose
        # ---- observation metrics ------------------------------------
        obs = frame.target
        observed_world = obs.world_xy if obs is not None else None
        identity_correct = False
        if observed_world is not None:
            gt_world = (target.x, target.y)
            identity_correct = math.hypot(
                observed_world[0] - gt_world[0], observed_world[1] - gt_world[1]
            ) < 0.9
        record = {
            "t": now,
            "state": cmd.state.value,
            "v": vel[0],
            "w": vel[1],
            "gt_visible": gt_visible,
            "gt_distance": gt_distance,
            "obs_distance": obs.distance if obs is not None else None,
            "obs_world": observed_world,
            "gt_world": (target.x, target.y),
            "identity_correct": identity_correct,
            "occluded_script": hidden_now,
            "track_id": frame.target_track_id,
            "reacquired": "reacquired" in cmd.events,
            "hidden": hidden_now,
        }
        # distance error is measured in steady-state TRACK: the first 2 s after
        # acquiring the target and 2 s after every re-acquisition are excluded
        # (that transient is reported separately as the recovery metric)
        for ev in events:
            if ev["start"] <= now < ev["start"] + ev["duration"] + 8.0:
                ev["records"].append(record)
        if (
            cmd.state == FollowState.TRACK
            and gt_visible
            and not hidden_now
            and now >= settle_until
        ):
            error = abs(gt_distance - controller.cfg.desired_distance)
            if error < 3.0:
                result.distance_errors.append(error)
        if cmd.state == FollowState.TRACK and (
            prev_state != FollowState.TRACK.value or "reacquired" in cmd.events
        ):
            settle_until = now + 2.0
        prev_state = cmd.state.value
        result.states.append(cmd.state.value)
        if collect_frames:
            result.frames.append(record)
        if on_frame is not None:
            on_frame(
                {
                    "t": now,
                    "pose": pose,
                    "vel": vel,
                    "rgb": rgb,
                    "depth": frame.depth,
                    "cmd": cmd,
                    "frame": frame,
                    "person": (target.x, target.y),
                    "target_person": target,
                    "occluded": hidden_now,
                    "gt_visible": gt_visible,
                    "grid": grid,
                    "world": world,
                    "camera": camera,
                    "stack": controller,
                    "target_color": target.color_rgb,
                    "distractor": distractor,
                }
            )
        previous_gt_visible = gt_visible
    # ------------------------------------------------------------------
    # evaluate occlusion events
    # ------------------------------------------------------------------
    for idx, ev in enumerate(events):
        reappear = ev["reappear"] if ev["reappear"] is not None else ev["start"] + ev["duration"]
        # first track frame at/after reappearance where the observation is the target
        reacquired = None
        identity_ok = False
        error_after = None
        for rec in ev["records"]:
            if rec["t"] < reappear - 1e-9:
                continue
            if rec["state"] == FollowState.TRACK.value and rec["obs_world"] is not None:
                gt_world = rec["gt_world"]
                d = math.hypot(rec["obs_world"][0] - gt_world[0], rec["obs_world"][1] - gt_world[1])
                if d < 0.9:
                    reacquired = rec["t"]
                    identity_ok = True
                    error_after = abs(rec["gt_distance"] - controller.cfg.desired_distance)
                    break
        success = reacquired is not None and (reacquired - reappear) <= 5.0
        result.events.append(
            OcclusionEventResult(
                event_id=idx,
                hide_start=ev["start"],
                hide_duration=ev["duration"],
                reappear_time=reappear,
                reacquired_time=reacquired,
                success=bool(success),
                recovery_time=None if reacquired is None else max(0.0, reacquired - reappear),
                identity_correct=identity_ok,
                distance_error_after=error_after,
            )
        )
    result.wall_time_s = _time.time() - t_wall
    result.transitions = [
        {"t": round(t, 2), "from": a.value, "to": b.value, "reason": reason}
        for t, a, b, reason in controller.transitions
    ]
    return result
