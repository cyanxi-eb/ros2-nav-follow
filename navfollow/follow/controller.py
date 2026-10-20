"""Person-following state machine: SEARCH / TRACK / OCCLUDED / REACQUIRE / ARRIVED / STOP.

State diagram::

    SEARCH --target seen--> TRACK --lost > lost_timeout--> OCCLUDED
      ^                       |  \\                          |
      |                       |   \\--target static 3 s--> ARRIVED
      |                       |                             | kalman predict
      |                       |                             v
      |                       +<--ReID strong match---- REACQUIRE <--occluded_timeout--
      |                                                     |
      +---search timeout / reacquire timeout----------------+--> STOP

The controller is deliberately independent from ROS2: it consumes
:class:`~navfollow.perception.pipeline.FrameResult` objects and returns
``(v, w)``.  ``follow_controller.py`` in the ROS2 package wraps exactly this
class and publishes ``/cmd_vel`` + a status string.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from ..geometry import Pose2D, angle_diff, clamp
from ..perception.pipeline import FrameResult, TrackedDetection
from .kalman import KalmanCV2D
from .pid import FollowPID, FollowPIDConfig


class FollowState(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    OCCLUDED = "OCCLUDED"
    REACQUIRE = "REACQUIRE"
    ARRIVED = "ARRIVED"
    STOP = "STOP"


@dataclass
class FollowConfig:
    desired_distance: float = 1.40
    body_radius_offset: float = 0.22  # depth camera sees the front surface, not the centroid
    feedforward: bool = True
    lost_timeout: float = 0.35
    occluded_timeout: float = 3.2
    reacquire_timeout: float = 6.0
    search_timeout: float = 12.0
    arrival_hold: float = 3.0
    arrival_speed: float = 0.14
    arrival_distance_window: tuple[float, float] = (0.8, 2.6)
    prediction_horizon: float = 0.18
    occluded_speed_scale: float = 0.65
    search_spin_speed: float = 0.55
    safety_lookahead: float = 0.32
    max_target_range: float = 6.0
    reid_high: float = 0.78
    reid_low: float = 0.52
    reid_consecutive: int = 3
    motion_gate_base: float = 1.6
    motion_gate_growth: float = 0.6


@dataclass
class FollowCommand:
    v: float
    w: float
    state: FollowState
    events: list[str] = field(default_factory=list)
    target_world: tuple[float, float] | None = None
    predicted_world: tuple[float, float] | None = None
    distance: float | None = None
    bearing: float | None = None
    reid_similarity: float | None = None


class FollowController:
    """PID + Kalman + ReID-dual-threshold state machine."""

    def __init__(
        self,
        config: FollowConfig | None = None,
        pid_config: FollowPIDConfig | None = None,
        matcher=None,
        grid=None,
    ) -> None:
        self.cfg = config or FollowConfig()
        pid_cfg = pid_config or FollowPIDConfig()
        pid_cfg.desired_distance = self.cfg.desired_distance
        self.pid = FollowPID(pid_cfg)
        self.kalman = KalmanCV2D()
        self.matcher = matcher  # PerceptionPipeline-compatible
        self.grid = grid
        self.state = FollowState.SEARCH
        self.state_since = 0.0
        self.last_seen: float | None = None
        self.occluded_since: float | None = None
        self.reacquire_since: float | None = None
        self.search_since = 0.0
        self.transitions: list[tuple[float, FollowState, FollowState, str]] = []
        self.events: list[dict] = []
        self.reacquire_attempts = 0
        self.reacquire_success = 0
        self._weak_matches: dict[int, int] = {}
        self._stuck_time = 0.0
        self._arrival_timer = 0.0
        self._last_cmd = (0.0, 0.0)
        self._kf_stamp: float | None = None
        self._v_ff = 0.0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.pid.reset()
        self.kalman = KalmanCV2D()
        self.state = FollowState.SEARCH
        self.state_since = 0.0
        self.last_seen = None
        self.occluded_since = None
        self.reacquire_since = None
        self.search_since = 0.0
        self.transitions = []
        self.events = []
        self.reacquire_attempts = 0
        self.reacquire_success = 0
        self._weak_matches = {}
        self._arrival_timer = 0.0
        self._last_cmd = (0.0, 0.0)
        self._kf_stamp = None
        self._v_ff = 0.0

    # ------------------------------------------------------------------
    def _transition(self, new_state: FollowState, now: float, reason: str) -> None:
        if new_state == self.state:
            return
        self.transitions.append((now, self.state, new_state, reason))
        self.events.append({"type": "state", "t": round(now, 3), "from": self.state.value, "to": new_state.value,
                            "reason": reason})
        self.state = new_state
        self.state_since = now

    def _to_base(self, world_xy: tuple[float, float], robot_pose: Pose2D) -> tuple[float, float]:
        dx = world_xy[0] - robot_pose.x
        dy = world_xy[1] - robot_pose.y
        c, s = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
        forward = c * dx + s * dy
        left = -s * dx + c * dy
        return forward, left

    def _command_to_base_target(
        self, world_xy: tuple[float, float], robot_pose: Pose2D, dt: float, speed_scale: float = 1.0
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        forward, left = self._to_base(world_xy, robot_pose)
        distance = math.hypot(forward, left)
        bearing = math.atan2(left, forward)
        # the Kalman state tracks the measured (front-surface) position; add an
        # approximate body radius so the *centre* is held at desired_distance
        control_distance = distance + self.cfg.body_radius_offset
        v, w = self.pid.compute(control_distance, bearing, dt)
        if self.cfg.feedforward and self.kalman.initialized:
            # feed the target's Kalman velocity forward: removes the steady-state
            # lag a pure P controller has when the person walks at constant speed
            vx, vy = self.kalman.velocity
            v_ff_raw = math.cos(robot_pose.theta) * vx + math.sin(robot_pose.theta) * vy
            if self.kalman.speed > 0.04:
                self._v_ff = 0.75 * self._v_ff + 0.25 * v_ff_raw
                v = clamp(v + self._v_ff, self.pid.cfg.min_v, self.pid.cfg.max_v)
        v *= speed_scale
        return (v, w), (control_distance, bearing)

    def _safety_stop(self, robot_pose: Pose2D, v: float, w: float) -> tuple[float, float]:
        """Stop the linear velocity if the footprint would hit an obstacle ahead."""
        if self.grid is None or abs(v) < 1e-6:
            return v, w
        direction = 1.0 if v >= 0 else -1.0
        reach = self.cfg.safety_lookahead * max(0.5, abs(v) / max(self.pid.cfg.max_v, 1e-6))
        probe = Pose2D(
            robot_pose.x + math.cos(robot_pose.theta) * direction * reach,
            robot_pose.y + math.sin(robot_pose.theta) * direction * reach,
            robot_pose.theta,
        )
        if self.grid.robot_blocked(probe.x, probe.y):
            return 0.0, w
        return v, w

    # ------------------------------------------------------------------
    def _try_reacquire(self, frame: FrameResult | None, now: float) -> tuple[TrackedDetection | None, str | None]:
        """Dual-threshold ReID re-association with a motion gate.

        Strong match (>= ``reid_high``) -> immediate re-acquisition.
        Weak match (>= ``reid_low``)   -> needs ``reid_consecutive`` frames.
        """
        if frame is None or self.matcher is None or not frame.detections:
            return None, None
        cfg = self.cfg
        predicted = self.kalman.predict_position(0.0) if self.kalman.initialized else None
        seconds_since = (now - self.last_seen) if self.last_seen is not None else 0.0
        candidates = sorted(frame.detections, key=lambda d: d.score, reverse=True)
        for det in candidates:
            if not det.valid_geometry or det.distance is None or det.distance > cfg.max_target_range:
                continue
            similarity = self.matcher.similarity_to_target(det.feature)
            gate_ok = self.matcher.motion_gate(predicted, det, seconds_since)
            if similarity >= cfg.reid_high and gate_ok:
                return det, f"reid_high:{similarity:.2f}"
            if similarity >= cfg.reid_low and gate_ok:
                self._weak_matches[det.track_id] = self._weak_matches.get(det.track_id, 0) + 1
                if self._weak_matches[det.track_id] >= cfg.reid_consecutive:
                    return det, f"reid_low_consecutive:{similarity:.2f}"
            else:
                self._weak_matches.pop(det.track_id, None)
        return None, None

    def _accept_reacquired(self, det: TrackedDetection, now: float, reason: str) -> None:
        if self.matcher is not None:
            self.matcher.lock_track(det.track_id, det.feature)
        if det.world_xy is not None:
            if not self.kalman.initialized:
                self.kalman.reset(*det.world_xy)
            else:
                dkf = (now - self._kf_stamp) if self._kf_stamp is not None else 0.1
                self.kalman.predict(max(1e-3, min(4.0, dkf)))
                self.kalman.update(*det.world_xy)
            self._kf_stamp = now
        self.reacquire_success += 1
        self.last_seen = now
        self._weak_matches.clear()
        self.events.append({"type": "reacquired", "t": round(now, 3), "detail": reason,
                            "track_id": det.track_id})
        self._transition(FollowState.TRACK, now, f"reacquired:{reason}")

    # ------------------------------------------------------------------
    def update(
        self,
        frame: FrameResult | None,
        dt: float,
        now: float,
        robot_pose: Pose2D,
    ) -> FollowCommand:
        cfg = self.cfg
        events: list[str] = []
        target = frame.target if frame is not None else None
        visible = target is not None and target.valid_geometry and target.distance is not None

        # -------------------- target visible ---------------------------
        if visible:
            world = target.world_xy
            if not self.kalman.initialized:
                self.kalman.reset(world[0], world[1])
                self.events.append({"type": "kalman_init", "t": round(now, 3)})
            else:
                dkf = (now - self._kf_stamp) if self._kf_stamp is not None else dt
                self.kalman.predict(max(1e-3, min(2.0, dkf)))
                self.kalman.update(world[0], world[1])
            self._kf_stamp = now
            previous = self.state
            self.last_seen = now
            if previous in (FollowState.SEARCH, FollowState.OCCLUDED, FollowState.REACQUIRE):
                if previous in (FollowState.OCCLUDED, FollowState.REACQUIRE):
                    self.reacquire_success += 1
                    self.events.append(
                        {"type": "reacquired", "t": round(now, 3), "detail": "target_track_visible"}
                    )
                events.append("target_acquired")
                self._transition(FollowState.TRACK, now, "target_visible")
            elif previous == FollowState.STOP:
                self._transition(FollowState.TRACK, now, "target_visible_again")

            # latency compensation: aim at the Kalman prediction, not the raw
            # observation (this is what removes the 1-2 frame RGB-D lag)
            predicted = self.kalman.predict_position(cfg.prediction_horizon)
            robot = robot_pose
            (v, w), (distance, bearing) = self._command_to_base_target(predicted, robot, dt)
            # if the filter velocity is tiny, use the direct depth measurement
            if self.kalman.speed < 0.08:
                control_distance = target.distance + cfg.body_radius_offset
                v, w = self.pid.compute(control_distance, target.bearing, dt)
                distance, bearing = control_distance, target.bearing
            v, w = self._safety_stop(robot_pose, v, w)
            # arrival: target stopped for arrival_hold seconds nearby
            in_window = cfg.arrival_distance_window[0] <= distance <= cfg.arrival_distance_window[1]
            if self.kalman.speed < cfg.arrival_speed and in_window:
                self._arrival_timer += dt
            else:
                self._arrival_timer = 0.0
            if self._arrival_timer >= cfg.arrival_hold:
                self._transition(FollowState.ARRIVED, now, "target_stopped")
            if self.state == FollowState.ARRIVED:
                v, w = 0.0, 0.0
            cmd = FollowCommand(
                v=v,
                w=w,
                state=self.state,
                events=events,
                target_world=world,
                predicted_world=predicted,
                distance=distance,
                bearing=bearing,
                reid_similarity=None,
            )
            self._last_cmd = (v, w)
            return cmd

        # -------------------- target not visible ----------------------
        if self.state == FollowState.SEARCH:
            if self.search_since == 0.0:
                self.search_since = now
            det, reason = self._try_reacquire(frame, now)
            if det is not None:
                self._accept_reacquired(det, now, reason or "reid")
                events.append("reacquired")
            if now - self.search_since > cfg.search_timeout and self.state == FollowState.SEARCH:
                self._transition(FollowState.STOP, now, "search_timeout")
                return FollowCommand(0.0, 0.0, self.state, events)
            v, w = 0.0, cfg.search_spin_speed
            self._last_cmd = (v, w)
            return FollowCommand(v, w, self.state, events)

        if self.state in (FollowState.TRACK, FollowState.ARRIVED):
            since = now - self.last_seen if self.last_seen is not None else 1e9
            if since > cfg.lost_timeout:
                self.occluded_since = now
                self._transition(FollowState.OCCLUDED, now, "target_lost")
                events.append("occluded")
                self.events.append({"type": "occluded", "t": round(now, 3), "since_last_seen": round(since, 2)})

        if self.state == FollowState.OCCLUDED:
            det, reason = self._try_reacquire(frame, now)
            if det is not None:
                self._accept_reacquired(det, now, reason or "reid")
                events.append("reacquired")
                return FollowCommand(0.0, 0.0, self.state, events)
            since = now - self.last_seen if self.last_seen is not None else 1e9
            if since > cfg.occluded_timeout:
                self._transition(FollowState.REACQUIRE, now, "occlusion_timeout")
                self.reacquire_since = now
                self.reacquire_attempts += 1
                events.append("reacquire")
                return FollowCommand(0.0, cfg.search_spin_speed, self.state, events)
            predicted = self.kalman.predict_position(0.0)
            (v, w), (distance, _bearing) = self._command_to_base_target(
                predicted, robot_pose, dt, speed_scale=cfg.occluded_speed_scale
            )
            v, w = self._safety_stop(robot_pose, v, w)
            self._last_cmd = (v, w)
            return FollowCommand(
                v,
                w,
                self.state,
                events,
                target_world=None,
                predicted_world=predicted,
                distance=distance,
            )

        if self.state == FollowState.REACQUIRE:
            det, reason = self._try_reacquire(frame, now)
            if det is not None:
                self._accept_reacquired(det, now, reason or "reid")
                events.append("reacquired")
                return FollowCommand(0.0, 0.0, self.state, events)
            elapsed = now - self.reacquire_since if self.reacquire_since is not None else 1e9
            if elapsed > cfg.reacquire_timeout:
                self._transition(FollowState.STOP, now, "reacquire_timeout")
                return FollowCommand(0.0, 0.0, self.state, events)
            # rotate to scan for the target
            direction = 1.0 if int(elapsed / 3.0) % 2 == 0 else -1.0
            v, w = 0.0, direction * cfg.search_spin_speed
            self._last_cmd = (v, w)
            return FollowCommand(v, w, self.state, events)

        # STOP: terminal for this episode
        self._last_cmd = (0.0, 0.0)
        return FollowCommand(0.0, 0.0, self.state, events)
