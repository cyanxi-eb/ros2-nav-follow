# 失败案例（真实遇到 + 修复）

以下都是本仓库开发过程中**真实出现过的 bug**，每条包含：现象 → 根因 → 修复 → 验证。
这些比“一切顺利”的叙事更适合面试追问，也对应了简历里“相机外参/时间戳不同步导致跟随抖动”的难点。

---

## 案例 1：DWA 原地打转、起步后停下（局部切线用错参考点）

**现象**：A* 路径正确，但机器人起步转向后原地抖动，速度从 0.55 m/s 掉到 0.03 m/s，
约 2.6s 后触发 stall 恢复。日志（`_scratch2.py`，开发期脚本）：

```
i=  0 pose=(2.00,6.50,+0.04) v=+0.00 w=+0.40  ev=['replan']
i= 38 pose=(2.19,7.27,+2.20) v=+0.03 w=+0.02
i= 59 pose=(2.16,7.31,+2.30) v=+0.00 w=+0.00  rec=stalled
```

**根因**：DWA 的“局部切线”用 `heading_of_segment(path[closest_idx], path[closest_idx+8])` 计算，
局部路径只有 4 个顶点时退化成“机器人 → 全局终点”的方向；机器人越过 lookahead carrot 后，
目标点落到身后，heading 项与目标点项互相拉扯 → 停车。

**修复**（`navfollow/dwa.py`）：把 carrot 改为**弧长投影**：`project_to_path()` 求机器人
在路径上的投影弧长，`point_at_distance(path, s + lookahead)` 取前视点；局部切线 = 机器人→carrot 方向。
**验证**：`tests/test_planning.py::test_path_projection_and_arclength`；修复后同一路径 24m
在 435 步内走完，最大横向偏差 <0.25m。

---

## 案例 2：合成相机全黑、深度反投影距离≈0（相机光学坐标系漏旋转）

**现象**：第一版合成 RGB-D 渲染 `depth[depth>0]` 为空数组，相机图全黑；修复渲染后，
深度反投影得到的目标“距离”只有 0.02m，跟随完全失效。

**根因**：两处几何错误叠加：
1. `CameraMount` 把 `rpy_to_matrix(roll, pitch, yaw)` 直接当成 `base_link→camera_optical`，
   漏掉了 camera_link → camera_optical 的 **REP-103 旋转**（x right / y down / z forward），
   导致光线方向指向天花板上方，地面永不相交；
2. pitch 符号写反：URDF 的 pitch 正值是**低头**，代码按“抬头”使用，相机朝上看。

**修复**（`navfollow/camera.py`、`navfollow/tf_tree.py`）：拆成 `body_matrix()`（URDF 外参）
和 `optical_matrix() = body_matrix @ optical_rotation`；`build_robot_tree()` 与相机渲染共用
`_OPTICAL_ROTATION`；pitch 正方向明确为“低头 8°”，并在 `docs/calibration.md` 标定流程中复核。
**验证**：`tests/test_geometry_tf.py::test_tf_optical_rotation_is_rep103`、
`tests/test_perception.py::test_deproject_project_roundtrip`、`test_camera_render_and_person_geometry`。

---

## 案例 3：跟踪器永远没有 track（ByteTrack 两阶段关联覆盖了未匹配检测）

**现象**：感知管线 `frame.detections` 恒为空，检测器明明找到人；跟随状态机永远 SEARCH。
`IoUTracker().update([det])` 返回的 tracks 为空。

**根因**：第二阶段关联把第一阶段的未匹配结果覆盖了：
```python
matched_high, rem_tracks, rem_dets = self._associate(self.tracks, high, ...)
matched_low, rem_tracks, rem_dets = self._associate(rem_tracks, low, ...)  # rem_dets 被覆盖
```
低分检测为空时，`rem_dets` 变成 `[]`，高分但未关联的检测（新目标）永远无法创建 track。

**修复**（`navfollow/perception/tracker.py`）：分别保留 `rem_high` / `rem_low`，新 track 只从
`rem_high` 创建（ByteTrack 行为：低分检测不新建轨迹）。
**验证**：`tests/test_perception.py::test_tracker_keeps_id_across_frames`、`test_pipeline_locks_and_tracks_target`。

---

## 案例 4：路径被完全堵死但不触发恢复行为（stall 判据错误）

**现象**：Nav2 demo 里在路径正前方插入一个 1.2×1.2m 障碍（pallet），机器人减速、原地左右转，
但 `recoveries=0`，一直转到 90s 超时失败。

**根因**：stall 判据是“窗口内位移 <0.10m **且** 旋转 <0.25rad”。被堵后 DWA 只能选原地旋转的
候选速度，旋转量超过阈值 → 永远不判定 stall → 恢复行为永不触发。

**修复**（`navfollow/navstack.py`）：改为**沿路径的弧长进度**判据：`project_to_path(image)` 的
弧长 3.5s 内没有增加 0.06m 即判定 stall，与机器人是否旋转无关；恢复队列
`clear costmap → spin → backup → relaxed-inflation replan` 在 demo 3 中连续触发 3 次后绕行成功。
**验证**：demo 3（`demos/03_recovery.gif`，阻塞 → recoveries=3 → success）、
`tests/test_simulation.py::test_navigation_trial_fails_for_unreachable_goal`（不可达目标必须触发恢复）。

---

## 案例 5：遮挡后 ReID 永远无法重捕获（运动门控把“首次捕获”也卡死）

**现象**：目标在 SEARCH 状态出现时，ReID 相似度 0.9+，但状态机一直不进入 TRACK；
而一旦曾经 TRACK 过，重捕获又正常。

**根因**：`_try_reacquire()` 用 Kalman 预测位置做运动门控，而 SEARCH 状态 Kalman 尚未初始化
（`predicted = None`）→ `motion_gate()` 返回 False → 任何候选都被拒绝。另外单测里把候选检测
放进了 `FrameResult.target`，但真实管线中“未锁定目标”只出现在 `detections` 里。

**修复**（`navfollow/follow/controller.py`）：`_try_reacquire` 在无预测时以“观测有效 + 距离阈值”
放行；pipeline 保证 `frame.target` 只由锁定的 track 提供，候选走 `frame.detections`。
**验证**：`tests/test_follow.py::test_controller_enters_occluded_then_reacquires_strong_match`、
`test_controller_requires_consecutive_low_similarity_matches`。

---

## 案例 6：时间戳不对齐导致整帧被丢、跟随状态抖动

**现象**（受控实验，`run_follow_episode(time_misalign=Δ)`）：当 RGB 与深度帧时间差
Δ=0.08s（> `sync_slop` 0.05s）时，感知管线每帧都判定 `depth_rgb_timestamp_mismatch` 并丢弃，
跟随退化为纯 Kalman 外推，最终进入 OCCLUDED/REACQUIRE；Δ=0.03s 时正常跟踪。

| Δ | tracking ratio | 距误差 mean |
| --- | --- | --- |
| 0.00s | 1.00 | 0.03m |
| 0.03s | 1.00 | 0.03m |
| 0.08s | 0.00 | n/a（全程无观测） |

**根因**：真实机器人上 RGB 与深度来自不同驱动/不同曝光时刻，且 TF 的 `base_link` 时间戳与图像
时间戳不一致；如果代码“用最近的 TF”而不检查时间，会得到外参正确但位姿错帧的观测 → 跟随抖动。

**修复**：显式时间同步与丢帧（`PerceptionPipeline.step` 的 `time_tolerance`、TF 查询失败
抛 `tf_unavailable`），机器人侧在 `docs/calibration.md` 给出硬同步/软同步与静态延迟补偿步骤。
**验证**：`tests/test_perception.py::test_pipeline_drops_timestamp_mismatch`。

---

## 面试可讲的“没修好”的问题

- 遮挡 >3.2s 时依赖“旋转扫描 + ReID”找回，未做主动多假设跟踪（如多目标 Kalman 分支），找回率未测。
- 行人密度高（>3 人交叉）时 HSV 检测器会粘连，颜色直方图 ReID 区分度不足；真机上应换 YOLOv8 + OSNet。
- 仿真里 DWA 用“感知到的动态圆”避让行人，没有使用真实激光点云聚类 → 真机上需要 republish 局部 costmap。
