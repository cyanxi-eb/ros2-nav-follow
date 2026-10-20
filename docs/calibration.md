# 相机-底盘标定与时间同步

跟随抖动的两个典型根因是 **外参不准** 和 **时间戳不同步**。本文件给出从零标定到验证的完整步骤，
以及与本仓库代码/参数的对应关系。标定结果写回三处：URDF `camera_joint`、`config/follow.yaml`
的 `tf_broadcaster` 参数、`docs/calibration.md` 的标定记录表。

## 0. 需要标定的量

| 参数 | 含义 | 默认值 | 写回位置 |
| --- | --- | --- | --- |
| `camera_x` | 相机光心在 base_link 前向偏移 | 0.12 m | URDF `camera_joint` / `follow.yaml` |
| `camera_y` | 横向偏移（理想 0） | 0.00 m | 同上 |
| `camera_z` | 光心高度 | 1.05 m | 同上 |
| `camera_pitch` | **低头**角（正=向下） | 0.14 rad ≈ 8° | 同上 |
| `camera_yaw` | 相对底盘朝向（理想 0） | 0.00 rad | 同上 |
| `fx, fy, cx, cy` | 相机内参 | 由相机出厂/`camera_info` 提供 | `CameraInfo` topic |
| `k1,k2,p1,p2` | 径向/切向畸变 | 由标定板标定 | `CameraInfo` |

> 注意：URDF 的 pitch 正值表示**低头**（绕 y 轴把 x 轴转向 -z）。代码里
> `CameraMount.pitch=0.14` 也是“低头 8°”，与 URDF 一致；这是开发中踩过的坑（见
> `docs/failure_cases.md` 案例 2）。

## 1. 内参标定（棋盘格）

1. 打印 9×6 棋盘格（方格 25mm），贴在硬板上；
2. 相机固定后采集 20–30 张不同角度/距离（0.5–4m）的图像；
3. 录制/回放：`ros2 bag record /camera/color/image_raw /camera/color/camera_info`；
4. 用 OpenCV 标定（仓库外的一次性脚本即可）：

```python
import cv2, numpy as np, glob
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
objp = np.zeros((6*9, 3), np.float32); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
objpoints, imgpoints = [], []
h, w = None, None
for f in glob.glob("calib/*.png"):
    img = cv2.imread(f); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = gray.shape
    ok, corners = cv2.findChessboardCorners(gray, (9, 6))
    if ok:
        objpoints.append(objp)
        imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
rms, K, D, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
print("RMS reprojection error", rms, "\nK=", K, "\nD=", D)
```

5. RMS < 0.5 px 视为可用；把 K/D 写入相机驱动的 `camera_info`（RealSense/Orbbec 的
   `camera_info` 通常已由出厂参数给出，可先用 `ros2 topic echo` 与出厂值对比验证）。

## 2. 外参标定（相机 → 底盘）

### 2.1 俯仰角 / 安装高度（地面法）

1. 机器人放在平整地面，正对一面可测距的墙；
2. 在图像中量取地面消失点对应的像素行 `v_horizon`，并量取相机光心高度 `h_cam`；
3. `pitch = atan((v_horizon - cy) / fy)`（低头为正）；
4. 也可以反过来：在正前方 `d=2.0m` 处放一块高 0.5m 的标定板，量取图像中板底/板顶像素，
   解出 `pitch` 与 `h_cam`。

### 2.2 横向偏航 / 横向偏移（前方已知点法）

1. 机器人沿直线对准正前方 3m 处的地贴十字（底盘 x 轴）；
2. 十字在图像中的 u 坐标应为 `cx`，偏差 `du` 对应
   `yaw ≈ atan(du / fx)`、`y ≈ d * du / fx`；
3. 重复 3 次取均值，写入 `camera_yaw` / `camera_y`。

### 2.3 激光-相机联合验证（推荐）

1. 把 RPLIDAR 与相机同时录 bag；
2. 在 RViz 里叠加 `/scan`（LaserScan）和相机点云（`/camera/depth/points`）：
   同一面墙的激光点线应与点云墙面重合；
3. 若不重合：调 `camera_yaw` 让墙面平行度对齐，再调 `camera_x` 让墙面距离对齐；
4. 残差要求：3m 处墙面重合误差 < 5cm、角度 < 1°。

## 3. 时间同步（比外参更容易被忽略）

| 方案 | 做法 | 适用 |
| --- | --- | --- |
| 硬件同步 | 相机触发输出接 MCU，RGB/深度同时曝光；RPLIDAR 用固定转速+时间戳 | 推荐 |
| 软件同步 | `message_filters.ApproximateTimeSynchronizer(slop=0.05)` + 统一使用 `use_sim_time`/chrony | 本仓库默认 |
| 静态延迟补偿 | 实测“图像时间戳 → TF 可用时间”的延迟，给 TF 查询加固定 offset | 临时验证 |

验证步骤（本仓库代码支持显式丢帧，便于定位）：

1. `ros2 topic delay /camera/depth/image_raw` 观察深度相对 RGB 的延迟；
2. 把 `time_misalign` 等效场景在仿真里复现：当 RGB/深度时间差 > `sync_slop` 时，
   管线会丢弃该帧并在 `/follow/status` 的 `frames_dropped` 里累计；
3. 记录表（示例）：

| 项目 | 值 | 验证人 | 日期 |
| --- | --- | --- | --- |
| RMS 重投影误差 | 0.31 px | 陈彦希 | 2026-05（仿真机） |
| 俯仰角 pitch | 0.140 rad | 陈彦希 | 2026-05 |
| 相机高度 | 1.052 m | 陈彦希 | 2026-05 |
| yaw 残差 | 0.004 rad | 陈彦希 | 2026-05 |
| RGB/深度延迟 | < 20 ms（仿真内为 0） | — | — |

> 上表来自离线仿真/合成相机验证；**真机标定尚未执行**，买齐硬件后按本流程填写。

## 4. 里程计补偿（odom → map）

- `odom → base_link` 由轮式里程计（`tf_broadcaster` 的 `publish_odom`）或 Gazebo diff-drive 插件给出；
- `map → odom` 由 AMCL/SLAM 发布；两者之间的漂移就是“定位误差”；
- 若发现机器人直行时 `map` 系航向缓慢偏转：先检查轮距/轮径（`wheel_separation=0.32`、
  `wheel_radius=0.065`），再用 `ros2 run nav2_util lifecycle` 观察 AMCL 粒子协方差；
- 跟随场景对地图系的绝对精度要求不高（目标在相机系闭环），但**相机-底盘外参**与**时间戳**
  必须准，否则表现为“目标距离周期性跳变、速度命令抖动”。

## 5. 与代码的对应

- `ros2_ws/src/follow_robot/urdf/follow_robot.urdf.xacro` 的 `camera_joint`；
- `ros2_ws/src/follow_robot/config/follow.yaml` 的 `tf_broadcaster.camera_*`；
- 离线仿真 `navfollow/camera.py` 的 `CameraMount` 与 `navfollow/tf_tree.py::build_robot_tree`；
- 标定残差可注入验证：`navfollow.simulation.run_follow_episode(...)` 通过修改 `CameraMount`
  可复现“外参有误差时的跟随抖动”（见 `docs/failure_cases.md` 案例 2/6）。
