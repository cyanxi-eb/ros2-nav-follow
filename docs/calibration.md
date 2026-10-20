# 相机-底盘标定与时间同步

跟随抖动的两个典型根因是**外参不准**和**时间戳不同步**。这里是从零标定到验证的完整步骤，
以及每一步和仓库代码/参数的对应关系。标定结果写回三处：URDF 的 `camera_joint`、
`config/follow.yaml` 的 `tf_broadcaster` 参数、本文末尾的标定记录表。

## 0. 要标定哪些量

| 参数 | 含义 | 默认值 | 写回位置 |
| --- | --- | --- | --- |
| `camera_x` | 相机光心在 base_link 前向偏移 | 0.12 m | URDF `camera_joint` / `follow.yaml` |
| `camera_y` | 横向偏移（理想 0） | 0.00 m | 同上 |
| `camera_z` | 光心高度 | 1.05 m | 同上 |
| `camera_pitch` | **低头**角（正=向下） | 0.14 rad ≈ 8° | 同上 |
| `camera_yaw` | 相对底盘朝向（理想 0） | 0.00 rad | 同上 |
| `fx, fy, cx, cy` | 相机内参 | 由相机出厂/`camera_info` 提供 | `CameraInfo` topic |
| `k1,k2,p1,p2` | 径向/切向畸变 | 由标定板标定 | `CameraInfo` |

> 坑先说：URDF 的 pitch 正值表示**低头**（绕 y 轴把 x 轴转向 -z）。代码里
> `CameraMount.pitch=0.14` 同样是"低头 8°"，两者一致。开发时这里符号搞反过，见
> `docs/failure_cases.md` 案例 2。

## 1. 内参标定（棋盘格）

1. 打印 9×6 棋盘格（方格 25mm），贴到硬板上；
2. 相机固定后，采 20–30 张不同角度/距离（0.5–4m）的图；
3. 录 bag：`ros2 bag record /camera/color/image_raw /camera/color/camera_info`；
4. 用 OpenCV 标（仓库外的一次性脚本就够）：

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

5. RMS < 0.5 px 才算可用。把 K/D 写进相机驱动的 `camera_info`。RealSense/Orbbec 一般出厂就带
   `camera_info`，先用 `ros2 topic echo` 和出厂值对一遍，差值大再自己标。

## 2. 外参标定（相机 → 底盘）

### 2.1 俯仰角 / 安装高度（地面法）

1. 机器人放平地上，正对一面能测距的墙；
2. 在图像里量地面消失点对应的像素行 `v_horizon`，再量相机光心高度 `h_cam`；
3. `pitch = atan((v_horizon - cy) / fy)`（低头为正）；
4. 也可以反着来：正前方 `d=2.0m` 放一块高 0.5m 的标定板，量图像里板底/板顶像素，联立解 `pitch` 和 `h_cam`。

### 2.2 横向偏航 / 横向偏移（前方已知点法）

1. 机器人沿直线对准正前方 3m 处的地贴十字（底盘 x 轴）；
2. 十字在图像里的 u 坐标应该是 `cx`，偏差 `du` 对应
   `yaw ≈ atan(du / fx)`、`y ≈ d * du / fx`；
3. 重复 3 次取均值，写进 `camera_yaw` / `camera_y`。

### 2.3 激光-相机联合验证（推荐）

1. RPLIDAR 和相机同时录 bag；
2. RViz 里叠 `/scan`（LaserScan）和相机点云（`/camera/depth/points`）：同一面墙的激光点线应该和点云墙面重合；
3. 不重合就先调 `camera_yaw` 对平行度，再调 `camera_x` 对距离；
4. 残差要求：3m 处墙面重合误差 <5cm、角度 <1°。

## 3. 时间同步（比外参更容易被忽略）

| 方案 | 做法 | 适用 |
| --- | --- | --- |
| 硬件同步 | 相机触发输出接 MCU，RGB/深度同时曝光；RPLIDAR 固定转速 + 时间戳 | 推荐 |
| 软件同步 | `message_filters.ApproximateTimeSynchronizer(slop=0.05)` + 统一 `use_sim_time`/chrony | 本仓库默认 |
| 静态延迟补偿 | 实测"图像时间戳 → TF 可用时间"的延迟，给 TF 查询加固定 offset | 临时验证 |

验证步骤（代码支持显式丢帧，方便定位）：

1. `ros2 topic delay /camera/depth/image_raw` 看深度相对 RGB 的延迟；
2. 在仿真里复现 `time_misalign` 等效场景：RGB/深度时间差 > `sync_slop` 时，管线丢帧并累加到
   `/follow/status` 的 `frames_dropped`；
3. 填记录表：

| 项目 | 值 | 日期 |
| --- | --- | --- |
| RMS 重投影误差 | 0.31 px | 2026-05 |
| 俯仰角 pitch | 0.140 rad | 2026-05 |
| 相机高度 | 1.052 m | 2026-05 |
| yaw 残差 | 0.004 rad | 2026-05 |
| RGB/深度延迟 | < 20 ms（仿真内为 0） | — |

> 上表来自离线仿真/合成相机验证；**真机标定还没做**，硬件到齐后按本流程重新填。

## 4. 里程计补偿（odom → map）

- `odom → base_link` 由轮式里程计（`tf_broadcaster` 的 `publish_odom`）或 Gazebo diff-drive 插件给；
- `map → odom` 由 AMCL/SLAM 发布，两者之间的漂移就是"定位误差"；
- 机器人直行时 `map` 系航向缓慢偏转，先查轮距/轮径（`wheel_separation=0.32`、`wheel_radius=0.065`），
  再用 `ros2 run nav2_util lifecycle` 看 AMCL 粒子协方差；
- 跟随时对地图系的绝对精度要求不高（目标位置在相机坐标系里解算），但**相机-底盘外参**和**时间戳**必须准，
  否则表现就是"目标距离周期性跳变、速度命令抖动"。

## 5. 和代码的对应

- `ros2_ws/src/follow_robot/urdf/follow_robot.urdf.xacro` 的 `camera_joint`；
- `ros2_ws/src/follow_robot/config/follow.yaml` 的 `tf_broadcaster.camera_*`；
- 离线仿真 `navfollow/camera.py` 的 `CameraMount` 与 `navfollow/tf_tree.py::build_robot_tree`；
- 外参误差可以注入验证：`navfollow.simulation.run_follow_episode(...)` 改 `CameraMount`
  就能复现"外参不准时的跟随抖动"（见 `docs/failure_cases.md` 案例 2/6）。
