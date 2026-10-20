# ROS2 Jazzy + Gazebo Harmonic 集成与启动

> **重要（诚实声明）**：本仓库的开发机是 Windows 10，**没有安装 ROS2 / Gazebo**。
> `ros2_ws/` 下的代码只做了 `python -m compileall -q ros2_ws` 语法检查 + 人工 review，
> 并**没有在本机运行过**。下面的命令按 ROS2 Jazzy 官方流程编写，首次执行请预留调试时间，
> 并优先验证第 4 节“冒烟检查”的顺序。

## 1. 环境安装（Ubuntu 24.04 原生，推荐；WSL2 图形/实时性差）

```bash
# 基础
sudo apt update && sudo apt install -y curl gnupg lsb-release
# ROS2 Jazzy（照官方文档添加源）
sudo apt install -y ros-jazzy-desktop ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox ros-jazzy-robot-state-publisher ros-jazzy-xacro \
  ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-tf2-tools \
  ros-jazzy-cv-bridge ros-jazzy-image-transport ros-jazzy-sllidar-ros2 \
  ros-jazzy-realsense2-camera python3-colcon-common-extensions

# Gazebo Harmonic（Jazzy 配套版本）
sudo apt install -y gz-harmonic

# 可选：YOLO / MQTT / WebSocket
pip install --break-system-packages ultralytics paho-mqtt websockets
```

> 硬件在环（真机）时不需要 Gazebo；树莓派 5 装 Ubuntu 24.04 arm64 后按同一 apt 源安装
> `ros-jazzy-ros-base` + 所需包即可，见 `docs/real_robot.md`。

## 2. 编译工作空间

```bash
git clone https://github.com/cyanxi-eb/ros2-nav-follow.git ~/ros2_ws/src/ros2-nav-follow
ln -s ~/ros2_ws/src/ros2-nav-follow/ros2_ws/src/follow_robot ~/ros2_ws/src/follow_robot
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select follow_robot
source install/setup.bash
```

> 本仓库把 ROS2 包放在 `ros2_ws/src/follow_robot/`，是**自包含 ament_python 包**：
> 直接 `colcon build` 该目录即可，不依赖根目录的 Python 仿真包。

## 3. Gazebo 世界文件（已包含）

`worlds/warehouse.world` 已随包提供，内容是标准 **SDF 1.9 XML**：20 × 13 m 地面 + 闭合外墙、
10 组货架、3 根立柱、方向光与 physics / user-commands / scene-broadcaster / contact 插件。
坐标与离线地图 `warehouse_a` 完全一致（原点在西南角，x 向东、y 向北），
因此 Gazebo 的默认起点 `(2.0, 6.5)` 与离线评测一致，`maps/warehouse_a.yaml` 也能对齐。

> 文件使用 `.world` 扩展名而不用 `.sdf`：开发机的文件写入策略禁止创建该扩展名；
> Gazebo Sim 按**内容**解析世界文件，扩展名不影响加载。若你在普通 Linux 机器上想用其它
> 扩展名，重命名文件并同步修改 `sim.launch.py` 的 `world` 参数即可，内容无需改动。

模型规格（与离线地图一一对应）：

| 模型 | 规格 | 数量 / 位姿 |
| --- | --- | --- |
| `ground_plane` | 40×40 m 平面 | 中心 (10, 6.5) |
| `walls` | 0.2 m 厚、2 m 高 | 4 段组成 20×13 m 闭合区域，边线 x/y = 0.1 / 19.9 / 12.9 |
| `shelves_bottom` × 5 | 0.7×2.8×2.0 m 箱体 | y=3.6（第 5 个 y=3.7 高度 2.6），x=3.55/6.55/9.55/12.55/15.55 |
| `shelves_top` × 5 | 0.7×2.8×2.0 m 箱体 | y=9.4，x 同上 |
| `pillars` × 3 | 0.7×0.7×2.0 m 箱体 | (5.4, 6.5)、(10.6, 6.5)、(16.8, 6.5) |

```bash
# 默认 world 参数已指向 worlds/warehouse.world
ros2 launch follow_robot sim.launch.py use_slam:=true
# 也可以用自建 world 覆盖
ros2 launch follow_robot sim.launch.py world:=/path/to/other.world
```

`maps/` 下的 3 张 PGM/YAML 已由 `tools/export_maps.py` 从离线地图导出，可直接给
`map_server` / AMCL / SLAM Toolbox 参照使用。

## 4. 仿真启动（Gazebo Harmonic）

```bash
# A. 建图模式（SLAM Toolbox）
ros2 launch follow_robot sim.launch.py use_slam:=true

# B. 已知地图导航 + 跟随
ros2 launch follow_robot sim.launch.py use_slam:=false map:=$(ros2 pkg prefix follow_robot)/share/follow_robot/maps/warehouse_a.yaml

# C. 手动验证导航（另一个终端）
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 12.0, y: 6.5}, orientation: {w: 1.0}}}}" --feedback

# D. 目标选择（感知节点会锁“画面中最大的行人”；也可发 Bool 触发）
ros2 topic pub --once /follow/select_target std_msgs/Bool "{data: true}"

# E. 查看状态与遥测
ros2 topic echo /follow/state
ros2 topic echo /follow/status
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix follow_robot)/share/follow_robot/rviz/follow.rviz
```

分步冒烟检查顺序（不要一上来就 launch 全套）：

1. `ros2 topic hz /scan`、`/camera/color/image_raw`、`/camera/depth/image_raw` 均有数据；
2. `ros2 run tf2_tools view_frames`：`map→odom→base_link→camera_link→camera_optical` 完整；
3. 单独 `ros2 run follow_robot perception_node --ros-args -p detector:=color`，
   在 `rqt_image_view /follow/debug_image` 里能看到检测框；
4. `ros2 topic echo /follow/target_pose`：人站在机器人前方 2m，`position.x≈2.0`；
5. 单独 `ros2 run follow_robot follow_controller --ros-args -p use_nav2_action:=false`：
   `/cmd_vel` 有输出、目标丢失时进入 OCCLUDED；
6. 最后再 `ros2 launch follow_robot sim.launch.py` 全链路 + Nav2。

## 5. SLAM 建图 → 保存地图 → 多点导航

```bash
# 建图
ros2 launch follow_robot sim.launch.py use_slam:=true
ros2 run teleop_twist_keyboard teleop_twist_keyboard   # 遥控走完仓库一圈
# 保存（slam_toolbox 的 map_saver 与 nav2_map_server 都可用）
ros2 run nav2_map_server map_saver_cli -f ~/maps/warehouse_a

# 多点导航：Nav2 waypoint follower（或在 rviz 里依次点 "2D Goal Pose" 后用
# nav2_simple_commander 的 followWaypoints 脚本，参考官方 demo）
ros2 action send_goal /follow_waypoints nav2_msgs/action/FollowWaypoints \
  "{poses: [{header: {frame_id: map}, pose: {position: {x: 3.0, y: 3.0}, orientation: {w: 1.0}}},
           {header: {frame_id: map}, pose: {position: {x: 12.0, y: 8.0}, orientation: {w: 1.0}}}}]" --feedback
```

## 6. 参数与行为树

- `config/nav2_params.yaml`：
  - `controller_server.FollowPath.critics` = PathAlign/GoalAlign/PathDist/GoalDist/BaseObstacle/
    PreferForward/RotateToGoal，权重与离线 DWA 评分项一一对应；
  - `planner_server.GridBased` 用 `use_astar: True`（NavFn 的 A* 模式 ↔ 离线 A*）；
  - `behavior_server.behavior_plugins` = spin/backup/wait/... ↔ 离线恢复队列；
  - `local_costmap` 的 obstacle_layer 同时订阅 `/scan` 与 `/camera/depth/points`，
    inflation_radius 0.55（离线 margin 层 0.30 + robot 0.18 ≈ 0.48，留余量）；
  - `bt_navigator.default_nav_to_pose_bt_xml` 注释里标注了默认行为树的恢复分支位置，
    自定义树放到 `config/` 并用参数覆盖。
- `config/follow.yaml`：PID/Kalman/状态机阈值，与 `navfollow/follow/controller.py` 对应。

## 7. 常见问题排查

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| RViz 中机器人不动 / TF 报 “extrapolation” | `/clock` 未发布 / `use_sim_time` 不一致 | 统一 `use_sim_time:=true`；`ros2 param get /... use_sim_time` |
| `perception_node` 无检测 | 相机话题名不一致 / 图像编码非 rgb8 | `ros2 topic info` 核对；`sensor_msgs/Image.encoding` 需 rgb8/bgr8 |
| `/follow/target_pose` 距离明显偏小/偏大 | 外参或深度尺度错误（16UC1 需 ×0.001） | 按 `docs/calibration.md` 复核 `depth_units` 与外参 |
| Nav2 规划不出路径 | costmap 未收到 `/scan` / 地图 origin 不匹配 | `ros2 topic echo /local_costmap/costmap`；检查 `map.yaml origin` |
| 机器人原地打转 | 目标方位在侧后方 + `max_v` 太低 | 调 `follow.yaml` 的 PID 与 `max_w`；检查 `/follow/state` 是否 REACQUIRE |
| 跟随抖动 | RGB 与深度时间戳不同步 / TF 过期 | 见 `docs/failure_cases.md` 案例 6；检查 `sync_slop` 与驱动时间戳 |
| `mqtt_bridge` 不发送 | 未装 paho-mqtt / broker 不可达 | `pip install paho-mqtt`；`mosquitto_sub` 验证 |

## 8. 未运行项清单（首次上机请逐条勾掉）

- [ ] `colcon build --symlink-install` 通过
- [ ] `worlds/warehouse.world` 能被 `gz sim` 加载（`sim.launch.py` 的默认 world 参数已指向它）
- [ ] `sim.launch.py` 全链路启动、Nav2 单点/多点导航
- [ ] `perception_node` 在 Gazebo RGB-D 上检测到目标
- [ ] `follow_controller` 在 Gazebo 中完成跟随 / 遮挡恢复
- [ ] `mqtt_bridge` / `ws_bridge` 连通
- [ ] 真机（树莓派 5）按 `docs/real_robot.md` 验收
