# 真机部署：树莓派 5 + RPLIDAR A1 + 深度相机

> 状态：**本仓库没有在该硬件上跑过**。以下是按设计整理的 BOM、接线、装机步骤和验收清单，
> 留给后面实机落地用。标定流程见 `docs/calibration.md`，ROS2 环境见 `docs/ros2_integration.md`。

## 1. BOM（预算 ~3000 元）

| # | 物料 | 推荐型号 | 数量 | 参考价 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 1 | 主控 | Raspberry Pi 5 8GB | 1 | ¥700 | 跑 ROS2 Jazzy、YOLOv8n(CPU)、感知/跟随节点 |
| 2 | 存储 | microSD 64GB A2 / NVMe SSD + HAT（可选） | 1 | ¥60–260 | 系统 + rosbag |
| 3 | 2D 激光 | RPLIDAR A1M8（12m, 10Hz） | 1 | ¥550 | /scan，Nav2 costmap + AMCL |
| 4 | 深度相机 | 奥比中光 Gemini 2 / Intel RealSense D435i | 1 | ¥900–1600 | RGB-D，YOLO + 深度反投影 |
| 5 | 底盘 | 2WD 差速底盘（带编码器，轮距 0.32m，轮径 0.065m） | 1 | ¥350 | 差速运动学 |
| 6 | 电机驱动 | TB6612 / DRV8833 ×2 或 ESP32 micro-ROS 驱动板 | 1 | ¥40–120 | 电机 PWM + 编码器 |
| 7 | 电池 | 3S 18650 锂电组 + 5V/5A UBEC | 1 | ¥180 | 供电（Pi5 峰值 5A） |
| 8 | IMU | ICM42688 / BNO085 | 1 | ¥40 | EKF 融合（可选） |
| 9 | 急停 | 自锁按钮 + 继电器 | 1 | ¥30 | 安全回路 |
| 10 | 结构件 | 亚克力/铝型材支架、螺丝、扎带 | — | ¥100 | 相机/雷达/电池固定 |

合计 **¥2500–3000**，按 2026 年的散件价估，实际会随行情浮动。

没有 GPU 时，YOLOv8n 在 Pi5 CPU 上大约 6–10 FPS（320×320 输入）。跟随控制建议从 10Hz 降到 5–8Hz；
要维持高频就换 Jetson Orin Nano，预算大概 +¥2000。

## 2. 接线（关键信号）

| 信号 | 树莓派 5 | 说明 |
| --- | --- | --- |
| RPLIDAR A1 | USB（CP2102） | `/dev/ttyUSB0`，`sudo usermod -aG dialout pi` |
| 深度相机 | USB 3.0（蓝色口） | RealSense 用 `realsense2_camera`；Gemini2 用 `orbbec_camera` |
| 电机 PWM L/R | GPIO12 / GPIO13（或经 ESP32） | 20kHz PWM，方向脚 GPIO5/6 |
| 编码器 A/B L/R | GPIO17/27, GPIO22/23 | 正交编码，`pigpio` 或 ESP32 计数 |
| 急停 | GPIO4 上拉 | 低电平 → 软件网关立即发零速 + 断电机使能 |
| 5V 供电 | UBEC 5V/5A → Pi5 USB-C | 与电机电源共地，避免欠压 |

> 建议把电机/编码器挂到 **ESP32 + micro-ROS**（`micro_ros_arduino`），Pi5 只跑高层节点。
> 这样 GPIO 实时性压力小，Python 抖动也不会直接影响 PWM。

## 3. 软件栈

```
Ubuntu 24.04 arm64 (Raspberry Pi 5)
  └─ ROS2 Jazzy (apt: ros-jazzy-desktop 的最小集合)
      ├─ slam_toolbox / nav2 / robot_state_publisher
      ├─ realsense2_camera 或 orbbec_camera（RGB-D）
      ├─ sllidar_ros2（RPLIDAR 驱动，/scan）
      ├─ micro_ros_agent（串口接 ESP32）
      └─ follow_robot（本仓库：perception_node / follow_controller / tf_broadcaster /
                       mqtt_bridge / ws_bridge）
```

ARM 上装 YOLO 依赖：

```bash
python3 -m venv ~/venv_yolo && source ~/venv_yolo/bin/activate
pip install ultralytics paho-mqtt websockets
# 注意 ROS2 节点需要系统 python 的 rclpy：用 --system-site-packages 建 venv，或者直接在系统
# python 里 pip install（树莓派官方系统允许；Ubuntu 24.04 要加 --break-system-packages）
python3 -m venv --system-site-packages ~/venv_ros
```

## 4. 开机流程

```bash
# 1) 编译工作空间（包含本仓库的 follow_robot）
cd ~/ros2_ws && colcon build --symlink-install
source /opt/ros/jazzy/setup.bash && source install/setup.bash

# 2) 传感器
ros2 launch sllidar_ros2 sllidar_launch.py          # /scan
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true

# 3) 底盘 + TF（micro-ROS 节点发布 /odom 与 odom->base_link；tf_broadcaster 发相机/雷达外参）
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB1
ros2 run follow_robot tf_broadcaster --ros-args --params-file config/follow.yaml

# 4) 定位 + 导航 + 跟随（把 <broker-ip> 换成你自己的 MQTT broker 地址）
ros2 launch follow_robot follow.launch.py map:=$HOME/maps/warehouse_a.yaml \
  use_mqtt:=true use_ws:=true mqtt_host:=<broker-ip>

# 5) 遥测（示例）
mosquitto_sub -h <broker-ip> -t 'follow_robot/state' -v
# WebSocket 客户端连 ws://<pi-ip>:8765

# 6) 复盘
ros2 bag record -o ~/bags/follow_$(date +%s) /scan /camera/color/image_raw \
  /camera/depth/image_raw /odom /cmd_vel /follow/status /tf /tf_static
```

## 5. 树莓派上的系统服务（示例）

`/etc/systemd/system/follow-robot.service`：

```ini
[Unit]
Description=follow_robot stack
After=network-online.target
[Service]
Type=simple
User=pi
Environment=ROS_DOMAIN_ID=42
ExecStart=/bin/bash -lc 'source /opt/ros/jazzy/setup.bash && source /home/pi/ros2_ws/install/setup.bash && \
  ros2 launch follow_robot follow.launch.py map:=/home/pi/maps/warehouse_a.yaml use_mqtt:=true'
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now follow-robot
journalctl -u follow-robot -f
```

## 6. 安全边界（进 ROS2 之前的统一网关）

1. **速度上限**：`follow_controller` 参数 `max_v=0.45`、`max_w=1.0`（实机建议比仿真值再低一点）；
2. **急停**：GPIO4 低电平 → 网关节点立刻发零速并断电机使能；
3. **超时停车**：`/follow/target_visible=false` 超过 `occluded_timeout (3.2s)` 进 REACQUIRE，
   超过 `reacquire_timeout (6s)` 进 STOP（零速）；
4. **电子围栏**：Nav2 costmap 之外再加一层 `map` 边界检查，禁止进入人工标注的危险区；
5. **看门狗**：`/cmd_vel` 超过 0.3s 没更新 → 底盘 MCU 自己刹车。

## 7. 验收清单（真机第一次上电）

- [ ] `/tf` 里 `map→odom→base_link→camera_link→camera_optical` 完整，`ros2 run tf2_tools view_frames`
- [ ] `/scan` 与相机点云在 RViz 里墙面重合（标定残差 <5cm / 1°）
- [ ] `ros2 topic hz /camera/color/image_raw` 稳定 ≥10Hz，`/scan` 10Hz
- [ ] 单独跑 `perception_node`：`/follow/debug_image` 能看到 YOLO 框，`/follow/target_pose` 数值合理
- [ ] 单独跑 `follow_controller`（先 `use_nav2_action:=false`）：目标静止 2m 时，`/cmd_vel` 收敛到 0
- [ ] 遥控走一圈：无碰撞、无 TF 报错，rosbag 完整
- [ ] 遮挡测试：目标走到货架后 1–3s，看 `/follow/state` 是否 OCCLUDED → TRACK
- [ ] 断网/重启：systemd 能自恢复，急停有效
- [ ] 记一份真机评测报告（模板同 `reports/eval_*.md`，指标口径一致）
