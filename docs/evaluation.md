# 评测方法与结果

本文件说明 `navfollow.evaluate` 的评测协议、指标定义与实测结果。
**所有数字都由本仓库代码在本机真实运行得到**，报告文件为
`reports/eval_YYYYMMDD.json` / `.md`；简历初稿中的 95% / 85% / <0.2m 是占位数字，
不是本项目的目标，README 只写实测值。

- 运行命令：`python -m navfollow.evaluate --no-demos`（全量）/ `--quick`（CI，约 50s）
- 全量耗时（本机 4 核 CPU、无 GPU、无 ROS2）：**245 s**
- 环境：Windows 10、Python 3.12.14、numpy 2.5.4、OpenCV 5.0.0

## 1. 导航 trials（30 次 = 3 地图 × 10）

每张地图用固定 seed 采样起点/终点（`GridMap.sample_free_pose`，离障碍 ≥0.45m，
起终点直线距离 >6m），随机初始化 2 名按固定路线行走的行人作为动态障碍。
每个 trial 的控制周期 0.1s、上限 90s。

判定与指标：

| 指标 | 定义 |
| --- | --- |
| success | 到达目标点 0.35m 内，且全程无碰撞 |
| collision | 机器人足迹（膨胀层）与静态障碍/行人发生接触 |
| path length ratio | 实际轨迹长度 / 首次 A* 全局路径长度（理想 1.0；绕行/恢复会变大） |
| recoveries | 恢复行为（stall/no-admissible → spin/backup/replan）触发次数 |
| replans | 全局重规划次数（含初始规划） |

结果：

| 指标 | 实测 |
| --- | --- |
| 成功率 | **96.7%（29/30）** |
| 碰撞 | **0 次** |
| 平均耗时 | 26.1 s |
| 路径长度比（成功，均值 / 中位数 / 最大） | **1.05 / 0.96 / 3.79** |
| 需要恢复行为的 trial | 2/30 |
| 恢复行为触发总次数 | 7 |
| 全局规划总次数 | 37 |

| 地图 | 成功/总数 | 成功率 | 碰撞 | 平均耗时 | 路径比（成功均值） | 恢复次数 |
| --- | --- | --- | --- | --- | --- | --- |
| warehouse_a（20×13m 货架 + 3 立柱） | 10/10 | 100% | 0 | 28.0 s | 1.25 | 3 |
| warehouse_b（18×12m 迷宫式货架 + 死胡同） | 9/10 | 90% | 0 | 26.2 s | 0.96 | 4 |
| corridor_c（16×11m 1.6m 窄走廊 + 门洞） | 10/10 | 100% | 0 | 24.1 s | 0.94 | 0 |

说明：
- 唯一失败 trial（warehouse_b seed 1005）在 4 次恢复后仍未脱困，按失败计；3.79 的路径比
  也来自 warehouse_a seed 1003（3 次恢复后绕行成功）。这两个 case 保留在 JSON 明细里，未做选择性剔除。
- 均值被少数恢复 case 拉高，因此同时报告 **中位数 0.96** 更能代表正常导航表现。

## 2. 遮挡 / 重识别（20 次事件 = 10 episode × 2）

每个 episode 30s（quick 模式更短），目标行人沿仓库中央通道循环行走；脚本在
`occlusion_script` 指定时刻把目标从传感器渲染中隐藏 **1.0–3.0s**（模拟“走到货架/立柱后面”），
后端的 Kalman 必须外推、ReID 必须在目标重现后重新绑定身份。画面/逻辑之外的
“隐藏”只作用于合成相机与几何可见性，机器人运动学与感知代码全部真实运行。

判定与指标：

| 指标 | 定义 |
| --- | --- |
| reacquire success | 目标重现后 5s 内状态回到 TRACK，且观测世界坐标与目标真值误差 <0.9m |
| identity correct | 重绑定的确实是目标（不是干扰行人），由上述位置校验确认 |
| recovery time | 目标重现 → 回到 TRACK 的时间差 |
| follow distance error | TRACK 状态下 \|真实人机距离 − 1.40m\|；**排除每次进入 TRACK 后 2.0s 的收敛窗口**（初始捕获与每次重捕获），该瞬态单独用 recovery time 衡量 |

结果：

| 指标 | 实测 |
| --- | --- |
| 遮挡事件 | 20（时长 1.0–2.7s） |
| 重识别找回 | **20/20（100%）** |
| 身份正确 | 20/20 |
| 恢复耗时（回到 TRACK） | mean **0.10 s** / p95 0.10 s |
| 跟随距离误差 | mean **0.057 m** / p95 **0.283 m** / max 0.685 m（1624 个采样） |
| TRACK 占比 | 88.6% |
| 跟随过程碰撞 | 0 |

说明：
- 找回率 100% 与遮挡时长上限 2.7s（< `occluded_timeout` 3.2s）有关：目标重现时 Kalman 预测仍然有效，
  ReID 相似度约 0.93，一次强匹配即恢复。**如果遮挡 >3.2s，会进入 REACQUIRE 扫描分支，成功率会下降**——
  这是未覆盖的场景，见“限制”。
- 跟随误差的 p95 0.283m 主要来自遮挡刚结束后的追赶段之外仍被计入的个别帧（距离 1.1–1.7m 波动）；
  mean 0.057m 才代表稳态表现。

## 3. 可复现性

```bash
python -m navfollow.evaluate --no-demos      # 全量（约 4–5 分钟）
python -m navfollow.evaluate --quick         # CI（约 50s）
python -m navfollow.demo                     # 3 段 demo GIF（约 70s）
```

trial 的 seed 固定（导航 `seed_base=1000 + k`，遮挡 `seed_base=2000 + e`），
仿真本身是确定性的（`np.random.default_rng` 固定种子、无 wall-clock 依赖），
同一机器上重复运行数字一致（除 wall time）。

## 4. 没有测什么（诚实边界）

- **ROS2 / Gazebo 未在本机运行**：`ros2_ws/` 只做语法检查与文档化启动步骤。
- **YOLOv8 未参与自动化评测**：本机无 GPU、不安装 torch；评测使用合成相机 + HSV 检测器，
  YOLOv8 适配器（`perception/detector.py`）可切换到真实机器人，但需要单独评测。
- 真实行人的外观/运动多样性、光照变化、多人交错、遮挡 >3.2s、动态障碍主动避让行人等场景未覆盖。
- DWA/A* 是 Nav2 的简化替身；Nav2 的 behavior tree、costmap 插件组合、AMCL 定位误差未在仿真中复现。
