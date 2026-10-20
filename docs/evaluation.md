# 评测方法与结果

这份文档说明 `navfollow.evaluate` 怎么测、指标怎么定义、实测结果是多少。报告文件在
`reports/eval_YYYYMMDD.json` / `.md`，数字全部由本仓库代码实际跑出来，没有手工填的。

- 命令：`python -m navfollow.evaluate --no-demos`（全量）/ `--quick`（CI 冒烟，约 50s）
- 全量耗时：**245 s**（纯 Python，无 GPU、无 ROS2）
- 评测环境：离线工作站（无 ROS2 / Gazebo / GPU）

## 1. 导航 trials（30 次 = 3 地图 × 10）

每张地图用固定 seed 采样起点/终点（`GridMap.sample_free_pose`，离障碍 ≥0.45m，起终点直线距离 >6m），
再随机初始化 2 名按固定路线行走的行人当动态障碍。每个 trial 控制周期 0.1s，上限 90s。

判定和指标：

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

两点说明：

- 唯一失败的 trial（warehouse_b seed 1005）在 4 次恢复后仍未脱困，按失败计；3.79 的路径比来自
  warehouse_a seed 1003（3 次恢复后绕行成功）。这两个 case 都留在 JSON 明细里，没有挑掉。
- 均值被少数恢复 case 拉高，所以同时报 **中位数 0.96**——正常导航下这个数更能代表水平。

## 2. 遮挡 / 重识别（20 次事件 = 10 episode × 2）

每个 episode 30s（quick 模式更短），目标行人沿仓库中央通道循环走；脚本在
`occlusion_script` 指定的时刻把目标从传感器渲染里"藏起来" **1.0–3.0s**（模拟走到货架/立柱后面）。
这期间 Kalman 必须靠外推撑住，目标重现后 ReID 必须重新把身份绑回来。

"隐藏"只作用于合成相机和几何可见性，机器人运动学与感知代码全部照常运行——不是把逻辑结果直接写死。

判定和指标：

| 指标 | 定义 |
| --- | --- |
| reacquire success | 目标重现后 5s 内状态回到 TRACK，且观测世界坐标与目标真值误差 <0.9m |
| identity correct | 重绑定的确实是目标（不是干扰行人），由上述位置校验确认 |
| recovery time | 目标重现 → 回到 TRACK 的时间差 |
| follow distance error | TRACK 状态下 \|真实人机距离 − 1.40m\|；**排除每次进入 TRACK 后 2.0s 的收敛窗口**（初始捕获与每次重捕获），这段瞬态单独用 recovery time 衡量 |

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

同样两点说明：

- 找回率 100% 和遮挡时长上限 2.7s（< `occluded_timeout` 3.2s）直接相关：目标重现时 Kalman 预测还有效，
  ReID 相似度约 0.93，一次强匹配就回来了。**遮挡超过 3.2s 会进 REACQUIRE 扫描分支，成功率会掉**——
  这个场景没覆盖，见"限制"。
- 跟随误差 p95 0.283m 来自遮挡刚结束、还在追赶的那几帧（距离 1.1–1.7m 波动）；均值和稳态表现看
  mean 0.057m。

## 3. 可复现性

```bash
python -m navfollow.evaluate --no-demos      # 全量（约 4–5 分钟）
python -m navfollow.evaluate --quick         # CI（约 50s）
python -m navfollow.demo                     # 3 段 demo GIF（约 70s）
```

trial 的 seed 固定（导航 `seed_base=1000 + k`，遮挡 `seed_base=2000 + e`），仿真本身是确定性的
（`np.random.default_rng` 固定种子、不看 wall-clock）。同一台机器上重复跑，数字一致——除了 wall time。

## 4. 没测什么

- **ROS2 / Gazebo 没跑过**：`ros2_ws/` 只做了语法检查和文档化启动步骤。
- **YOLOv8 不参与这组数字**：没有 GPU、不装 torch；评测用合成相机 + HSV 检测器。YOLOv8 适配器在
  `perception/detector.py` 里，切到真机时用，但需要单独评测。
- 没覆盖：真实行人的外观/运动多样性、光照变化、多人交错、遮挡 >3.2s、主动避让行人。
- DWA/A* 是 Nav2 的简化替身，Nav2 的 behavior tree、costmap 插件组合、AMCL 定位误差都没有在仿真里复现。
