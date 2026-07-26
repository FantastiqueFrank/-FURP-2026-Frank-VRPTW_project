# 第八周进度报告：POMO_Test_2 — EVRP-TW 完整训练管线

**姓名**：赵云龙
**日期**：2026年7月26日
**本周重点**：完成 `POMO_Test_2.py` 自包含训练脚本，尚未开始训练

---

## 1. 项目背景回顾

本项目是 SEP 暑期科研项目，研究带容量约束的车辆路径问题（CVRP）、带时间窗的车辆路径问题（VRPTW）、电动车辆路径问题（EVRP-TW）以及卡车-无人机协同路径优化。前期已完成的里程碑：

| 周次 | 主要内容 |
|------|----------|
| Week 1 | 环境搭建，PyVRP 基线求解 A-n32-k5（最优解 784） |
| Week 2 | VRP 文献阅读：OR-Tools / Attention Model / POMO 方法论 |
| Week 3 | OR-Tools 在 EVRP-TW（ESOGU 数据集）上的求解评估 |
| Week 4 | PyVRP vs OR-Tools 对比实验 + POMO 基线环境搭建 |
| Week 6 | 统一评估框架设计 + EVRP-TW 的 MDP 形式化描述 |

前期 POMO 相关探索路径：

- **`src/rl4co_envs/evrptw_env.py`**：早期 EVRP-TW 环境框架（仅完成类和方法签名，多数方法为占位）
- **`src/POMO_test.py`**：早期训练入口（调用不完整的 `evrptw_env.py`，无法实际训练）
- **Week 4 附录**：在 TSP20 上验证了 POMO + RL4CO 框架可用（GPU 加速正常，~1分42秒/epoch）

---

## 2. POMO_Test_2.py 概述

本周在 `src/POMO_Test_2.py` 中完成了一个**自包含、可训练**的 EVRP-TW + POMO 实现。与之前分散在多个文件中的半成品不同，该文件将所有组件整合在一个模块中，从头到尾形成一个完整的训练管线。

### 2.1 设计目标

将 EVRP-TW 的硬约束（容量、电池、时间窗）编码进 POMO 的动作掩码机制中，使模型在训练过程中只能选择合法动作，从而学习到满足所有约束的路径构造策略。

### 2.2 文件结构

文件包含四个主要部分：

```
POMO_Test_2.py
├── 1. EVRP_TWGenerator    — 随机实例生成器
├── 2. EVRP_TWEnv           — RL4CO 环境（硬约束 + 动作掩码）
├── 3. Custom Embeddings    — 定制化节点/上下文嵌入
└── 4. train()              — 训练入口
```

---

## 3. 各组件详细介绍

### 3.1 EVRP_TWGenerator — 随机实例生成器

继承自 RL4CO 的 `Generator` 基类，负责按批次生成随机 EVRP-TW 实例。

**节点索引约定**：

| 索引范围 | 含义 |
|----------|------|
| `0` | 仓库（depot） |
| `1 .. num_stations` | 充电站 |
| `num_stations+1 .. total-1` | 客户 |

**生成的数据字段**：

| 字段 | 形状 | 说明 |
|------|------|------|
| `locs` | `[B, total_nodes, 2]` | 所有节点坐标（仓库固定于 (0.5, 0.5)） |
| `demand` | `[B, total_nodes-1]` | 客户需求（仓库和充电站为 0） |
| `tw_early` / `tw_late` | `[B, total_nodes]` | 时间窗（仓库和充电站为 0） |
| `service_time` | `[B, total_nodes]` | 服务时间（仓库和充电站为 0） |
| `battery_capacity` | `[B, 1]` | 电池最大容量 |
| `consumption_rate` | `[B, 1]` | 单位距离能耗 |
| `vehicle_capacity` | `[B, 1]` | 车辆最大载重 |
| `speed` | `[B, 1]` | 行驶速度（距离/时间） |

**可配置参数**（及默认值）：

```python
num_loc=5, num_stations=2,
battery_capacity=3.0, consumption_rate=1.0,
vehicle_capacity=30.0, speed=1.0,
depot_loc=(0.5, 0.5),
min_demand=1, max_demand=10,
tw_horizon=5.0, min_tw_width=5.0, max_tw_width=10.0,
min_service=1.0, max_service=3.0
```

**设计要点**：`num_loc` 在初始化时被覆写为 `total_nodes`（= 1 + num_stations + num_customers），因为 POMO 的解码步数需要覆盖所有节点（含仓库和充电站），而非仅客户数。

---

### 3.2 EVRP_TWEnv — 强化学习环境

继承自 `RL4COEnvBase`，是文件的核心。所有硬约束通过 `get_action_mask` 实现。

#### 状态空间（`_reset` 初始化）

| 状态变量 | 说明 |
|----------|------|
| `current_node` | 当前所在节点（初始为仓库 0） |
| `remaining_battery` | 剩余电量（初始 = battery_capacity） |
| `remaining_capacity` | 剩余载重（初始 = vehicle_capacity） |
| `current_time` | 当前时间（初始 = 0） |
| `total_distance` | 累计行驶距离 |
| `visited` | 布尔向量，标记已访问节点 |

#### 动作掩码（`get_action_mask`）

每一步，模型只能选择未被掩码屏蔽的节点。掩码逻辑包含五个层面：

1. **已访问掩码**：已访问节点不可再选（仓库除外，由第 5 条控制）
2. **容量掩码**：需求量 > 剩余容量的客户不可选
3. **电量掩码**：当前电量不足以到达的节点不可选
4. **时间窗掩码**：预计到达时间 > `tw_late` 的节点不可选
5. **仓库掩码**：当车辆在仓库且仍有可行客户时，禁止停留在仓库（强制继续服务）

这一设计确保模型在训练过程中永远不会执行非法动作，无需在奖励函数中加入约束违反惩罚。

#### 状态转移（`_step`）

执行动作后的更新逻辑：

| 事件 | 处理方式 |
|------|----------|
| 行驶距离 | 累加至 `total_distance` |
| 时间推进 | `arrival = current_time + travel_time`，若早于 `tw_early` 则等待（`max(arrival, tw_early)`），最后加上 `service_time` |
| 电量消耗 | `distance × consumption_rate` |
| 充电恢复 | 到达仓库或充电站时，电量重置为 `battery_capacity` |
| 容量消耗 | 到达客户时扣除需求；到达仓库时重置为 `vehicle_capacity` |
| 终止条件 | 所有客户均已访问 → `done = True` |

#### 多起点支持（POMO 核心机制）

```python
def get_num_starts(self, td):
    return self.generator.num_customers

def select_start_nodes(self, td, num_starts):
    # 从所有客户节点中选择不同的起始节点
```

POMO 通过从不同客户节点出发、探索多条轨迹，利用对称性增强策略梯度信号。

#### 奖励函数（`_get_reward`）

总奖励 = **负总路径长度**（含返回仓库的最后一段）：

```python
return -get_tour_length(full)  # full = [depot, action_seq..., depot]
```

#### 解验证（`check_solution_validity`）

提供了一个完整的解验证器，逐批次检查：
- 容量约束（每位客户的需求 ≤ 剩余容量）
- 时间窗约束（到达时间 ≤ `tw_late`）
- 电量约束（电量始终 ≥ 0）
- 每位客户恰好访问一次

---

### 3.3 定制化嵌入层

为适配 EVRP-TW 的节点特征，在运行时通过 monkey-patching 注入自定义嵌入：

**`_EVRPTWInitEmbedding`**：
- 仓库节点：仅使用 2 维坐标特征
- 非仓库节点：使用 4 维特征（坐标 x, y + 需求 demand + 最晚时间窗 tw_late）

**`_EVRPTWContext`**：
- 除默认上下文外，额外拼接 `remaining_capacity` 和 `current_time` 作为状态嵌入

通过替换 `rl4co.models.nn.env_embeddings.init` 和 `rl4co.models.nn.env_embeddings.context` 中的工厂函数实现注入，无需修改 RL4CO 源码。

---

### 3.4 训练入口（`train()`）

```python
def train():
    env = EVRP_TWEnv(
        generator_params=dict(
            num_loc=5,           # 5 个客户
            num_stations=2,      # 2 个充电站
            battery_capacity=10.0,
            vehicle_capacity=50.0,
            tw_horizon=2.0,
            min_tw_width=30.0,
            max_tw_width=50.0,
        ),
        check_solution=False,    # 训练时关闭验证以加速
    )
    model = POMO(env, batch_size=64, optimizer_kwargs={"lr": 1e-4})
    trainer = RL4COTrainer(max_epochs=1, accelerator="gpu", enable_progress_bar=True)
    trainer.fit(model)
```

默认配置为极小规模（5 客户 + 2 充电站），适合快速验证环境逻辑正确性。

---

## 4. 与之前版本的对比

| 维度 | 旧版 (`evrptw_env.py` + `POMO_test.py`) | 新版 (`POMO_Test_2.py`) |
|------|------------------------------------------|--------------------------|
| 文件组织 | 分散在 2 个文件，需跨文件 import | 单文件自包含 |
| 数据生成 | `_generate_data` 返回普通 dict | 使用 `Generator` 类 + `TensorDict` |
| `_reset` | 部分实现 | 完整实现，含所有状态字段 |
| `_step` | 未实现 | 完整实现，含距离/时间/电量/容量更新 |
| 动作掩码 | 未实现 | 五层硬约束掩码 |
| 充电逻辑 | 无 | 仓库/充电站自动充满 |
| 多起点 | 无 | 支持 POMO multi-start |
| 嵌入层 | 无 | 定制 4 维节点嵌入 + 上下文嵌入 |
| 解验证 | 无 | `check_solution_validity` 完整检查 |
| 可训练 | ❌ | ✅（待运行） |

---

## 5. 当前状态与待办事项

### 5.1 当前状态

| 组件 | 状态 |
|------|------|
| `POMO_Test_2.py` 代码编写 | ✅ 已完成 |
| 语法 / 导入检查 | ⚠️ 尚未验证 |
| 小规模训练测试（5 客户） | ⏳ 待运行 |
| 训练结果分析 | ⏳ 待进行 |
| 解质量评估 | ⏳ 待进行 |
| 与 OR-Tools / PyVRP 基线对比 | ⏳ 待进行 |

### 5.2 下一步计划

1. **运行训练**：执行 `python src/POMO_Test_2.py`，验证代码无运行时错误，观察首个 epoch 的 loss 曲线和奖励趋势。
2. **检查可行性**：训练后使用 `check_solution_validity` 验证模型生成的解是否满足所有硬约束。
3. **小规模调参**：若 5 客户实例训练稳定，逐步增加客户数（10、20），调整 `batch_size` 和 `max_epochs`。
4. **对比基线**：将 POMO 的解与 OR-Tools / PyVRP 在相同随机实例上的解进行对比。
5. **扩展到真实数据**：用 ESOGU 或 Solomon 格式的真实 EVRP-TW 实例替换随机生成数据。

---

## 6. 技术备注

- **RL4CO 版本兼容性**：代码基于 RL4CO 的 `RL4COEnvBase`、`POMO`、`RL4COTrainer` API，若 RL4CO 版本更新导致 API 变更，需相应调整。
- **嵌入层注入方式**：当前通过运行时替换工厂函数实现自定义嵌入，这是一种临时方案。若 RL4CO 后续版本提供正式的注册机制，建议迁移至官方方式。
- **掩码效率**：当前 `get_action_mask` 基于逐元素张量操作，在小规模（总节点 < 50）上应无性能瓶颈。若扩展到大规模（100+ 节点），可考虑使用稀疏操作优化。
- **默认参数说明**：`train()` 中的默认参数（如 `battery_capacity=10.0`、`tw_horizon=2.0`、`min_tw_width=30.0`）为测试用途，实际训练时应根据目标数据集调整。

---

## 附录：文件变更摘要

```
本周新增：
  src/POMO_Test_2.py          — 自包含 EVRP-TW + POMO 训练管线（627 行）

此前已有（RL 相关）：
  src/POMO_test.py            — 早期训练入口（调用不完整环境）
  src/rl4co_envs/__init__.py  — 环境包初始化
  src/rl4co_envs/evrptw_env.py — 早期环境框架（未完成）
```
