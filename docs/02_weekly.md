# 阅读笔记：车辆路径问题（VRP）及其求解方法

> 基于以下材料：
> - [OR-Tools Routing 官方文档](https://developers.google.com/optimization/routing?hl=zh-cn)
> - Kool et al., *Attention, Learn to Solve Routing Problems!* (ICLR 2019)
> - Kwon et al., *POMO: Policy Optimization with Multiple Optima for Reinforcement Learning* (NeurIPS 2020)
>
> 笔记日期：2026-06-17

---

## 1. 什么是车辆路径问题（VRP）？

车辆路径问题（Vehicle Routing Problem, VRP）是组合优化领域最经典的问题之一。简单来说：

> **有一组客户需要送货，车队从仓库出发，如何规划路线使总成本（通常是行驶距离）最小？**

这个问题在物流配送、外卖送餐、垃圾回收、校车路线规划等场景中无处不在。

VRP 属于 **NP-hard** 问题，意味着随着客户数量增加，求解难度呈指数级增长，不存在能在多项式时间内找到最优解的通用算法。因此，实际应用中通常采用：

| 方法类型       | 特点                                               | 适用场景             |
| -------------- | -------------------------------------------------- | -------------------- |
| **精确算法**   | 保证找到最优解，但只适用于小规模（如 < 50 个客户） | 理论研究、小规模验证 |
| **启发式算法** | 在合理时间内找到“足够好”的解，不保证最优           | 大多数实际应用       |
| **学习型方法** | 用神经网络从数据中学习构建路径的策略               | 需要快速推理的场景   |

---

## 2. 问题变体：从简单到复杂

OR-Tools 文档和两篇论文共同勾勒出一条清晰的“问题进化路线”：

| 变体        | 全称                           | 核心约束                                   | 现实场景           |
| ----------- | ------------------------------ | ------------------------------------------ | ------------------ |
| **CVRP**    | Capacitated VRP                | 车辆有**容量上限**                         | 货车载重有限       |
| **VRPTW**   | VRP with Time Windows          | 客户有**时间窗**（必须在指定时间段内到达） | 快递时效、外卖配送 |
| **EVRP-TW** | Electric VRP with Time Windows | 电池容量 + 充电站 + 时间窗                 | 电动货车配送       |

### 2.1 CVRP（带容量约束的车辆路径问题）

CVRP 是 VRP 最基本的扩展。每辆车有最大载重，所有客户的需求之和不能超过车辆容量。

**OR-Tools 核心实现思路**：
```python
# 创建容量维度
routing.AddDimension(
    demand_callback,    # 获取客户需求的回调
    0,                  # slack（容量不能为负，设为0）
    vehicle_capacity,   # 车辆容量上限
    True,               # 起点累积值强制为0
    'Capacity'          # 维度名称
)
```

## 2.2 VRPTW（带时间窗的车辆路径问题）

VRPTW 在 CVRP 基础上增加时间窗约束：每个客户有一个允许到达的时间区间 `[e_i, l_i]`，车辆必须在区间内开始服务。

**OR-Tools 核心实现思路**：

```python
# 创建时间维度
routing.AddDimension(
    time_callback,      # 计算行驶时间的回调
    slack,              # 允许等待的时间
    max_time,           # 每辆车最大总时间
    False,              # 起点累积值不强制为0
    'Time'              # 维度名称
)
time_dimension = routing.GetDimensionOrDie('Time')

# 为每个客户设置时间窗
for location_idx, time_window in enumerate(data['time_windows']):
    if location_idx == depot:
        continue
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
```

### 2.3 EVRP-TW（带时间窗的电动车辆路径问题）

EVRP-TW 是 Schneider 等人在 2014 年提出的问题，增加了电动车特有的约束：

1.  **电池容量 (Q)**：车辆的最大电量
    
2.  **电量消耗率 (h)**：单位距离耗电量
    
3.  **充电站 (F)**：路径中可补充电量的节点
    
4.  **充电时间**：取决于到达时的剩余电量和充电速率
    

**论文中的核心参数**：

| 参数 | 符号 | 说明 |
| --- | --- | --- |
| 电池容量 | Q | 车辆最大电量（kWh） |
| 消耗率 | h | 每公里耗电量（kWh/km） |
| 充电速率 | g | 单位时间充电量（kWh/min） |
| 到达电量 | y\_i | 到达节点 i 时的剩余电量 |

* * *

## 3\. 求解方法演进

### 3.1 传统方法：OR-Tools 路由求解器

OR-Tools 是 Google 开发的开源优化工具包，其路由求解器采用 **启发式搜索**（如局部搜索、禁忌搜索）结合元启发式算法，能在合理时间内给出高质量解。

**核心优势**：

*   通过 `AddDimension()` 灵活添加各类约束
    
*   支持 CVRP、VRPTW、带 pickup/delivery 的多种变体
    
*   提供 Python / C++ / Java 等多语言接口
    
*   生产环境验证，稳定可靠
    

**OR-Tools 求解流程**：

```python
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# 1. 创建路由模型
manager = pywrapcp.RoutingIndexManager(len(data), num_vehicles, depot)
routing = pywrapcp.RoutingModel(manager)

# 2. 定义距离回调
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# 3. 设置优化目标
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# 4. 添加约束（通过 Dimension）

# 5. 求解
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
solution = routing.SolveWithParameters(search_parameters)
```

### 3.2 深度学习路线①：Attention Model（AM）

Kool 等人在 2019 年提出基于 **注意力机制** 的端到端学习框架。

**核心架构**：

**核心架构**：

Attention Model 采用编码器-解码器结构，整体流程如下：

1. **输入层**：接收所有节点的二维坐标 `(x, y)` 作为输入特征。

2. **编码器（Encoder）**：由 N 层（论文中 N=3）相同的注意力层堆叠而成。每一层包含：
   - **多头注意力层（Multi-Head Attention）**：使用 M=8 个注意力头，每个头的维度为 d_k = 16，让节点之间相互传递信息
   - **前馈网络（Feed-Forward）**：隐藏层维度 d_ff = 512，使用 ReLU 激活函数
   - **残差连接 + 批归一化（Batch Normalization）**：每层之后添加，稳定训练过程

   编码器的输出包含两部分：
   - **节点嵌入（Node Embeddings）** `h_i`：每个节点的向量表示，包含其在图中上下文信息
   - **图嵌入（Graph Embedding）** `h̄`：所有节点嵌入的平均值，表示整个问题的全局信息

3. **解码器（Decoder）**：自回归地逐个生成路径节点。在每一步：
   - **构建上下文**：将图嵌入、当前节点（路径最后一个节点）和起点节点拼接，构成上下文向量
   - **掩码（Masking）**：屏蔽所有已访问过的节点，确保不会重复访问
   - **单头注意力计算**：通过注意力机制计算上下文与所有未访问节点的匹配度（logits）
   - **Softmax 输出**：将 logits 转换为概率分布，选择下一个访问节点

4. **输出层**：重复解码过程直到所有节点被访问，最终输出一个完整的节点排列 `π = (π₁, π₂, ..., πₙ)`，即车辆的行驶路径。

**关键设计**：

*   **编码器**：Transformer 风格，N=3 层，M=8 个注意力头，d\_h=128
    
*   **解码器**：每一步使用当前节点、第一个节点、图嵌入作为上下文
    
*   **训练方法**：REINFORCE + Greedy Rollout Baseline
    

**训练方法对比**：

| 方法 | Baseline 类型 | 特点 |
| --- | --- | --- |
| Bello et al. | Actor-Critic（价值网络） | 需要额外训练 Critic |
| **AM** | Greedy Rollout | 用当前最优模型的贪心解作为基准，周期性更新 |
| **POMO** | Shared Baseline | 多条轨迹互相比较，方差更小 |

**实验结果**（TSP100）：

| 方法 | 最优性差距 | 推理时间 |
| --- | --- | --- |
| AM（贪心） | 3.51% | < 1 秒 |
| AM（采样1280） | 1.98% | 22 分钟 |
| Bello et al.（采样） | 3.03% | ~10 秒 |

### 3.3 深度学习路线②：POMO

Kwon 等人在 2020 年提出 POMO（Policy Optimization with Multiple Optima），核心洞察：

> **VRP 的最优解存在“对称性”**——同一个环路可以从任意一个节点开始描述。传统模型固定使用一个 START token，浪费了这种对称性。

**POMO 的核心创新**：

**① 多起点探索**

对比传统方法与 POMO 方法的差异：

```text
传统方法：
START token → 选择第一个节点（由模型决定）→ 后续节点

POMO 方法：
节点1作为起点 → 生成路线1
节点2作为起点 → 生成路线2    → 互相比较 → 共同改进
...
节点N作为起点 → 生成路线N
```
**② 共享基线**

传统 Greedy Rollout：每条轨迹独立与自己比较（样本轨迹 vs 贪心轨迹）

POMO Shared Baseline：

baseline = 所有 N 条轨迹的平均值
每条轨迹的 advantage = 该轨迹的收益 - baseline

优势：方差更低，不易陷入局部最优。

**③ 多贪心推理 + 实例增强**

推理时从 N 个起点各生成一条贪心路线，取最优。还可以通过旋转/翻转坐标生成 ×8 个变体，进一步提升。

**实验结果对比**：

| 方法 | TSP20 | TSP50 | TSP100 |
| --- | --- | --- | --- |
| AM（采样1280） | 0.07% | 0.39% | 1.98% |
| POMO（×8 增强） | **0.00%** | **0.03%** | **0.14%** |
| POMO 推理时间 | ≤ 3s | ≤ 16s | ≤ 1min |

**结论**：POMO 在所有规模上均显著优于 AM，且推理时间快一个数量级以上。

* * *

## 4\. EVRP-TW 约束图

以下约束图展示了一个典型的 **EVRP-TW** 问题中各个元素和约束之间的关系：

```mermaid
graph TD
    subgraph 核心元素
        Depot(仓库)
        Customer(客户)
        Station(充电站)
        Vehicle(车辆)
    end

    subgraph 约束类型
        Cap[容量约束]
        TW[时间窗约束]
        Bat[电池约束]
        Cha[充电约束]
    end

    subgraph 目标
        Obj[最小化总距离]
    end

    Depot -->|"出发和返回"| Vehicle
    Vehicle -->|"服务"| Customer
    Vehicle -->|"补电"| Station

    Customer -.->|"需求不超过容量"| Cap
    Customer -.->|"到达时间在窗口内"| TW
    Vehicle -.->|"电量在0到Q之间"| Bat
    Station -.->|"充电后电量恢复"| Cha

    Cap --> Obj
    TW --> Obj
    Bat --> Obj
    Cha --> Obj

    style Depot fill:#f9f,stroke:#333
    style Customer fill:#bbf,stroke:#333
    style Station fill:#bfb,stroke:#333
    style Vehicle fill:#fcf,stroke:#333
    style Cap fill:#ffe0b2,stroke:#333
    style TW fill:#ffe0b2,stroke:#333
    style Bat fill:#ffe0b2,stroke:#333
    style Cha fill:#ffe0b2,stroke:#333
    style Obj fill:#c8e6c9,stroke:#333
​```
```
