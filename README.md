
* * *

## SEP 暑研项目：车辆路径问题与扩展优化

## 中文版

### 项目简介

本项目是 SEP 暑期科研项目的一部分，主要研究带容量约束的车辆路径问题（CVRP）、带时间窗的车辆路径问题（VRPTW）、电动车辆路径问题（EVRP-TW）以及卡车-无人机协同路径优化。项目使用 Python 和 PyVRP 求解器作为基线，后续将引入强化学习方法（POMO、Attention Model）并扩展至更复杂的现实约束。

### 仓库结构

```
.
├── data/                      # 数据集（CVRP / VRPTW 实例）
│   ├── A-n32-k5/              # Augerat CVRP 实例
│   └── solomon/               # Solomon VRPTW 实例
├── docs/                      # 文档与笔记
│   ├── 00_weekly.md           # 每周进度记录
│   └── meeting_notes/         # 会议笔记（分周存放）
├── notebooks/                 # Jupyter Notebook 探索性分析
├── results/                   # 求解结果（文本 / 图表）
├── src/                       # 源代码（Python 脚本）
├── FURP_Showcase.pdf          # 项目展示海报
├── .gitignore                 # Git 忽略规则
└── README.md                  # 本文件
```

### 已完成工作

*   环境配置：Python 3.14 + PyVRP + vrplib 安装，VSCode 开发环境
    
*   基线求解：成功求解 CVRP 经典实例 A-n32-k5，达到已知最优解（784，0% 差距）
    
*   数据准备：整理 Augerat 和 Solomon 数据集，支持 .vrp 和 .sol 文件读取
    
*   脚本编写：自动路径解析、求解并输出结果（可行性、总距离、车辆数、运行时间、最优差距）
    

### 下一步计划

*   批量测试多个 CVRP / VRPTW 实例并记录对比表格
    
*   扩展至 VRPTW（Solomon 实例），输出时间窗违反情况
    
*   实现 EVRP-TW 的简单建模（电池 + 充电站）
    
*   搭建卡车-无人机协同的启发式搜索
    
*   （可选）引入 POMO / Attention Model 进行学习型策略对比
    

### 运行方式

1.  克隆仓库并进入目录：
    
    git clone <repo-url\>
    cd <repo-name\>
    
2.  创建虚拟环境并激活：
    
    python3 \-m venv .venv
    source .venv/bin/activate
    
3.  安装依赖：
    
    pip install pyvrp vrplib
    
4.  运行示例求解脚本：
    
    python src/run\_first.py
    

### 环境要求

*   Python 3.9+
    
*   macOS / Linux / Windows（推荐 macOS 或 WSL）
    
*   PyVRP 0.12+
    
*   vrplib
    

### 联系方式

如有问题，请通过 GitHub Issues 联系或发送邮件至项目负责人。

* * *

## English Version

### Project Overview

This project is part of the SEP summer research program, focusing on Capacitated Vehicle Routing Problem (CVRP), Vehicle Routing Problem with Time Windows (VRPTW), Electric Vehicle Routing Problem with Time Windows (EVRP-TW), and truck-drone collaborative routing. The project uses PyVRP as a baseline solver, with future extensions to reinforcement learning methods (POMO, Attention Model) and more complex real-world constraints.

### Repository Structure

```
.
├── data/                      # Datasets (CVRP / VRPTW instances)
│   ├── A-n32-k5/              # Augerat CVRP instance
│   └── solomon/               # Solomon VRPTW instances
├── docs/                      # Documentation and notes
│   ├── 00_weekly.md           # Weekly progress log
│   └── meeting_notes/         # Meeting notes (by week)
├── notebooks/                 # Jupyter Notebooks for exploratory analysis
├── results/                   # Output results (text / figures)
├── src/                       # Source code (Python scripts)
├── FURP_Showcase.pdf          # Project poster
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Completed Work

*   Environment setup: Python 3.14 + PyVRP + vrplib installed, VSCode ready
    
*   Baseline solved: CVRP classic instance A-n32-k5 achieved known optimum (784, 0% gap)
    
*   Data preparation: Augerat and Solomon datasets organized, support .vrp and .sol files
    
*   Script writing: auto path resolution, solve and output results (feasibility, total distance, #vehicles, runtime, optimality gap)
    

### Next Steps

*   Batch test multiple CVRP / VRPTW instances and record comparison tables
    
*   Extend to VRPTW (Solomon instances), report time window violations
    
*   Implement a simple EVRP-TW model (battery + charging stations)
    
*   Build a heuristic search for truck-drone coordination
    
*   (Optional) Introduce POMO / Attention Model for learning-based comparison
    

### How to Run

1.  Clone the repository and enter the directory:
    
    git clone <repo-url\>
    cd <repo-name\>
    
2.  Create and activate a virtual environment:
    
    python3 \-m venv .venv
    source .venv/bin/activate
    
3.  Install dependencies:
    
    pip install pyvrp vrplib
    
4.  Run the example script:
    
    python src/run\_first.py
    

### Requirements

*   Python 3.9+
    
*   macOS / Linux / Windows (macOS or WSL recommended)
    
*   PyVRP 0.12+
    
*   vrplib
    

### Contact

Please use GitHub Issues or email the project lead for any questions.

* * *
