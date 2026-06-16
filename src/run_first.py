# src/run_first.py

import os
import vrplib
from pyvrp import read, solve
from pyvrp.stop import MaxRuntime

# 路径配置
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
vrp_file = os.path.join(project_root, "data", "A-n32-k5", "A-n32-k5.vrp")
sol_file = os.path.join(project_root, "data", "A-n32-k5", "A-n32-k5.sol")

instance = read(vrp_file, round_func="round")

bks = None
if os.path.exists(sol_file):
    bks = vrplib.read_solution(sol_file)
    print(f"已知最优解成本: {bks['cost']}")

result = solve(instance, stop=MaxRuntime(60), seed=42, display=False)

if result:
    print(f"\n可行: {result.is_feasible()}")
    print(f"总行驶距离: {result.cost():.2f}")
    print(f"车辆数: {result.best.num_routes()}")
    print(f"求解器运行时间: {result.runtime:.2f} 秒")   # 修正：runtime 是属性，不是方法
    if bks:
        gap = (result.cost() - bks['cost']) / bks['cost'] * 100
        print(f"与最优解差距: {gap:.2f}%")
else:
    print("无可行解")