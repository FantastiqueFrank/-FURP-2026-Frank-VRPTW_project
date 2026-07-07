import math
import os
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ==================== VRP 文件解析 ====================
def parse_vrp_file(file_path):
    """
    解析标准 CVRP .vrp 文件，返回：
        coords: list of (x, y) 坐标，索引从0开始（0为仓库）
        demands: list of 需求，索引从0开始
        capacity: 车辆容量
        dimension: 节点总数
        depot_index: 仓库索引（通常为0）
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    dimension = None
    capacity = None
    coords = []
    demands = []
    depot = None

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('NAME'):
            pass
        elif line.startswith('TYPE'):
            pass
        elif line.startswith('COMMENT'):
            pass
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('CAPACITY'):
            capacity = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            pass
        elif line == 'NODE_COORD_SECTION':
            section = 'coords'
        elif line == 'DEMAND_SECTION':
            section = 'demands'
        elif line == 'DEPOT_SECTION':
            section = 'depot'
        elif line == 'EOF':
            break
        else:
            if section == 'coords':
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
            elif section == 'demands':
                parts = line.split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    demand = float(parts[1])
                    demands.append(demand)
            elif section == 'depot':
                parts = line.split()
                if parts[0] != '-1':
                    depot = int(parts[0]) - 1

    if dimension is None:
        dimension = len(coords)
    if len(coords) != dimension:
        print(f"警告：实际坐标数 {len(coords)} 与 DIMENSION {dimension} 不一致，使用实际坐标数")
        dimension = len(coords)
    if len(demands) != dimension:
        print(f"警告：需求数 {len(demands)} 与节点数 {dimension} 不一致，将缺失的需求设为0")
        while len(demands) < dimension:
            demands.append(0)

    if depot is None:
        depot = 0

    coords = coords[:dimension]
    demands = demands[:dimension]

    return coords, demands, capacity, dimension, depot


# ==================== 欧几里得距离计算 ====================
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ==================== 主函数 ====================
def main():
    # 文件路径（请根据实际位置修改）
    vrp_file = r"D:\Projects\-FURP-2026-Frank-VRPTW_project-main\data\VRP_dataset\A-n55-k9\A-n55-k9.vrp"
    sol_file = r"D:\Projects\-FURP-2026-Frank-VRPTW_project-main\data\VRP_dataset\A-n55-k9\A-n55-k9.sol"

    # 1. 解析 VRP 文件
    coords, demands, capacity, n, depot = parse_vrp_file(vrp_file)
    print(f"节点数: {n}, 容量: {capacity}, 仓库索引: {depot}")

    # 2. 构建距离矩阵（欧几里得距离，四舍五入为整数，与 PyVRP 的 round_func="round" 一致）
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = euclidean_distance(coords[i], coords[j])
                dist_matrix[i][j] = int(round(d))
            else:
                dist_matrix[i][j] = 0

    # 3. 设定车辆数（足够大，以便 OR-Tools 可以自由使用）
    num_vehicles = 20  # 可以设为较大的数

    # 4. 构建 OR-Tools 模型
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # 距离回调
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 容量约束
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [capacity] * num_vehicles,
        True,
        'Capacity'
    )

    # ========== 修改搜索参数以追求更好解质量 ==========
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    # 初始解策略
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    # 使用禁忌搜索，并强制运行满设定时间
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )

    # 设置更长的时间限制（例如 300 秒）
    search_parameters.time_limit.seconds = 300

    # 增加解数量限制（可选）
    search_parameters.solution_limit = 100

    # 关闭日志（或设为 True 观察进度）
    search_parameters.log_search = False

    # =================================================

    # 6. 求解
    start_time = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    elapsed = time.time() - start_time

    if solution:
        total_distance = solution.ObjectiveValue()
        print(f"\n可行: {total_distance > 0}")
        print(f"总行驶距离: {total_distance:.2f}")

        # 计算实际使用的车辆数
        used_vehicles = 0
        for vehicle_id in range(num_vehicles):
            if solution.Value(routing.NextVar(routing.Start(vehicle_id))) != routing.End(vehicle_id):
                used_vehicles += 1
        print(f"车辆数: {used_vehicles}")

        print(f"求解器运行时间: {elapsed:.2f} 秒")

        # 如果存在已知最优解文件，读取并比较
        if os.path.exists(sol_file):
            import vrplib
            bks = vrplib.read_solution(sol_file)
            print(f"已知最优解成本: {bks['cost']}")
            gap = (total_distance - bks['cost']) / bks['cost'] * 100
            print(f"与最优解差距: {gap:.2f}%")
        else:
            print("未找到 .sol 文件，无法对比最优解")

        # 可选：打印每条路线
        # for vehicle_id in range(used_vehicles):
        #     index = routing.Start(vehicle_id)
        #     route = []
        #     while not routing.IsEnd(index):
        #         route.append(manager.IndexToNode(index))
        #         index = solution.Value(routing.NextVar(index))
        #     route.append(manager.IndexToNode(index))
        #     print(f"车辆 {vehicle_id}: {route}")
    else:
        print("无可行解")


if __name__ == "__main__":
    main()