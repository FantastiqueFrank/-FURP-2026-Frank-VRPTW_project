import math
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ==================== 数据读取 ====================
def read_esogu_data(file_path):
    """读取 ESOGU 节点信息文件，只保留类型为 d, c, cs 的节点"""
    nodes = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        typ = parts[1]
        if typ not in ('d', 'c', 'cs'):
            continue
        try:
            node_id_str = parts[0]
            lat = float(parts[2])
            lon = float(parts[3])
            demand = float(parts[4]) if parts[4] != '0' else 0.0
            prize = float(parts[5]) if parts[5] != '0' else 0.0
            service = float(parts[6])
            tw_early = float(parts[7])
            tw_late = float(parts[8])
        except ValueError:
            continue
        nodes.append({
            'id_str': node_id_str,
            'type': typ,
            'lat': lat,
            'lon': lon,
            'demand': demand,
            'prize': prize,
            'service': service,
            'tw_early': tw_early,
            'tw_late': tw_late
        })
    return nodes


def read_distance_matrix(file_path, n):
    """
    读取 ESOGU 距离矩阵文件，提取前 n 行和前 n 列的距离值
    文件格式：第一行是列索引，后续每行第一个是行号，后面是距离值
    """
    dist = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 跳过第一行（列索引）
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        # 跳过行号，取后面的数值，但只取前 n 个
        if len(parts) >= n + 1:
            row = list(map(float, parts[1:1 + n]))
        elif len(parts) >= n:
            row = list(map(float, parts[:n]))
        else:
            continue
        if len(row) == n:
            dist.append(row)
        if len(dist) == n:
            break
    if len(dist) != n:
        raise ValueError(f"无法读取足够的距离矩阵行，需要 {n} 行，实际 {len(dist)}")
    return dist


# ==================== 可视化函数 ====================
def plot_solution(nodes, manager, routing, solution, num_vehicles):
    """绘制所有车辆的路线图"""
    plt.figure(figsize=(12, 10))

    # 提取所有节点的坐标（经度, 纬度）
    lons = [node['lon'] for node in nodes]
    lats = [node['lat'] for node in nodes]

    # 绘制仓库（用红色星标）
    depot_idx = manager.IndexToNode(routing.Start(0))
    plt.scatter(lons[depot_idx], lats[depot_idx], c='red', s=200, marker='*', label='Depot', zorder=5)

    # 绘制客户（用蓝色圆圈）
    customer_indices = [i for i, node in enumerate(nodes) if node['type'] == 'c']
    for i in customer_indices:
        plt.scatter(lons[i], lats[i], c='blue', s=80, marker='o', label='Customer' if i == customer_indices[0] else "")

    # 绘制充电站（用绿色方块）
    station_indices = [i for i, node in enumerate(nodes) if node['type'] == 'cs']
    for i in station_indices:
        plt.scatter(lons[i], lats[i], c='green', s=80, marker='s',
                    label='Charging Station' if i == station_indices[0] else "")

    # 为每辆车绘制路线
    colors = ['orange', 'purple', 'cyan', 'magenta', 'brown', 'gray']
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_x = []
        route_y = []
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            route_x.append(lons[node_idx])
            route_y.append(lats[node_idx])
            index = solution.Value(routing.NextVar(index))
        # 添加终点（仓库）
        node_idx = manager.IndexToNode(index)
        route_x.append(lons[node_idx])
        route_y.append(lats[node_idx])
        # 绘制路线
        color = colors[vehicle_id % len(colors)]
        plt.plot(route_x, route_y, color=color, linewidth=2, marker='o', markersize=6,
                 label=f'Vehicle {vehicle_id}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('EVRP-TW Route Map (Simplified Battery Constraint)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ==================== 主函数 ====================
def main():
    # 文件路径（请根据实际位置修改）
    node_file = r"D:\Projects\-FURP-2026-Frank-VRPTW_project-main\data\ESOGU-EVRP-PD-TW DATASET\ESOGU Dataset-EVRP_PD_TW1\ESOGU_C20_TW1.txt"
    dist_file = r"D:\Projects\-FURP-2026-Frank-VRPTW_project-main\data\ESOGU-EVRP-PD-TW DATASET\V3.2_DistanceMatrix_SUIT_PDP_TW1\Distance_ESOGU_C20_TW1.txt"

    nodes = read_esogu_data(node_file)
    n = len(nodes)
    print(f"成功读取 {n} 个节点")
    if n == 0:
        return

    distance_matrix = read_distance_matrix(dist_file, n)
    print(f"成功读取 {len(distance_matrix)}×{len(distance_matrix[0])} 距离矩阵")

    # ========== EVRP-TW 电池参数（可调整） ==========
    BATTERY_CAPACITY = 10000  # kWh（设置很大，使约束不紧）
    CONSUMPTION_RATE = 0.2  # kWh/km
    # 速度转换系数（距离 → 时间，单位分钟）
    SPEED_MPS = 12.5  # 12.5 米/秒
    DIST_TO_TIME = 1 / (SPEED_MPS * 60)  # 距离(m) → 时间(min)
    # ===========================================

    depot_indices = [i for i, node in enumerate(nodes) if node['type'] == 'd']
    if not depot_indices:
        raise ValueError("未找到仓库节点")
    depot = depot_indices[0]
    print(f"仓库节点索引: {depot}, ID: {nodes[depot]['id_str']}")

    customer_indices = [i for i, node in enumerate(nodes) if node['type'] == 'c']
    print(f"客户数量: {len(customer_indices)}")

    vehicle_capacity = 350
    num_vehicles = 6

    # ========== 构建模型 ==========
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def safe_index_to_node(index):
        try:
            node = manager.IndexToNode(index)
            if node < 0 or node >= n:
                return -1
            return node
        except:
            return -1

    # 距离回调（返回整数）
    def distance_callback(from_index, to_index):
        from_node = safe_index_to_node(from_index)
        to_node = safe_index_to_node(to_index)
        if from_node == -1 or to_node == -1:
            return 100000
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 容量约束
    def demand_callback(from_index):
        from_node = safe_index_to_node(from_index)
        if from_node == -1:
            return 0
        return int(nodes[from_node]['demand'])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity] * num_vehicles,
        True,
        'Capacity'
    )

    # 允许跳过充电站
    for i, node in enumerate(nodes):
        if node['type'] == 'cs':
            index = manager.NodeToIndex(i)
            routing.AddDisjunction([index], 0)

    # ========== 添加时间窗约束 ==========
    def time_callback(from_index, to_index):
        from_node = safe_index_to_node(from_index)
        to_node = safe_index_to_node(to_index)
        if from_node == -1 or to_node == -1:
            return 100000
        travel_time = distance_matrix[from_node][to_node] * DIST_TO_TIME
        service_time = nodes[from_node]['service']
        return int(travel_time + service_time)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(
        time_callback_index,
        500,  # slack 足够大，允许大量等待
        20000,  # max_time 足够大
        False,
        'Time'
    )
    time_dimension = routing.GetDimensionOrDie('Time')

    for i, node in enumerate(nodes):
        if node['type'] == 'c':
            index = manager.NodeToIndex(i)
            time_dimension.CumulVar(index).SetRange(
                int(node['tw_early']),
                int(node['tw_late'])
            )

    depot_index = manager.NodeToIndex(depot)
    time_dimension.CumulVar(depot_index).SetRange(0, 5000)

    # # ========== 添加电池（能量）约束 ==========
    # def energy_callback(from_index, to_index):
    #     from_node = safe_index_to_node(from_index)
    #     to_node = safe_index_to_node(to_index)
    #     if from_node == -1 or to_node == -1:
    #         return 0
    #     # 对充电站不耗电（也不充电）
    #     if nodes[to_node]['type'] == 'cs':
    #         return 0
    #     # 普通弧消耗：距离(m) * 0.2 kWh/km = 距离 * 0.0002 kWh/m
    #     consumption = distance_matrix[from_node][to_node] * CONSUMPTION_RATE  # float
    #     return int(consumption + 0.5)  # 四舍五入取整
    #
    # energy_callback_index = routing.RegisterTransitCallback(energy_callback)
    # routing.AddDimension(
    #     energy_callback_index,
    #     0,  # slack（不允许负电量）
    #     BATTERY_CAPACITY,  # 电池容量上限
    #     True,  # 起始累积值为0（满电）
    #     'Energy'
    # )
    # # energy_dimension = routing.GetDimensionOrDie('Energy')  # 可提取用于输出

    # ========== 求解 ==========
    print("\n尝试求解 EVRP-TW（带时间窗 + 电池约束）...")
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # 1. 更有效的初始解策略（对时间窗问题更友好）
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    # 2. 使用禁忌搜索（在复杂约束下通常效果更好）
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    )
    # 3. 给求解器更多时间（5分钟）
    search_parameters.time_limit.seconds = 120
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print(f"EVRP-TW 求解成功！总行驶距离: {solution.ObjectiveValue()}")
        print("\n详细路线：")
        time_dimension = routing.GetDimensionOrDie('Time')
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_str = f"车辆 {vehicle_id}: "
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                route_str += f"{node_idx} (ID={nodes[node_idx]['id_str']}, 到达时间 {solution.Min(time_var):.0f}) -> "
                index = solution.Value(routing.NextVar(index))
            node_idx = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route_str += f"{node_idx} (到达时间 {solution.Min(time_var):.0f})"
            print(route_str)

        # 可视化
        print("\n正在生成路线图...")
        plot_solution(nodes, manager, routing, solution, num_vehicles)
    else:
        print("EVRP-TW 无可行解！请调整参数。")


if __name__ == "__main__":
    main()