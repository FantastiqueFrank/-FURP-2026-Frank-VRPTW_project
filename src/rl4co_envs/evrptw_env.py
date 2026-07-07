import torch
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from typing import Optional, Tuple, Dict, Any

log = get_pylogger(__name__)

class EVRP_TWEnv(RL4COEnvBase):
    """
    Electric Vehicle Routing Problem with Time Windows (EVRP-TW) environment.
    """
    name = "evrptw"

    def __init__(
        self,
        num_loc: int = 20,
        num_stations: int = 3,
        battery_capacity: float = 80.0,
        consumption_rate: float = 0.2,
        vehicle_capacity: float = 100.0,
        speed: float = 1.0,  # 距离单位/时间单位
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc  # 客户数
        self.num_stations = num_stations
        self.battery_capacity = battery_capacity
        self.consumption_rate = consumption_rate
        self.vehicle_capacity = vehicle_capacity
        self.speed = speed

    def _generate_data(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """生成 EVRP-TW 实例数据"""
        # 客户坐标 (均匀分布)
        loc = torch.rand(batch_size, self.num_loc, 2, dtype=torch.float32)
        # 需求 (整数，1-10)
        demand = torch.randint(1, 10, (batch_size, self.num_loc), dtype=torch.float32)
        # 时间窗 (随机生成，[0, 200] 范围)
        tw_early = torch.randint(0, 100, (batch_size, self.num_loc), dtype=torch.float32)
        tw_late = tw_early + torch.randint(10, 50, (batch_size, self.num_loc), dtype=torch.float32)
        # 充电站位置 (可与客户同分布)
        station_loc = torch.rand(batch_size, self.num_stations, 2, dtype=torch.float32)
        # 仓库位置 (可能固定)
        depot_loc = torch.zeros(batch_size, 1, 2, dtype=torch.float32)
        # 所有节点坐标 [depot, stations, customers]
        all_loc = torch.cat([depot_loc, station_loc, loc], dim=1)  # (batch, 1+N_sta+N_cus, 2)

        return {
            "loc": all_loc,
            "demand": demand,
            "tw_early": tw_early,
            "tw_late": tw_late,
            "depot_idx": 0,
            "station_indices": list(range(1, 1 + self.num_stations)),
            "customer_indices": list(range(1 + self.num_stations, 1 + self.num_stations + self.num_loc)),
            "battery_capacity": self.battery_capacity,
            "consumption_rate": self.consumption_rate,
            "vehicle_capacity": self.vehicle_capacity,
            "speed": self.speed,
        }

    def _reset(self, td: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """重置状态，返回初始状态"""
        batch_size = td["loc"].shape[0]
        # 初始位置：仓库 (索引0)
        current_node = torch.zeros(batch_size, dtype=torch.long)
        # 剩余电量：满电
        remaining_battery = torch.full((batch_size,), self.battery_capacity, dtype=torch.float32)
        # 剩余容量：满容量
        remaining_capacity = torch.full((batch_size,), self.vehicle_capacity, dtype=torch.float32)
        # 已访问客户 (布尔掩码)
        visited = torch.zeros(batch_size, self.num_loc, dtype=torch.bool)
        # 已服务总量
        served_demand = torch.zeros(batch_size, dtype=torch.float32)
        # 累计距离
        total_distance = torch.zeros(batch_size, dtype=torch.float32)
        # 当前时间 (从0开始)
        current_time = torch.zeros(batch_size, dtype=torch.float32)

        return {
            "current_node": current_node,
            "remaining_battery": remaining_battery,
            "remaining_capacity": remaining_capacity,
            "visited": visited,
            "served_demand": served_demand,
            "total_distance": total_distance,
            "current_time": current_time,
        }

    def _step(self, td: Dict[str, torch.Tensor], action: torch.LongTensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool, Dict]:
        """执行一步动作，返回新状态、奖励、是否结束、额外信息"""
        # 从 td 中提取当前状态 (略)
        # 计算移动距离、消耗、时间、容量变化等
        # 更新状态
        # 奖励 = - 距离增量 (可加入约束违反惩罚)
        # done = 所有客户已访问
        # 返回更新后的状态
        pass

    def _get_mask(self, td: Dict[str, torch.Tensor]) -> torch.Tensor:
        """生成动作掩码 (True 表示该动作可用)"""
        # 实现电量掩码、时间窗掩码、容量掩码、访问掩码
        pass

    def _get_reward(self, td: Dict[str, torch.Tensor], actions: torch.LongTensor) -> torch.Tensor:
        """计算总奖励 (负的总距离)"""
        return -td["total_distance"]  # 或其他