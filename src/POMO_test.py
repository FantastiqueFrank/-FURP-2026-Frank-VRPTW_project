from rl4co.models import POMO
from rl4co.utils import RL4COTrainer
from rl4co_envs.evrptw_env import EVRP_TWEnv  # 请根据你的实际模块路径调整

# 创建 EVRP-TW 环境（20个客户，3个充电站）
# 你可以调整这些参数
env = EVRP_TWEnv(
    num_loc=20,              # 客户数量
    num_stations=3,          # 充电站数量
    battery_capacity=80.0,   # 电池容量 (kWh)
    consumption_rate=0.2,    # 每公里耗电 (kWh/km)
    vehicle_capacity=100.0,  # 车辆载重容量
    speed=1.0,               # 速度（距离单位/时间单位）
)

# 创建 POMO 模型
model = POMO(
    env,
    batch_size=64,           # 批次大小（可根据GPU显存调整）
    optimizer_kwargs={"lr": 1e-4},
)

# 训练模型（1个epoch，使用GPU）
trainer = RL4COTrainer(
    max_epochs=1,            # 先跑1个epoch快速测试
    accelerator="gpu",       # 使用GPU
    # 可选：开启进度条
    enable_progress_bar=True,
)

trainer.fit(model)