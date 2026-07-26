"""
POMO_Test_2.py - POMO training for EVRP-TW with hard constraints.

Self-contained implementation:
  - EVRP_TWGenerator: random instance generator (coordinates, demand, time windows, stations)
  - EVRP_TWEnv:       RL4CO environment with hard constraints (capacity, battery, time windows)
  - train():          POMO training entry point via RL4CO Trainer

Node indexing:
  0                          = depot
  1 .. num_stations           = charging stations
  num_stations+1 .. total-1   = customers

Constraints (hard, enforced via action mask):
  - Each customer visited at most once
  - Remaining capacity >= demand of selected customer
  - Remaining battery >= distance to node (can reach)
  - Arrival time <= tw_late (early arrival allowed, waits)

Reward: -total_tour_length  (full route including return to depot)
"""

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import Generator
from rl4co.models import POMO
from rl4co.utils import RL4COTrainer
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# =============================================================================
# 1.  Data generator
# =============================================================================

class EVRP_TWGenerator(Generator):
    """Random instance generator for EVRP-TW.

    Each instance contains:
        locs          [batch, total_nodes, 2]  -- coordinates (depot, stations, customers)
        demand        [batch, total_nodes]      -- 0 for depot & stations
        tw_early      [batch, total_nodes]      -- 0 for depot & stations
        tw_late       [batch, total_nodes]      -- 0 for depot & stations
        service_time  [batch, total_nodes]      -- 0 for depot & stations
        battery_capacity  [batch, 1]
        consumption_rate  [batch, 1]
        vehicle_capacity  [batch, 1]
        speed             [batch, 1]

    Args:
        num_loc:         Number of customer nodes.
        num_stations:    Number of charging stations.
        battery_capacity:   Max battery energy (kWh / abstract units).
        consumption_rate:   Energy consumed per unit distance.
        vehicle_capacity:   Max load capacity.
        speed:              Distance per time unit (for time-window checks).
        depot_loc:          Fixed depot coordinates (x, y) in [0,1].
    """

    def __init__(
        self,
        num_loc: int = 5,
        num_stations: int = 2,
        battery_capacity: float = 3.0,
        consumption_rate: float = 1.0,
        vehicle_capacity: float = 30.0,
        speed: float = 1.0,
        depot_loc: tuple = (0.5, 0.5),
        min_demand: int = 1,
        max_demand: int = 10,
        tw_horizon: float = 5.0,
        min_tw_width: float = 5.0,
        max_tw_width: float = 10.0,
        min_service: float = 1.0,
        max_service: float = 3.0,
    ):
        # num_loc is used by POMO for decoding steps - set it to total_nodes
        self.num_customers = num_loc  # actual customer count
        self.num_loc = num_loc        # temporarily, will be overridden below
        self.num_stations = num_stations
        self.battery_capacity = battery_capacity
        self.consumption_rate = consumption_rate
        self.vehicle_capacity = vehicle_capacity
        self.speed = speed
        self.depot_loc = depot_loc
        self.total_nodes = 1 + num_stations + num_loc  # depot + stations + customers
        # POMO uses generator.num_loc as decoding steps; give it total_nodes
        self.num_loc = self.total_nodes

        self.min_demand = min_demand
        self.max_demand = max_demand
        self.tw_horizon = tw_horizon
        self.min_tw_width = min_tw_width
        self.max_tw_width = max_tw_width
        self.min_service = min_service
        self.max_service = max_service

        # Convenience slices (used by the environment too)
        self.depot_slice = slice(0, 1)
        self.station_slice = slice(1, 1 + num_stations)
        self.customer_slice = slice(1 + num_stations, self.total_nodes)

    def _generate(self, batch_size) -> TensorDict:
        device = "cpu"

        # --- locations ---
        depot = torch.tensor(self.depot_loc, dtype=torch.float32).view(1, 2)
        depot = depot.unsqueeze(0).expand(*batch_size, -1, -1)  # [B, 1, 2]

        stations = torch.rand(*batch_size, self.num_stations, 2, dtype=torch.float32)
        customers = torch.rand(*batch_size, self.num_customers, 2, dtype=torch.float32)

        locs = torch.cat([depot, stations, customers], dim=-2)  # [B, T, 2]

        # --- demand (0 for depot & stations) ---
        raw_demand = torch.randint(
            self.min_demand, self.max_demand + 1,
            (*batch_size, self.num_customers), dtype=torch.float32,
        )
        demand = torch.zeros(*batch_size, self.total_nodes - 1, dtype=torch.float32)
        demand[..., self.num_stations:] = raw_demand

        # --- time windows (0 for depot & stations) ---
        tw_early = torch.zeros(*batch_size, self.total_nodes, dtype=torch.float32)
        tw_late = torch.zeros(*batch_size, self.total_nodes, dtype=torch.float32)

        tw_start = torch.rand(*batch_size, self.num_customers, dtype=torch.float32) * self.tw_horizon
        tw_width = torch.rand(*batch_size, self.num_customers, dtype=torch.float32) * (
            self.max_tw_width - self.min_tw_width
        ) + self.min_tw_width
        tw_early[..., self.customer_slice] = tw_start
        tw_late[..., self.customer_slice] = tw_start + tw_width

        # --- service time ---
        service_time = torch.zeros(*batch_size, self.total_nodes, dtype=torch.float32)
        raw_service = torch.rand(*batch_size, self.num_customers, dtype=torch.float32) * (
            self.max_service - self.min_service
        ) + self.min_service
        service_time[..., self.customer_slice] = raw_service

        # --- scalar parameters ---
        battery_capacity = torch.full((*batch_size, 1), self.battery_capacity, dtype=torch.float32)
        consumption_rate = torch.full((*batch_size, 1), self.consumption_rate, dtype=torch.float32)
        vehicle_capacity = torch.full((*batch_size, 1), self.vehicle_capacity, dtype=torch.float32)
        speed = torch.full((*batch_size, 1), self.speed, dtype=torch.float32)
        time_windows = torch.stack([tw_early, tw_late], dim=-1)  # [B, total, 2]

        return TensorDict(
            {
                "locs": locs,
                "demand": demand,
                "tw_early": tw_early,
                "tw_late": tw_late,
                "time_windows": time_windows,
                "service_time": service_time,
                "durations": service_time,
                "battery_capacity": battery_capacity,
                "consumption_rate": consumption_rate,
                "vehicle_capacity": vehicle_capacity,
                "speed": speed,
            },
            batch_size=batch_size,
        )


# =============================================================================
# 2.  EVRP-TW Environment
# =============================================================================

class EVRP_TWEnv(RL4COEnvBase):
    """EVRP-TW environment with hard constraints.

    At each step the agent selects a node (depot, station, or customer).
    Constraints are enforced via ``get_action_mask``:
      - already visited (except depot is always available)
      - capacity exceeded
      - battery too low to reach the node
      - time-window violation (arrival after tw_late)
    """

    name = "cvrptw"

    def __init__(
        self,
        generator: EVRP_TWGenerator = None,
        generator_params: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator_params = generator_params or {}
            generator = EVRP_TWGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    # ---- helpers ----

    @staticmethod
    def _node_distance(locs, i, j):
        """Pairwise distance between nodes i and j.  i, j: [B, 1] or [B]."""
        if i.dim() == 1:
            i = i[:, None]
        if j.dim() == 1:
            j = j[:, None]
        loc_i = gather_by_index(locs, i, squeeze=False)  # [B, 1, 2]
        loc_j = gather_by_index(locs, j, squeeze=False)
        return (loc_i - loc_j).norm(p=2, dim=-1)  # [B, 1]

    # ---- specs ----

    def _make_spec(self, generator: EVRP_TWGenerator):
        total = generator.total_nodes
        self.observation_spec = Composite(
            locs=Bounded(
                low=0.0, high=1.0,
                shape=(total, 2), dtype=torch.float32,
            ),
            current_node=Unbounded(shape=(1,), dtype=torch.int64),
            demand=Unbounded(shape=(total,), dtype=torch.float32),
            tw_early=Unbounded(shape=(total,), dtype=torch.float32),
            tw_late=Unbounded(shape=(total,), dtype=torch.float32),
            service_time=Unbounded(shape=(total,), dtype=torch.float32),
            remaining_battery=Unbounded(shape=(1,), dtype=torch.float32),
            remaining_capacity=Unbounded(shape=(1,), dtype=torch.float32),
            current_time=Unbounded(shape=(1,), dtype=torch.float32),
            total_distance=Unbounded(shape=(1,), dtype=torch.float32),
            visited=Unbounded(shape=(total,), dtype=torch.uint8),
            action_mask=Unbounded(shape=(total,), dtype=torch.bool),
            battery_capacity=Unbounded(shape=(1,), dtype=torch.float32),
            consumption_rate=Unbounded(shape=(1,), dtype=torch.float32),
            vehicle_capacity=Unbounded(shape=(1,), dtype=torch.float32),
            speed=Unbounded(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,), dtype=torch.int64,
            low=0, high=total - 1,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

        # ---- multi-start (POMO) ----

    def get_num_starts(self, td):
        return self.generator.num_customers

    def select_start_nodes(self, td, num_starts):
        n_sta = self.generator.num_stations
        n_cust = self.generator.num_loc - 1 - n_sta
        max_starts = min(num_starts, n_cust)
        cust_start = 1 + n_sta
        import torch
        customers = torch.arange(cust_start, cust_start + n_cust, device=td.device)
        selected = customers[:max_starts].repeat_interleave(td.shape[0])
        return selected

# ---- reset ----

    def _reset(self, td, batch_size):
        device = td.device
        total = self.generator.total_nodes

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],  # [B, total_nodes-1] for VRPTW embedding
                "demand_full": torch.cat([torch.zeros(*batch_size, 1, dtype=torch.float32, device=device), td["demand"]], dim=-1),  # [B, total_nodes] for mask/step,
                "tw_early": td["tw_early"],
                "tw_late": td["tw_late"],
                "service_time": td["service_time"],
                "battery_capacity": td["battery_capacity"],
                "consumption_rate": td["consumption_rate"],
                "vehicle_capacity": td["vehicle_capacity"],
                "speed": td["speed"],
                # start at depot
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "remaining_battery": td["battery_capacity"].clone(),
                "remaining_capacity": td["vehicle_capacity"].clone(),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "total_distance": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "visited": torch.zeros(
                    *batch_size, total, dtype=torch.uint8, device=device
                ),
            },
            batch_size=batch_size,
        )
        td_reset["used_capacity"] = td_reset["vehicle_capacity"] - td_reset["remaining_capacity"]
        td_reset["time_windows"] = td["time_windows"]
        td_reset["durations"] = td["durations"]
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    # ---- step ----

    def _step(self, td):
        action = td["action"]  # [B]  or [B, 1]
        if action.dim() == 1:
            action = action[:, None]  # [B, 1]

        G = self.generator
        n_sta = G.num_stations
        cust_slice = G.customer_slice

        # --- distance travelled ---
        dist = self._node_distance(td["locs"], td["current_node"], action)  # [B, 1]

        # --- total distance ---
        total_distance = td["total_distance"] + dist

        # --- time ---
        travel_time = dist / td["speed"]
        arrival = td["current_time"] + travel_time
        # wait if early
        tw_early_a = gather_by_index(td['tw_early'], action, squeeze=False)  # [B, 1]
        service_a = gather_by_index(td['service_time'], action, squeeze=False)
        current_time = torch.max(arrival, tw_early_a) + service_a

        # --- battery ---
        battery_consumed = dist * td["consumption_rate"]
        remaining_battery = td["remaining_battery"] - battery_consumed
        # reset at depot or station
        is_charge = (action == 0) | ((action >= 1) & (action < 1 + n_sta))
        remaining_battery = torch.where(
            is_charge, td["battery_capacity"], remaining_battery
        )

        # --- capacity ---
        demand_a = gather_by_index(td['demand_full'], action, squeeze=False)
        remaining_capacity = td["remaining_capacity"] - demand_a
        is_depot = action == 0
        remaining_capacity = torch.where(
            is_depot, td["vehicle_capacity"], remaining_capacity
        )

        # --- visited ---
        visited = td["visited"].scatter(-1, action, 1)

        # --- done: all customers visited ---
        n_cust = G.num_loc
        if n_cust > 0:
            cust_visited = visited[..., cust_slice]  # [B, n_cust]
            done = cust_visited.all(dim=-1).bool()   # [B]
        else:
            done = torch.ones(*visited.shape[:-1], dtype=torch.bool, device=visited.device)

        # reward is 0 at each step; actual reward computed in _get_reward
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "current_node": action,
                "total_distance": total_distance,
                "current_time": current_time,
                "remaining_battery": remaining_battery,
                "remaining_capacity": remaining_capacity,
                "visited": visited,
                "reward": reward,
                "done": done,
            }
        )
        td["used_capacity"] = td["vehicle_capacity"] - td["remaining_capacity"]
        td.set("action_mask", self.get_action_mask(td))
        return td

    # ---- action mask ----

    def get_action_mask(self, td):
        G = self.generator
        n_sta = G.num_stations
        total = G.total_nodes
        cust_slice = G.customer_slice

        # 1) already visited  (depot is *never* masked by visited alone)
        visited = td["visited"].bool()  # [B, total]

        # 2) capacity
        demand = td["demand_full"]  # [B, total]
        rem_cap = td["remaining_capacity"]  # [B, 1]
        exceeds_cap = demand > rem_cap  # [B, total]

        # 3) battery -- can we reach the node?
        locs = td["locs"]
        curr = td["current_node"]  # [B, 1]

        # distance from current to every node
        loc_curr = gather_by_index(locs, curr, squeeze=False)  # [B, 1, 2]
        loc_all = locs  # [B, total, 2]
        d_curr_to_all = (loc_all - loc_curr).norm(p=2, dim=-1)  # [B, total]

        battery_needed = d_curr_to_all * td["consumption_rate"]  # [B, total]
        no_battery = battery_needed > td["remaining_battery"] + 1e-6  # [B, total]

        # 4) time window -- can we arrive before tw_late?
        travel = d_curr_to_all / td["speed"]  # [B, total]
        arrival = td["current_time"] + travel   # [B, total]
        tw_late = td["tw_late"]                 # [B, total]
        violates_tw = arrival > tw_late + 1e-6  # [B, total]

        # 5) depot: not allowed when we are already at depot AND
        #    there are still feasible (unvisited, within constraints) customers
        # --- first compute which customers are feasible ---
        cust_visited = visited[..., cust_slice]

        # A customer is feasible if NOT visited AND NOT exceeds_cap AND NOT no_battery AND NOT violates_tw
        cust_feasible = (
            ~cust_visited
            & ~exceeds_cap[..., cust_slice]
            & ~no_battery[..., cust_slice]
            & ~violates_tw[..., cust_slice]
        )  # [B, n_cust]

        has_feasible_customers = cust_feasible.any(dim=-1, keepdim=True)  # [B, 1]

        # depot is masked if at depot AND there are feasible customers
        at_depot = td["current_node"] == 0  # [B, 1]
        depot_masked = at_depot & has_feasible_customers  # [B, 1]

        # --- combine into final mask ---
        # Start with visited mask (True = cannot visit)
        mask = visited.clone()

        # Depot is never blocked by visited, only by above rule
        mask[:, 0:1] = depot_masked

        # Add constraint violations for customers
        cust_mask = visited[..., cust_slice].clone()
        cust_mask = cust_mask | exceeds_cap[..., cust_slice]
        cust_mask = cust_mask | no_battery[..., cust_slice]
        cust_mask = cust_mask | violates_tw[..., cust_slice]
        mask[..., cust_slice] = cust_mask

        # Stations: only battery matters
        if n_sta > 0:
            sta_mask = visited[..., 1:1 + n_sta].clone()
            sta_mask = sta_mask | no_battery[..., 1:1 + n_sta]
            mask[..., 1:1 + n_sta] = sta_mask

        # Return True = allowed
        return ~mask

    # ---- reward ----

    def _get_reward(self, td, actions):
        """Compute total tour length including return to depot."""
        # actions: [B, num_steps]
        depot_loc = td["locs"][..., 0:1, :]  # [B, 1, 2]
        ordered = gather_by_index(td['locs'], actions, squeeze=False)  # [B, num_steps, 2]
        full = torch.cat([depot_loc, ordered, depot_loc], dim=1)  # [B, steps+2, 2]
        return -get_tour_length(full)

    # ---- solution validity ----

    def check_solution_validity(self, td, actions):
        """Verify that the solution is feasible."""
        G = self.generator
        n_cust = G.num_loc
        n_sta = G.num_stations
        cust_slice = G.customer_slice

        batch_size = actions.shape[0]
        locs = td["locs"]
        demand = td["demand_full"]
        tw_late = td["tw_late"]
        tw_early = td["tw_early"]
        service = td["service_time"]
        bat_cap = td["battery_capacity"]
        cons_rate = td["consumption_rate"]
        veh_cap = td["vehicle_capacity"]
        spd = td["speed"]

        for b in range(batch_size):
            route = actions[b].tolist()
            # strip trailing depot visits
            while route and route[-1] == 0:
                route.pop()

            pos = 0  # depot
            bat = bat_cap[b].item()
            cap = veh_cap[b].item()
            t = 0.0
            visited_set = set()

            for node in route:
                d = (locs[b, node] - locs[b, pos]).norm(p=2).item()

                if node >= 1 + n_sta:  # customer
                    dem = demand[b, node].item()
                    assert cap >= dem - 1e-5, f"b{b}: capacity exceeded ({cap} < {dem})"
                    cap -= dem

                    arr = t + d / spd[b].item()
                    assert arr <= tw_late[b, node].item() + 1e-4, (
                        f"b{b}: time-window violation at node {node}"
                        f" (arr={arr:.2f} > {tw_late[b, node].item():.2f})"
                    )
                    svc = service[b, node].item()
                    t = max(arr, tw_early[b, node].item()) + svc

                    assert node not in visited_set, f"b{b}: revisit customer {node}"
                    visited_set.add(node)

                elif 1 <= node < 1 + n_sta:  # station
                    arr = t + d / spd[b].item()
                    t = arr  # no service at station
                    bat = bat_cap[b].item()  # full charge

                else:  # depot (node == 0)
                    arr = t + d / spd[b].item()
                    t = arr
                    cap = veh_cap[b].item()
                    bat = bat_cap[b].item()

                # battery
                bat -= d * cons_rate[b].item()
                assert bat >= -1e-4, f"b{b}: battery depleted ({bat:.3f})"
                pos = node

            # all customers visited
            expected = set(range(1 + n_sta, 1 + n_sta + n_cust))
            assert visited_set == expected, (
                f"b{b}: not all customers visited "
                f"(missing {expected - visited_set})"
            )

    # ---- render (optional placeholder) ----

    def render(self, *args, **kwargs):
        """Not implemented."""
        pass



# =============================================================================
# Custom embeddings for EVRP-TW (registered at runtime)
# =============================================================================

import torch.nn as nn
import rl4co.models.nn.env_embeddings.init as _init_mod
import rl4co.models.nn.env_embeddings.context as _ctx_mod
from rl4co.models.nn.env_embeddings.context import EnvContext

class _EVRPTWInitEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=True):
        super().__init__()
        depot_dim = 2; node_dim = 4
        self.init_embed = nn.Linear(node_dim, embed_dim, bias=linear_bias)
        self.init_embed_depot = nn.Linear(depot_dim, embed_dim, bias=linear_bias)

    def forward(self, td):
        depot, cities = td['locs'][..., :1, :], td['locs'][..., 1:, :]
        depot_embed = self.init_embed_depot(depot)
        nodes = torch.cat([cities, td['demand'][..., 1:, None], td['tw_late'][..., 1:, None]], dim=-1)
        return torch.cat([depot_embed, self.init_embed(nodes)], -2)

class _EVRPTWContext(EnvContext):
    def __init__(self, embed_dim):
        super().__init__(embed_dim=embed_dim, step_context_dim=embed_dim + 2)

    def _state_embedding(self, embeddings, td):
        return torch.cat([td['remaining_capacity'], td['current_time']], -1)

_orig_init = _init_mod.env_init_embedding
_orig_ctx = _ctx_mod.env_context_embedding

def _patched_init(name, cfg):
    return _EVRPTWInitEmbedding(**cfg) if name == 'evrptw' else _orig_init(name, cfg)

def _patched_ctx(name, cfg):
    return _EVRPTWContext(**cfg) if name == 'evrptw' else _orig_ctx(name, cfg)

_init_mod.env_init_embedding = _patched_init
_ctx_mod.env_context_embedding = _patched_ctx


# =============================================================================
# 3.  Training entry point
# =============================================================================

def train():
    """Create environment -- POMO model -- RL4CO Trainer and fit."""
    env = EVRP_TWEnv(
        generator_params=dict(
            num_loc=5,
            num_stations=2,
            battery_capacity=10.0,
            consumption_rate=1.0,
            vehicle_capacity=50.0,
            speed=1.0,
            tw_horizon=2.0,
            min_tw_width=30.0,
            max_tw_width=50.0,
            min_service=1.0,
            max_service=2.0,
        ),
        check_solution=False,
    )

    model = POMO(
        env,
        batch_size=64,
        optimizer_kwargs={"lr": 1e-4},
    )

    trainer = RL4COTrainer(
        max_epochs=1,
        accelerator="gpu",
        enable_progress_bar=True,
    )

    trainer.fit(model)


if __name__ == "__main__":
    train()

