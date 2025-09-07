"""
Multi-Period Inventory Management Problem (IMP)

This file is the reimplementation of the IMP from the OR-Gym library, an open-source project
developed to bring reinforcement learning to the operations research community.
"""

from typing import Callable
import copy
import os

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from config import env_configs, get_env_configs
from utils import visualize_state, parse_stage_agent_id, clear_dir
from data_simulation import generate_sup_dem_relations
from sc_graph import create_agent_profiles, SupplyChain_Graph

np.random.seed(0)


class InventoryManagementEnv(MultiAgentEnv):
    """
    Multi-period, multi-echelon production-inventory environment (single product, discrete quantities,
    with lead times, holding/backlog costs).

    Event order each period (at start of period):
    1) Receive deliveries (based on lead times).
    2) Place replenishment orders; retail demand realizes at the retailer.
    3) Ship as much as possible to satisfy downstream demand/orders; unmet portion becomes backlog
       (retail backlogs have priority next period).
    4) Compute profit (sales revenue − inbound order cost − backlog penalty − holding cost) and add to assets.
    """

    def __init__(
        self,
        num_stages: int,
        num_agents_per_stage: int,
        num_periods: int,
        init_inventories: list,
        lead_times: list,
        demand_dist: str,
        demand_fn: Callable,
        prod_capacities: list,
        sale_prices: np.array,
        order_costs: np.array,
        prod_costs: np.array,
        backlog_costs: np.array,
        holding_costs: np.array,
        state_format: str,
        supply_relations: dict,
        demand_relations: dict,
        stage_names: list,
        sc_graph: SupplyChain_Graph,
        agent_profiles: list,
        enable_graph_change: bool,
        enable_price_change: bool,
        emergent_events: dict,
        shut_seq: dict,
        rec_seq: dict,
        llm_agents: list = None,
        init_assets: np.array = None,
        init_seed: int = 0,
    ):
        super().__init__()

        # ---- sanity checks ----
        assert num_stages >= 2, "The number of stages should be at least 2."
        assert num_periods >= 1, "The number of periods should be at least 1."
        assert len(init_inventories) == num_stages * num_agents_per_stage, \
            "init_inventories size must be num_stages * num_agents_per_stage."
        assert min(init_inventories) >= 0, "Initial inventories must be non-negative."
        assert np.prod(np.array(lead_times).shape) == num_stages * num_agents_per_stage * num_agents_per_stage, \
            "lead_times size must be num_stages * agents_per_stage * agents_per_stage."
        assert np.min(lead_times) >= 0, "Lead times must be non-negative."
        assert len(prod_capacities) == num_stages * num_agents_per_stage
        assert np.min(prod_capacities) > 0
        assert len(sale_prices) == num_stages * num_agents_per_stage
        assert np.min(sale_prices) >= 0
        assert len(order_costs) == num_stages * num_agents_per_stage
        assert np.min(order_costs) >= 0
        assert len(backlog_costs) == num_stages * num_agents_per_stage
        assert np.min(backlog_costs) >= 0
        assert len(holding_costs) == num_stages * num_agents_per_stage
        assert np.min(holding_costs) >= 0
        assert len(supply_relations) == num_stages and len(supply_relations[0]) == num_agents_per_stage
        assert len(demand_relations) == num_stages and len(demand_relations[0]) == num_agents_per_stage
        assert len(stage_names) == num_stages

        # ---- config attributes ----
        self.num_stages = num_stages
        self.num_agents_per_stage = num_agents_per_stage
        self.num_periods = num_periods
        self.stage_names = stage_names
        self.demand_dist = demand_dist

        self.init_inventories = np.array(init_inventories, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.lead_times = np.array(lead_times, dtype=int).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)
        self.max_lead_time = int(np.max(self.lead_times))
        self.lead_time = self.lead_times

        self.demand_fn = demand_fn
        self.prod_capacities = np.array(prod_capacities, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.max_production = int(np.max(self.prod_capacities))

        self.init_sale_prices = np.array(sale_prices, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.order_costs = np.array(order_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.prod_costs = np.array(prod_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.backlog_costs = np.array(backlog_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.holding_costs = np.array(holding_costs, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)

        self.init_supply_relations = supply_relations
        self.init_demand_relations = demand_relations
        self.llm_agent_set = llm_agents
        self.state_format = state_format
        self.agent_profiles = agent_profiles
        self.sc_graph = sc_graph
        self.enable_graph_change = enable_graph_change
        self.enable_price_change = enable_price_change
        self.emergent_events = emergent_events
        self.shut_seq = shut_seq
        self.rec_seq = rec_seq

        self.init_assets = np.array(init_assets, dtype=float).reshape(self.num_stages, self.num_agents_per_stage)
        self.assets = self.init_assets.copy()

        # ---- runtime tensors ----
        self.period = 0
        self.inventories = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.orders = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.arriving_orders = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.sales = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.backlogs = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.demands = np.zeros(self.num_periods + 1, dtype=int)
        self.profits = np.zeros((self.num_stages, self.num_agents_per_stage, self.num_periods + 1), dtype=int)
        self.total_profits = np.zeros(self.num_periods + 1, dtype=int)
        self.running_agents = np.ones((self.num_stages, self.num_agents_per_stage))
        self.shutdown_agents_set = set()
        self.bankruptcy_log = []

        # ---- action spaces (placeholders; actions are order vectors in practice) ----
        max_order = self.max_production
        max_coop = np.array([1 for _ in range(self.num_agents_per_stage)])
        self.action_order_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.Discrete(n=max_order + 1, start=0, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(self.num_agents_per_stage)
        })
        self.action_supply_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(self.num_agents_per_stage)
        })
        self.action_demand_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(self.num_agents_per_stage)
        })

        self.state_dict = {f"stage_{m}_agent_{x}": None for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

    # ---------------------------- core loop ----------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.period = 0
        self.inventories.fill(0)
        self.orders.fill(0)
        self.arriving_orders.fill(0)
        self.sales.fill(0)
        self.backlogs.fill(0)
        self.demands.fill(0)
        self.profits.fill(0)
        self.total_profits.fill(0)
        self.running_agents = np.ones((self.num_stages, self.num_agents_per_stage))
        self.shutdown_agents_set = set()
        self.assets = self.init_assets.copy()

        self.inventories[:, :, 0] = self.init_inventories
        self.supply_relations = copy.deepcopy(self.init_supply_relations)
        self.demand_relations = copy.deepcopy(self.init_demand_relations)
        self.sale_prices = copy.deepcopy(self.init_sale_prices)
        self.sc_graph.reset_G()
        self.update_state()

        return self.state_dict, {}

    def step(self, order_dict: dict, sup_dict: dict, dem_dict: dict, price_dict: dict):
        """
        Keyword parameters (match mas_model.run_simulation):
        - order_dict: { "stage_i_agent_j": np.array(len=num_agents_per_stage) }
        - sup_dict:   same structure (used when enable_graph_change=True)
        - dem_dict:   reserved
        - price_dict: { "stage_i_agent_j": price } (used when enable_price_change=True)
        """
        # minimal checks
        assert np.all(f"stage_{m}_agent_{x}" in order_dict for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities for all stages are required."
        assert np.all(np.asarray(order_dict[f"stage_{m}_agent_{x}"]).min() >= 0 for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities must be non-negative."

        # advance one period
        self.period += 1
        t = self.period
        M = self.num_stages

        # start-of-period inventory
        current_inventories = self.inventories[:, :, t - 1].copy()

        # optional: update supply relations
        if self.enable_graph_change:
            self.supply_relations = np.stack(
                [sup_dict[f"stage_{m}_agent_{x}"] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]
            ).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)

        # record orders (masked by current supply relations)
        self.orders[:, :, :, t] = np.stack([
            np.asarray(order_dict[f"stage_{m}_agent_{x}"], dtype=int) * self.supply_relations[m][x]
            for m in range(self.num_stages) for x in range(self.num_agents_per_stage)
        ]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)

        # retail demand for this period
        self.demands[t] = int(self.demand_fn(t))

        # add arrivals to inventory (based on lead times)
        for m in range(self.num_stages):
            for i in range(self.num_agents_per_stage):
                for j in range(self.num_agents_per_stage):
                    lt = int(self.lead_times[m][i][j])
                    if lt <= 0:
                        # zero lead time: arrival is handled by arriving_orders at t itself below
                        continue
                    if t >= lt:
                        current_inventories[m][i] += self.arriving_orders[m, i, j, t - lt]

        # total requested orders each supplier receives this period
        cum_req_orders = np.sum(self.orders[:, :, :, t], axis=1)  # shape: (stage, agent)

        # feasible fulfilled orders (bounded by capacity & on-hand inventory); top stage fully fulfills
        fulfilled_orders = np.zeros((self.num_stages, self.num_agents_per_stage), dtype=int)
        fulfilled_orders[:-1] = np.minimum(
            np.minimum(self.backlogs[1:, :, t - 1] + cum_req_orders[:-1], current_inventories[1:]),
            self.prod_capacities[1:]
        )
        # top stage: assumed can fulfill all
        fulfilled_orders[M - 1] = cum_req_orders[M - 1]

        # distribute fulfillment to downstream edges proportionally
        fulfilled_rates = (fulfilled_orders + 1e-10) / (cum_req_orders + 1e-10)
        fulfilled_rates = np.repeat(fulfilled_rates[:, np.newaxis, :], self.num_agents_per_stage, axis=1)
        self.arriving_orders[:, :, :, t] = (self.orders[:, :, :, t] * fulfilled_rates).astype(int)

        # sales: for stages > 0 sales equal downstream arrivals; retailer bounded by demand/inventory/capacity
        # === 分清“发货(出库)”与“到货(销售/成本确认)” ===
        M, A = self.num_stages, self.num_agents_per_stage

        # 1) 本期发货量（dispatch）：用于库存与 backlog 结转（上游 s>=1 才对下游发货）
        dispatched_out = np.zeros((M, A), dtype=int)
        for s in range(1, M):
            for a in range(A):
                # 本期 t 从 (s,a) 发往下游阶段 s-1 的所有量（按你上面比例分配后的整数结果）
                dispatched_out[s, a] = int(self.arriving_orders[s - 1, :, a, t].sum())

        # 2) 上游销售（收入口径）：按“下游本期收到 = 上游在 t-lt 时发出的货”
        sales_up = np.zeros((M, A), dtype=int)
        for s in range(1, M):
            for a in range(A):
                total_recv = 0
                for i in range(A):
                    lt = int(self.lead_times[s - 1][i][a])
                    idx = t if lt <= 0 else t - lt
                    if idx >= 0:
                        total_recv += int(self.arriving_orders[s - 1, i, a, idx])
                sales_up[s, a] = total_recv

        # 3) 销售写回：零售商按需卖，上游按“到货”确认收入
        self.sales[0, :, t] = np.minimum(
            np.minimum(self.backlogs[0, :, t - 1] + self.demands[t], current_inventories[0]),
            self.prod_capacities[0]
        )
        self.sales[1:, :, t] = sales_up[1:, :]

        # 4) backlog 结转：用“本期发货量（dispatch）”冲减（不是 sales）
        #    s=0：外部需求 - 当期销售
        self.backlogs[0, :, t] = np.maximum(
            0,
            self.backlogs[0, :, t - 1] + self.demands[t] - self.sales[0, :, t]
        )
        #    s>=1：下游本期请求 - 我本期实际发走
        #    cum_req_orders[s] 是本期下游对阶段 s 的总请求
        self.backlogs[1:, :, t] = np.maximum(
            0,
            self.backlogs[1:, :, t - 1] + cum_req_orders[1:] - dispatched_out[1:, :]
        )

        # 5) 期末库存：零售商扣“当期销售”，上游扣“本期发货”
        inv_end = current_inventories.copy()
        inv_end[0] -= self.sales[0, :, t]
        for s in range(1, M):
            inv_end[s, :] -= dispatched_out[s, :]
        # 数值保护（如仍出现负值，说明上游放货超库存/产能，需回查上游 fulfilled 分配与取整）
        self.inventories[:, :, t] = np.maximum(0, inv_end)

        # 6) 采购成本（买方收到时确认）：按“本期到货 = 上游在 t-lt 发的货”
        order_costs_now = np.zeros((M, A), dtype=int)
        for m in range(M):  # 买方阶段（下游索引）
            for i in range(A):  # 买方 agent
                total_cost = 0
                for j in range(A):  # 对应上游供应商
                    lt = int(self.lead_times[m][i][j])
                    idx = t if lt <= 0 else t - lt
                    if idx >= 0:
                        qty = int(self.arriving_orders[m, i, j, idx])
                        total_cost += qty * int(self.order_costs[m][j])
                order_costs_now[m, i] = total_cost

        # 7) 利润
        self.profits[:, :, t] = (
                self.sale_prices * self.sales[:, :, t]
                - order_costs_now
                - self.backlog_costs * self.backlogs[:, :, t]
                - self.holding_costs * self.inventories[:, :, t]
        )
        self.total_profits[t] = int(np.sum(self.profits[:, :, t]))

        # assets update
        self.assets += self.profits[:, :, t]

        # bankruptcy: if an active agent's assets ≤ 0, shut it down immediately (assets clamped to 0)
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                if self.running_agents[m][x] == 1 and self.assets[m][x] <= 0:
                    print(f"[Bankruptcy] t={t}: stage_{m}_agent_{x} shuts down (assets={self.assets[m][x]})")
                    self.bankruptcy_log.append((t, m, x, float(self.assets[m][x])))
                    self.assets[m][x] = 0.0
                    self.create_shutdown_event(m, x, self.state_dict)

        # optional dynamic pricing: update prices and propagate upstream->downstream order costs
        if self.enable_price_change:
            for m in range(self.num_stages):
                for x in range(self.num_agents_per_stage):
                    self.sale_prices[m][x] = price_dict.get(f"stage_{m}_agent_{x}", self.sale_prices[m][x])
            self.order_costs[:M - 1, :] = self.sale_prices[1:, :]

        # RL-style outputs
        rewards = {f"stage_{m}_agent_{x}": int(self.profits[m, x, t]) for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        all_termination = self.period >= self.num_periods
        terminations = {f"stage_{m}_agent_{x}": all_termination for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        terminations["__all__"] = all_termination
        truncations = {f"stage_{m}_agent_{x}": False for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        truncations["__all__"] = False
        infos = {f"stage_{m}_agent_{x}": {} for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

        # early termination if any stage becomes fully closed
        stage_all_closed = np.all(self.running_agents == 0, axis=1)
        if stage_all_closed.any():
            dead_stage = int(np.where(stage_all_closed)[0][0])
            print(f"[Terminate] All agents in stage {dead_stage} are closed at t={t}. Ending episode.")
            terminations = {f"stage_{m}_agent_{x}": True for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
            terminations["__all__"] = True
            truncations = {f"stage_{m}_agent_{x}": False for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
            truncations["__all__"] = False
            infos = {f"stage_{m}_agent_{x}": {"terminated_stage": dead_stage} for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

        # update human-readable state
        self.update_state()
        return self.state_dict, rewards, terminations, truncations, infos

    # ---------------------------- state helpers ----------------------------

    def update_state(self) -> None:
        t = self.period
        states = dict()
        states["prod_capacities"] = self.prod_capacities
        states["sale_prices"] = self.sale_prices
        states["order_costs"] = self.order_costs
        states["backlog_costs"] = self.backlog_costs
        states["holding_costs"] = self.holding_costs
        states["prod_costs"] = self.prod_costs
        states["lead_times"] = self.lead_times
        states["inventories"] = self.inventories[:, :, t]
        states["backlogs"] = self.backlogs[:, :, t]
        states["upstream_backlogs"] = np.zeros((self.num_stages, self.num_agents_per_stage), dtype=int)
        states["upstream_backlogs"][:-1] = self.backlogs[1:, :, t]
        states["suppliers"] = self.supply_relations
        states["customers"] = self.demand_relations

        lt_max = self.max_lead_time
        states["recent_sales"] = np.zeros((self.num_stages, self.num_agents_per_stage, lt_max), dtype=int)
        if lt_max > 0:
            if t >= lt_max:
                states["recent_sales"] = self.sales[:, :, (t - lt_max + 1):(t + 1)]
            elif t > 0:
                states["recent_sales"][:, :, -t:] = self.sales[:, :, 1:(t + 1)]

        states["arriving_deliveries"] = np.zeros(
            (self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, lt_max), dtype=int
        )
        if lt_max > 0:
            for m in range(self.num_stages):
                for x in range(self.num_agents_per_stage):
                    for j in range(self.num_agents_per_stage):
                        lt = int(self.lead_times[m][x][j])
                        if lt <= 0:
                            continue
                        if t >= lt:
                            # align a window of last lt arrivals
                            states["arriving_deliveries"][m, x, j, -min(lt, lt_max):] = \
                                self.arriving_orders[m, x, j, (t - lt + 1):(t + 1)]
                        elif t > 0:
                            states["arriving_deliveries"][m, x, j, -t:] = self.arriving_orders[m, x, j, 1:(t + 1)]

        self.state_dict = {}
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                # NOTE: put the entire row vector of order_costs for this stage to match the upper-level prompt usage
                agent_state = [
                    states["prod_capacities"][m][x],
                    states["sale_prices"][m][x],
                    states["order_costs"][m],
                    states["backlog_costs"][m][x],
                    states["holding_costs"][m][x],
                    states["lead_times"][m][x],
                    states["inventories"][m][x],
                    states["backlogs"][m][x],
                    states["upstream_backlogs"][m][x],
                    states["suppliers"][m][x],
                    states["customers"][m][x],
                    states["recent_sales"][m][x],
                    states["arriving_deliveries"][m][x],
                    states["prod_costs"][m][x],
                    self.assets[m][x],
                ]
                self.state_dict[f"stage_{m}_agent_{x}"] = agent_state

        return self.state_dict

    def _parse_state(self, state: list) -> dict:
        return {
            'prod_capacity': state[0],
            'sale_price': state[1],
            'order_costs': state[2],
            'backlog_cost': state[3],
            'holding_cost': state[4],
            'lead_times': state[5],
            'inventory': state[6],
            'backlog': state[7],
            'upstream_backlog': state[8],
            "suppliers": state[9],
            "customers": state[10],
            'sales': np.asarray(state[11]).tolist(),
            'deliveries': np.asarray(state[12]).tolist(),
            'prod_cost': state[13],
            'assets': state[14],
        }

    def parse_state(self, state_dict: dict = None) -> dict:
        if state_dict is None:
            state_dict = self.state_dict
        parsed_state = {}
        for stage_agent_id_name, state in state_dict.items():
            parsed_state[stage_agent_id_name] = self._parse_state(state)
        return parsed_state

    # ---------------------------- utilities ----------------------------

    def no_backlog_env_proxy(self, stage_id: int, agent_id: int,
                             action_order_dict: dict, action_sup_dict: dict, action_price_dict: dict):
        # keep (or restore) supply links
        sup_action = self.supply_relations[stage_id][agent_id]
        if sum(sup_action) == 0:
            sup_action = self.init_supply_relations[stage_id][agent_id]
        action_sup_dict[f'stage_{stage_id}_agent_{agent_id}'] = sup_action

        # naive order split
        demand = self.demand_fn(self.period)
        if stage_id == 0:
            stage_order_action = ((demand * sup_action) / max(1, sum(sup_action))).astype(int)
        elif stage_id == self.num_stages - 1:
            avg_order = np.mean([np.sum(action_order_dict[x]) for x in action_order_dict if f"stage_{stage_id - 1}" in x])
            stage_order_action = sup_action * int(avg_order)
        else:
            avg_order = np.mean([np.sum(action_order_dict[x]) for x in action_order_dict if f"stage_{stage_id - 1}" in x]) / max(1, sum(sup_action))
            stage_order_action = sup_action * int(avg_order)

        action_order_dict[f'stage_{stage_id}_agent_{agent_id}'] = stage_order_action

        price = self.init_sale_prices[stage_id][agent_id]
        action_price_dict[f"stage_{stage_id}_agent_{agent_id}"] = price

        return action_sup_dict, action_order_dict, action_price_dict

    def get_all_shutdown_agents(self):
        print("The closed agents are", ", ".join(self.shutdown_agents_set))

    def create_shutdown_event(self, stage_id: int, agent_id: int, state_dict: dict):
        print(f"Shutdown stage_{stage_id}_agent_{agent_id}.")
        self.running_agents[stage_id][agent_id] = 0
        # remove this supplier from downstream customers' supplier lists
        if stage_id != 0:
            for down_agent_id in range(self.num_agents_per_stage):
                self.supply_relations[stage_id - 1][down_agent_id][agent_id] = 0
        self.shutdown_agents_set.add(f"stage_{stage_id}_agent_{agent_id}")
        # clear its ending inventory (backlog persists and is charged)
        self.inventories[stage_id, agent_id, self.period] = 0
        return state_dict

    def create_recovery_event(self, stage_id: int, agent_id: int):
        print(f"Re-open stage_{stage_id}_agent_{agent_id}.")
        self.running_agents[stage_id][agent_id] = 1
        self.shutdown_agents_set.discard(f"stage_{stage_id}_agent_{agent_id}")
        if stage_id != 0:
            for down_agent_id in range(self.num_agents_per_stage):
                self.supply_relations[stage_id - 1][down_agent_id][agent_id] = \
                    self.init_supply_relations[stage_id - 1][down_agent_id][agent_id]
    def create_demand_surge(self):
        if self.demand_fn.dist == "constant_demand":
            self.demand_fn.mean *= 2
        elif self.demand_fn.dist == "uniform_demand":
            self.demand_fn.lb *= 2
            self.demand_fn.ub *= 2
        elif self.demand_fn.dist == "normal_demand":
            self.demand_fn.mean *= 2
        elif "poisson" in self.demand_fn.dist:
            self.demand_fn.mean *= 2


def env_creator(env_config):
    agent_profiles = create_agent_profiles(env_config=env_config)
    sc_graph = SupplyChain_Graph(agent_profiles=agent_profiles,
                                 num_stages=env_config["num_stages"],
                                 num_agents_per_stage=env_config["num_agents_per_stage"])

    return InventoryManagementEnv(
        num_stages=env_config['num_stages'],
        num_agents_per_stage=env_config['num_agents_per_stage'],
        num_periods=env_config['num_periods'],
        init_inventories=env_config['init_inventories'],
        lead_times=env_config['lead_times'],
        demand_dist=env_config['demand_dist'],
        demand_fn=env_config['demand_fn'],
        prod_capacities=env_config['prod_capacities'],
        sale_prices=env_config['sale_prices'],
        order_costs=env_config['order_costs'],
        prod_costs=env_config['prod_costs'],
        backlog_costs=env_config['backlog_costs'],
        holding_costs=env_config['holding_costs'],
        supply_relations=env_config['supply_relations'],
        demand_relations=env_config['demand_relations'],
        stage_names=env_config['stage_names'],
        llm_agents=env_config['llm_agents'],
        state_format=env_config['state_format'],
        enable_graph_change=env_config["enable_graph_change"],
        enable_price_change=env_config["enable_price_change"],
        emergent_events=env_config["emergent_events"],
        shut_seq=env_config["shut_seq"],
        rec_seq=env_config["rec_seq"],
        init_assets=env_config['init_assets'],
        agent_profiles=agent_profiles,
        sc_graph=sc_graph,
    )

