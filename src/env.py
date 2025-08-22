"""
Multi-Period Inventory Management Problem (IMP)

This file is the reimplementation of the IMP from the OR-Gym library, an open-source project developed to bring
reinforcement learning to the operations research community. OR-Gym is licensed under the MIT License. For more
information, please visit the OR-Gym GitHub repository: https://github.com/hubbs5/or-gym.
"""
from typing import Callable

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from config import env_configs, get_env_configs
from utils import visualize_state, parse_stage_agent_id, clear_dir
from data_simulation import generate_sup_dem_relations
import os
import copy
from sc_graph import create_agent_profiles, SupplyChain_Graph

np.random.seed(0)


class InventoryManagementEnv(MultiAgentEnv):
    """
    Inventory Management Environment

    A multi-period, multi-echelon production-inventory system for a single non-perishable product sold in discrete
    quantities. Each stage in the supply chain consists of an inventory holding area and a production area. The
    inventory holding area stores the materials necessary for production at that stage. One unit of inventory
    produces one unit of product at each stage. There are lead times for transferring products between stages. The
    outgoing material from stage i serves as the feed material for production at stage i-1. Stages are numbered in
    ascending order: Stages = {0, 1, ..., M-1}, with the zeroth stage being the retailer. Production at each stage is
    limited by the stage's production capacity and available inventory.

    At the beginning of each time period, the following sequence of events occurs:

    1) Check deliveries: Each stage receives incoming inventory replenishment shipments that have arrived after the
       stage's respective lead time.
    2) Check orders and demands: Each stage places replenishment orders to their  respective suppliers. Replenishment
       orders are filled according to the available production capacity and inventory at the suppliers. Customer demand
       occurs at the retailer and is filled based on the available  inventory at the retailer.
    3) Deliver orders and demands: Each stage delivers as many products as possible to satisfy  downstream demand or
       replenishment orders. Unfulfilled sales and replenishment orders are backlogged, with backlogged sales taking
       priority in the following period.
    4) Compute profits: Each stage computes the profit and cost for product sales, material orders, backlog penalties,
       and surplus inventory holding costs.
    """

    def __init__(
        self, num_stages: int, num_agents_per_stage: int, num_periods: int, init_inventories: list, lead_times: list, demand_dist: str, demand_fn: Callable,
        prod_capacities: list, sale_prices: np.array, order_costs: np.array, prod_costs: np.array, backlog_costs: np.array, holding_costs: np.array, state_format: str, 
        supply_relations: dict, demand_relations: dict, stage_names: list, sc_graph: SupplyChain_Graph, agent_profiles: list, enable_graph_change: bool, 
        enable_price_change: bool, emergent_events: dict, shut_seq: dict, rec_seq: dict, llm_agents: list=None, init_assets: np.array=None, init_seed: int = 0):
        """
        Initialize the inventory management environment

        :param num_stages: number of stages (M)
        :param num_agents_per_stage: number of agents in each stage (X_m)
        :param num_periods: number of periods (N)
        :param init_inventories: initial inventory quantities (I0)
        :param lead_times: lead times (L)
        :param demand_fn: demand function (D)
        :param prod_capacities: production capacities (c)
        :param sale_prices: unit sale prices (p)
        :param order_costs: unit order costs (r)
        :param backlog_costs: unit backlog costs for unfulfilled orders (k)
        :param holding_costs: unit inventory holding costs (h)
        :param relations: the supply&demand relations of each companies {company: [upstream, downstream]}
        :param stage_names: stage names
        :param init_seed: initial seed
        """
        super().__init__()

        # Check the validity of inputs
        assert num_stages >= 2, "The number of stages should be at least 2."
        assert num_periods >= 1, "The number of periods should be at least 1."
        assert len(init_inventories) == num_stages * num_agents_per_stage, \
            "The number of initial inventories quantities should be the total number of agents in the environment (i.e., number of stages * number of agents per stage)."
        assert min(init_inventories) >= 0, "The initial inventory quantities should be non-negative."
        assert np.prod(lead_times.shape) == num_stages * num_agents_per_stage * num_agents_per_stage, "The number of lead times should be the total number of agent pair in the environment."
        assert np.min(lead_times) >= 0, "The lead times should be non-negative."
        assert len(prod_capacities) == num_stages * num_agents_per_stage, "The number of production capacities should be the total number of agents in the environment."
        assert np.min(prod_capacities) > 0, "The production capacities should be positive."
        assert len(sale_prices) == num_stages * num_agents_per_stage, "The number of unit sale prices should be the total number of agents in the environment."
        assert np.min(sale_prices) >= 0, "The unit sale prices should be non-negative."
        assert len(order_costs) == num_stages * num_agents_per_stage, "The number of unit order costs should be the total number of agents in the environment."
        assert np.min(order_costs) >= 0, "The unit order costs should be non-negative."
        assert len(backlog_costs) == num_stages * num_agents_per_stage, \
            "The number of unit backlog costs for unfulfilled orders should be the total number of agents in the environment."
        assert np.min(backlog_costs) >= 0, "The unit penalties for unfulfilled orders should be non-negative."
        assert len(holding_costs) == num_stages * num_agents_per_stage, \
            "The number of unit inventory holding costs should be the total number of agents in the environment."
        assert np.min(holding_costs) >= 0, "The unit inventory holding costs should be non-negative."
        assert len(supply_relations) == num_stages and len(supply_relations[0]) == num_agents_per_stage, \
            "The number of supply relations should be the total number of agents in the environment"
        assert len(demand_relations) == num_stages and len(demand_relations[0]) == num_agents_per_stage, \
            "The number of demand relations should be the total number of agents in the environment"
        assert len(stage_names) == num_stages, "The number of stage names should equal to the number of stages."

        # Set the environment configurations
        self.num_stages = num_stages
        self.num_agents_per_stage = np.max([num_agents_per_stage])
        self.num_periods = num_periods
        self.stage_names = stage_names
        self.demand_dist = demand_dist
        self.init_inventories = np.array(init_inventories, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.lead_times = np.array(lead_times, dtype=int).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)
        self.max_lead_time = np.max(self.lead_times)
        self.demand_fn = demand_fn
        self.prod_capacities = np.array(prod_capacities, dtype=int).reshape(self.num_stages, self.num_agents_per_stage)
        self.max_production = np.max(self.prod_capacities)
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
        
        # Create all variables
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

        # Compute the upper bounds for state variables
        max_production = self.max_production
        # max_sale_price = np.max(self.sale_prices)
        max_order_cost = np.max(self.order_costs)
        max_backlog_cost = np.max(self.backlog_costs)
        max_holding_cost = np.max(self.holding_costs)
        max_lead_time = self.max_lead_time
        max_order = max_production
        max_inventory = max_order * self.num_periods

        # Set the observation and action spaces
        max_coop = np.array([1 for _ in range(num_agents_per_stage)])
        self.action_order_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.Discrete(n=max_order + 1, start=0, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })
        self.action_supply_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })
        self.action_demand_space = spaces.Dict({
            f"stage_{stage}_agent_{agent}": spaces.MultiDiscrete(nvec=max_coop, seed=init_seed + stage)
            for stage in range(self.num_stages) for agent in range(num_agents_per_stage)
        })


        self.state_dict = {f"stage_{m}_agent_{x}": None for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment variables including:
            - I: inventory at each stage
            - O: order placed by each stage
            - R: arriving order for each stage
            - S: sales by each stage
            - B: backlog for each stage
            - D: customer demand at the retailer
            - P: profit at each stage

        :param seed: seed for the new episode
        :param options: options
        :return: states, infos
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset all variables
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

        # Set the initial condition and state
        self.inventories[:, :, 0] = self.init_inventories # (stage, agent, period)
        self.supply_relations = copy.deepcopy(self.init_supply_relations)
        self.demand_relations = copy.deepcopy(self.init_demand_relations)
        self.sale_prices = copy.deepcopy(self.init_sale_prices)
        self.sc_graph.reset_G()
        self.update_state()

        return self.state_dict, {}

    def update_state(self) -> None:
        """
        Update the environment state including the current stage features, inventory, backlog, upstream backlog,
        supply relations, demand relations, 
        previous sales, arriving deliveries, 

        State: s_{m,x,t} = [c_m_x, p_m_x, r_m_x, k_m_x, h_m_x, L_m_x, I_{m,x,t-1}, B_{m,x,t-1}, B_{m+1,x,t-1},
        sup_rel, dem_rel, 
        S_{m,x,t-L_max}, ..., S_{m,x,t-1}, 0, ..., 0, R_{m,x,t-L_m}, ..., R_{m,x,t-1}]
        """
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
        states["recent_sales"] = np.zeros(shape=(self.num_stages, self.num_agents_per_stage, lt_max), dtype=int)
        if t >= lt_max:
            states["recent_sales"] = self.sales[:, :, (t - lt_max + 1):(t + 1)]
        elif t > 0:
            states["recent_sales"][:, :, -t:] = self.sales[:, :, 1:(t + 1)]

        states["arriving_deliveries"] = np.zeros(shape=(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage, lt_max), dtype=int)

        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                for j in range(self.num_agents_per_stage):
                    # if self.supply_relations[m][x][j] == 1:
                    lt = self.lead_times[m][x][j]
                    if t >= lt:
                        states["arriving_deliveries"][m, x, j, -lt:] = self.arriving_orders[m, x, j, (t - lt + 1):(t + 1)]
                    elif t > 0:
                        states["arriving_deliveries"][m, x, j, -t:] = self.arriving_orders[m, x, j, 1:(t + 1)]

        self.state_dict = {}
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                agent_state = []
                agent_state.append(states["prod_capacities"][m][x])
                agent_state.append(states["sale_prices"][m][x])
                agent_state.append(states["order_costs"][m])
                agent_state.append(states["backlog_costs"][m][x])
                agent_state.append(states["holding_costs"][m][x])
                agent_state.append(states["lead_times"][m][x])
                agent_state.append(states["inventories"][m][x])
                agent_state.append(states["backlogs"][m][x])
                agent_state.append(states["upstream_backlogs"][m][x])
                agent_state.append(states["suppliers"][m][x])
                agent_state.append(states["customers"][m][x])
                agent_state.append(states["recent_sales"][m][x])
                agent_state.append(states["arriving_deliveries"][m][x])
                agent_state.append(states["prod_costs"][m][x])
                agent_state.append(self.assets[m][x])
                
                self.state_dict[f"stage_{m}_agent_{x}"] = agent_state

        return self.state_dict

    def step(self, order_dict: dict[str, int], price_dict: dict[str, int], sup_dict: dict[str, list], dem_dict: dict[str, list]) -> tuple[dict, dict, dict, dict, dict]:
        """
        Take a step and return the next observation

        :param action_dict: action (order quantity) for each stage
        :return: states, rewards, terminations, truncations, infos
        """
        assert np.all(f"stage_{m}_agent_{x}" in order_dict for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities for all stages are required."
        assert np.all(order_dict[f"stage_{m}_agent_{x}"] >= 0 for m in range(self.num_stages) for x in range(self.num_agents_per_stage)), \
            "Order quantities must be non-negative integers."

        # env.py ▸ InventoryManagementEnv.step()
        def step(self, order_dict, sup_dict, dem_dict, price_dict):
            t = self.cur_period

            for s in range(self.num_stages):
                for a in range(self.num_agents_per_stage):
                    key = f"stage_{s}_agent_{a}"
                    order_vec = order_dict[key]
                    unit_costs = self.order_costs[s, a]
                    total_cost = np.dot(order_vec, unit_costs)

                    cash = float(self.assets[s, a])
                    if total_cost > cash and total_cost > 0:
                        ratio = cash / total_cost * 0.9
                        order_vec[:] = np.floor(order_vec * ratio).astype(int)
                        order_dict[key] = order_vec

                        if t == 0:
                            print(f"[ASSET_CLIP] {key}: cost {total_cost:.0f} > cash {cash:.0f}, "
                                  f"clip ratio={ratio:.2f}")

        # Get the inventory at the beginning of the period
        self.period += 1
        t = self.period
        M = self.num_stages
        current_inventories = self.inventories[:, :, t - 1]
        # update supple/demand relations if needed
        if self.enable_graph_change:
            self.supply_relations = np.stack([sup_dict[f"stage_{m}_agent_{x}"] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)                                                                                                                                    
        self.orders[:, :, :, t] = np.stack([order_dict[f"stage_{m}_agent_{x}"]*self.supply_relations[m][x] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)]).reshape(self.num_stages, self.num_agents_per_stage, self.num_agents_per_stage)

        self.demands[t] = int(self.demand_fn(t))
        # Add the delivered orders
        for m in range(self.num_stages):
            for i in range(self.num_agents_per_stage):
                for j in range(self.num_agents_per_stage):                    
                    lt = self.lead_times[m][i][j]
                    if t >= lt:
                        current_inventories[m][i] += self.arriving_orders[m, i, j, t - lt]

        # Compute the fulfilled orders
        cum_req_orders = np.sum(self.orders[:, :, :, t], axis=1)
        fulfilled_orders = np.zeros(shape=(self.num_stages, self.num_agents_per_stage), dtype=int)
        fulfilled_orders[:-1] = np.minimum(
            np.minimum(self.backlogs[1:, :, t - 1] + cum_req_orders[:-1], current_inventories[1:]),
            self.prod_capacities[1:])
        # R_{M-1,t} = O_{M-1,t}
        fulfilled_orders[M - 1] = cum_req_orders[M - 1]
        fulfilled_rates = (fulfilled_orders+1e-10) / (cum_req_orders+1e-10)
        fulfilled_rates = np.repeat(fulfilled_rates[:, np.newaxis, :], self.num_agents_per_stage, axis=1)
        self.arriving_orders[:, :, :, t] = (self.orders[:, :, :, t] * fulfilled_rates).astype(int)

        # Compute the sales
        cum_fulfilled_orders = np.sum(self.arriving_orders, axis=1)
        # S_{m,t} = R_{m-1,t}, m = 1, ..., M - 1
        self.sales[1:, :, t] = cum_fulfilled_orders[:-1, :, t]
        # S_{0,t} = min(B_{0,t-1} + D_{t}, I_{0,t-1} + R_{0,t-L_m}, c_0)

        self.sales[0, :, t] = np.minimum(
            np.minimum(self.backlogs[0, :, t - 1] + self.demands[t], current_inventories[0]),
            self.prod_capacities[0])
        
        # Compute the backlogs
        # B_{m,t} = B_{m,t-1} + O_{m-1,t} - S_{m,t}, m = 1, ..., M - 1
        self.backlogs[1:, :, t] = self.backlogs[1:, :, t - 1] + cum_req_orders[:-1] - self.sales[1:, :, t]
        # B_{0,t} = B_{0,t-1} + D_{t} - S_{0,t}
        self.backlogs[0, :, t] = self.backlogs[0, :, t - 1] + self.demands[t] - self.sales[0, :, t]

        # Compute the inventory at the end of the period
        self.inventories[:, :, t] = current_inventories - self.sales[:, :, t]

        # Compute the profits
        order_costs = np.repeat(self.order_costs[:, np.newaxis, :], self.num_agents_per_stage, axis=1)
        order_costs = order_costs * self.arriving_orders[:, :, :, t]
        order_costs = np.sum(order_costs, axis=2)
        self.profits[:, :, t] = self.sale_prices * self.sales[:, :, t] - order_costs \
                             - self.backlog_costs * self.backlogs[:, :, t] - self.holding_costs * self.inventories[:, :, t]
        self.total_profits[t] = np.sum(self.profits[:, :, t])

        # Update the assets 
        self.assets += self.profits[:, :, t]

        # Bankruptcy rule: if an operating agent's assets ≤ 0, shut it down immediately
        for m in range(self.num_stages):
            for x in range(self.num_agents_per_stage):
                if self.running_agents[m][x] == 1 and self.assets[m][x] <= 0:
                    print(f"[Bankruptcy] t={t}: stage_{m}_agent_{x} shuts down (assets={self.assets[m][x]})")
                    self.bankruptcy_log.append((t, m, x, float(self.assets[m][x])))
                    self.assets[m][x] = 0
                    self.create_shutdown_event(m, x, self.state_dict)
        
        # Update price if needed
        if self.enable_price_change:
            for m in range(self.num_stages):
                for x in range(self.num_agents_per_stage):
                    self.sale_prices[m][x] = price_dict[f"stage_{m}_agent_{x}"]
            self.order_costs[:M-1, :] = self.sale_prices[1:, :]
            
        # Determine rewards and terminations
        rewards = {f"stage_{m}_agent_{x}": self.profits[m, x, t] for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        all_termination = self.period >= self.num_periods
        terminations = {f"stage_{m}_agent_{x}": all_termination for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        terminations["__all__"] = all_termination
        truncations = {f"stage_{m}_agent_{x}": False for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        truncations["__all__"] = False
        infos = {f"stage_{m}_agent_{x}": {} for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
        
        #Stage-wide termination check: if any stage has all agents closed, end the episode
        stage_all_closed = np.all(self.running_agents == 0, axis=1)
        if stage_all_closed.any():
            dead_stage = int(np.where(stage_all_closed)[0][0])  # first fully-dead stage
            print(f"[Terminate] All agents in stage {dead_stage} are closed at t={t}. Ending episode.")

            # mark everyone terminated
            terminations = {f"stage_{m}_agent_{x}": True
                            for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
            terminations["__all__"] = True
            truncations = {f"stage_{m}_agent_{x}": False
                        for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}
            truncations["__all__"] = False
            infos = {f"stage_{m}_agent_{x}": {"terminated_stage": dead_stage}
                    for m in range(self.num_stages) for x in range(self.num_agents_per_stage)}

        if t == 0:
            print("[DBG] total_order_cost =", total_order_cost)
            print("[DBG] holding_cost     =", holding_cost)
            print("[DBG] backlog_cost     =", backlog_cost)

        # print("arriving orders in step", self.arriving_orders)
        self.update_state()

        return self.state_dict, rewards, terminations, truncations, infos

    def _parse_state(self, state: list) -> dict:
        """
        Parse a single stage state

        :param state: state
        :return: parsed state
        """
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
            'sales': state[11].tolist(),
            'deliveries': state[12].tolist(),
            'prod_cost': state[13], 
            'assets': state[14],
        }

    def parse_state(self, state_dict: dict = None) -> dict:
        """
        Parse the state dictionary

        :param state_dict: state dictionary
        :return: parsed state dict
        """
        if state_dict is None:
            state_dict = self.state_dict

        parsed_state = {}

        for stage_agent_id_name, state in state_dict.items():
            parsed_state[stage_agent_id_name] = self._parse_state(state)

        return parsed_state

    def no_backlog_env_proxy(self, stage_id: int, agent_id: int,
                         action_order_dict: dict, action_sup_dict: dict, action_price_dict: dict):
        # Keep the supply relation at the initial stage
        sup_action = self.supply_relations[stage_id][agent_id]
        if sum(sup_action) == 0: # if it is just recovered
            sup_action = self.init_supply_relations[stage_id][agent_id]
        action_sup_dict[f'stage_{stage_id}_agent_{agent_id}'] = sup_action
        
        # stage_order_action = np.random.uniform(1, 10, num_agents_per_stage).astype(int) * sup_action
        # split demand over multiple stage 1 suppliers
        demand=self.demand_fn(self.period)
        if stage_id == 0:
            stage_order_action = ((demand * sup_action)/sum(sup_action)).astype(int)
        elif stage_id == self.num_stages - 1: # manufacturers only have one shared suppliers
            avg_order = np.mean([np.sum(action_order_dict[x]) for x in action_order_dict if f"stage_{stage_id-1}" in x])
            stage_order_action = sup_action * avg_order.astype(int)
        else: # split demand over multiple stage_id + 1 suppliers
            avg_order = np.mean([np.sum(action_order_dict[x]) for x in action_order_dict if f"stage_{stage_id-1}" in x])/sum(sup_action)
            stage_order_action = sup_action * avg_order.astype(int)
        action_order_dict[f'stage_{stage_id}_agent_{agent_id}'] = stage_order_action

        # Keep the initial price
        price = self.init_sale_prices[stage_id][agent_id]
        action_price_dict[f"stage_{stage_id}_agent_{agent_id}"] = price

        return action_sup_dict, action_order_dict, action_price_dict


    def get_all_shutdown_agents(self):
        print("The closed agents are", ", ".join(self.shutdown_agents_set))

    def create_shutdown_event(self, stage_id: int, agent_id: int, state_dict: dict):
        print(f"Shutdown stage_{stage_id}_agent_{agent_id}. ")
        self.running_agents[stage_id][agent_id] = 0
        # remove shutdown agents from the suppliers of each downstream agents
        if stage_id != 0: # has downstream customer
            for down_agent_id in range(self.num_agents_per_stage):
                # state_dict[f"stage_{stage_id-1}_agent_{down_agent_id}"]["suppliers"][agent_id] = 0
                self.supply_relations[stage_id-1][down_agent_id][agent_id] = 0
        # store newly shutdown agents in the list
        self.shutdown_agents_set.add(f"stage_{stage_id}_agent_{agent_id}")
        # clear the inventory of shutdown agents
        self.inventories[stage_id, agent_id, self.period] = 0

        return state_dict

    def create_recovery_event(self, stage_id: int, agent_id: int):
        print(f"Re-open stage_{stage_id}_agent_{agent_id}.")
        self.running_agents[stage_id][agent_id] = 1
        # Remove the recovered agents from the shutdown list
        self.shutdown_agents_set.discard(f"stage_{stage_id}_agent_{agent_id}")

    def create_demand_surge(self):
        # double the expectation of demand distribution
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
    """
    Create the environment
    """
    agent_profiles = create_agent_profiles(env_config=env_config)
    sc_graph = SupplyChain_Graph(agent_profiles=agent_profiles, num_stages=env_config["num_stages"], num_agents_per_stage=env_config["num_agents_per_stage"])
    
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
        sc_graph = sc_graph,
    )


if __name__ == '__main__':

    config_name = 'large_graph_test'
    # create the dir to store the results
    os.makedirs(f"results/{config_name}", exist_ok=True)
    clear_dir(f"results/{config_name}")
    # create the dir to store the env setup
    os.makedirs(f"env/{config_name}", exist_ok=True)
    clear_dir(f"env/{config_name}")
    ec = get_env_configs(env_configs[config_name])
    im_env = env_creator(env_config=ec)
    im_env.reset()
    
    print(f"stage_names = {im_env.stage_names}")
    print(f"state_dict = {im_env.state_dict}")
    print(f"state_dict = {im_env.parse_state(im_env.state_dict)}")
    print(f"observation_space = {im_env.observation_space}")
    print(f"observation_sample = {im_env.observation_space.sample()}")
    print(f"action_order_space = {im_env.action_order_space}")
    print(f"action_order_sample = {im_env.action_order_space.sample()}")
    print(f"action_supply_space = {im_env.action_supply_space}")
    print(f"action_demand_space = {im_env.action_demand_space}")
    visualize_state(env=im_env, rewards={}, t=-1, save_prefix=config_name)
    num_agents_per_stage = im_env.num_agents_per_stage

    for t in range(im_env.num_periods):
        print('-' * 80)
        print(f"period = {t}")
        sup_dict = {}
        dem_dict = {}
        supply_relations, demand_relations = generate_sup_dem_relations(type=env_configs[config_name]["sup_dem_relation_type"], \
                                                                        num_customers=env_configs[config_name]["num_init_customers"], num_suppliers=env_configs[config_name]["num_init_suppliers"], \
                                                                        num_stages=im_env.num_stages, num_agents_per_stage=im_env.num_agents_per_stage)
        for m in range(im_env.num_stages):
            for x in range(im_env.num_agents_per_stage):
                if m == 0: # retailer
                    sup_dict[f"stage_{m}_agent_{x}"] = supply_relations[m][x]
                    dem_dict[f"stage_{m}_agent_{x}"] = demand_relations[m][x]
                elif m == im_env.num_stages - 1: # manufacturer
                    sup_dict[f"stage_{m}_agent_{x}"] = supply_relations[m][x]
                    dem_dict[f"stage_{m}_agent_{x}"] = demand_relations[m][x]
                else:
                    sup_dict[f"stage_{m}_agent_{x}"] = supply_relations[m][x]
                    dem_dict[f"stage_{m}_agent_{x}"] = demand_relations[m][x]

        next_state_dict, rewards, terminations, truncations, infos = im_env.step(
            order_dict={f"stage_{m}_agent_{x}": np.array([4 for _ in range(num_agents_per_stage)]) for m in range(im_env.num_stages) for x in range(im_env.num_agents_per_stage)}, 
            sup_dict=sup_dict,
            dem_dict=dem_dict
        )
               
        # print(f"next_state_dict = {next_state_dict}")
        # print(f"next_state_dict = {im_env.parse_state(next_state_dict)}")
        # print(f"rewards = {rewards}")
        # print(f"terminations = {terminations}")
        # print(f"truncations = {truncations}")
        # print(f"infos = {infos}")
        visualize_state(env=im_env, rewards=rewards, t=t, save_prefix=config_name)
        if np.any(im_env.backlogs[1:, :, t]< 0) :
            print("backlogs become negative")
            break
