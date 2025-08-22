"""
Environment Configurations
"""
import numpy as np
from data_simulation import generate_lead_time, generate_prod_capacity
from data_simulation import generate_cost_price, generate_sup_dem_relations
from data_simulation import generate_holding_costs, generate_backlog_costs, generate_init_inventories
from data_simulation import Demand_fn, generate_init_assets
import os
from utils import save_dict_to_json
from collections import defaultdict

np.random.seed(0)

env_configs = {
    "llm_sampling" : {
        "config_name": "llm_sampling",
        "sup_dem_relation_type": "fix",
        "num_init_suppliers": 1,
        "num_init_customers": 1,
        "num_agents_per_stage": 1,
        "num_periods": 80,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],

        "init_inventory_dist": ("uniform", 10, 15),
        "price_cost_dist": "uniform",
        "lead_time_dist": ("constant", 2),
        "prod_capacity_dist": ("uniform", 10, 20),
        "demand_fn": ("uniform_demand", 4, 8),
        "holding_costs_dist": "constant",
        "backlog_costs_dist": "constant",
        "profit_rate_dist": ("uniform", 0, 1),
        "init_assets_dist": ("cost_based",),
        "init_assets_multiple": 10,
        "llm_agents": [(0,0), (1,0), (2,0), (3,0)],
        "enable_graph_change": False,
        "enable_price_change": False,
        "state_format": "base",
        "env_no_backlog": False,
        "emergent_events": [],
        "shut_seq": {},
        "rec_seq": {}
    },
    "test": {
        "config_name": "test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 1,
        "num_init_customers": 1,
        "num_agents_per_stage": 4, # >= 2
        "num_periods": 16,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": "constant", # constant/uniform/etc
        "price_cost_dist": "constant", # constant/uniform/normal/etc
        "lead_time_dist": "constant", # constant/uniform
        "prod_capacity_dist": "constant", # constant/uniform
        "demand_fn": "constant_demand", # constant/functional
        "holding_costs_dist": "constant",
        "backlog_costs_dist": "constant",
        "profit_rate_dist": "constant",
        "llm_agents": None,
        "state_format": "base", 
    },
    "basic": {
        "config_name": "basic",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 1,
        "num_init_customers": 1, 
        "num_stages": 4,
        "num_agents_per_stage": 4, # >= 2
        "num_periods": 8,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 5), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 20), # constant/uniform
        "demand_fn": ("constant_demand", 5), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(i, j) for i in range(4) for j in range(4)],
        "state_format": "base", 
        "enable_graph_change": True, 
    }, 
    "large_graph_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 10,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 50, 80), # constant/uniform
        "demand_fn": ("constant_demand", 10), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "env_no_backlog": True, 
        "emergent_events": [], 
        "shut_seq": {},
        "rec_seq": {},
    },
    "large_graph_test_ee": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 10,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 80), # constant/uniform
        "demand_fn": ("constant_demand", 10), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "emergent_events": [(5, "sudden_shutdown"), (7, "recovery")], 
        "shut_seq": {5: [(2, 2), (2, 10), (2, 13)]},
        "rec_seq": {7: [(2,2), (2,10)]},
    },
    "large_graph_normal_demand_test": {
        "config_name": "large_graph_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 3,
        "num_init_customers": 3,
        "num_agents_per_stage": 20, # >= 2
        "num_periods": 8,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 80), # constant/uniform
        "demand_fn": ("normal_demand", 10, 3), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(0, 1)],
        "enable_graph_change": True, 
        "state_format": "base", 
    },
    "large_graph_DynPoisson_demand_test": {
        "config_name": "large_graph_DynPoisson_demand_test",
        "sup_dem_relation_type": "random", # random/fixed
        "num_init_suppliers": 3,
        "num_init_customers": 3,
        "num_agents_per_stage": 100, # >= 2
        "num_periods": 10,
        "num_stages": 4,
        "stage_names": ['retailer', 'wholesaler', 'distributor', 'manufacturer'],
        "init_inventory_dist": ("uniform", 10, 15), # constant/uniform/etc
        "price_cost_dist": "uniform", # constant/uniform/normal/etc
        "lead_time_dist": ("uniform", 1, 10), # constant/uniform
        "prod_capacity_dist": ("uniform", 10, 80), # constant/uniform
        "demand_fn": ("normal_demand", 10, 3), # constant/functional
        "holding_costs_dist": "constant", 
        "backlog_costs_dist": "constant", 
        "profit_rate_dist": ("uniform", 0, 1), 
        "llm_agents": [(1, 1)],
        "enable_graph_change": True, 
        "enable_price_change": False, 
        "state_format": "base", 
        "emergent_events": [(5, "demand_surge"), (6, "sudden_shutdown"), (7, "recovery")], 
        "shut_seq": {6:[(0, 3), (2, 5)]},
        "rec_seq": {7:[(2, 5)]}, 
    },
}

def get_env_configs(env_configs: dict):
    
    save_dict_to_json(data=env_configs, save_path=env_configs['config_name'])
    num_stages = env_configs["num_stages"]
    num_agents_per_stage = env_configs["num_agents_per_stage"]
    num_periods = env_configs["num_periods"]
    num_total_agents = num_stages * num_agents_per_stage
    
    supply_relations, demand_relations = \
        generate_sup_dem_relations(type=env_configs["sup_dem_relation_type"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, \
                                   num_suppliers=env_configs["num_init_suppliers"], num_customers=env_configs["num_init_customers"])
    order_costs, sale_prices, prod_costs = \
        generate_cost_price(prod_cost_dist=env_configs["price_cost_dist"], profit_rate_dist=env_configs["profit_rate_dist"], \
                            num_stages=num_stages, num_agents_per_stage=num_agents_per_stage, config_name=env_configs["config_name"])
    holding_costs = \
        generate_holding_costs(dist=env_configs["holding_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    backlog_costs = \
        generate_backlog_costs(dist=env_configs["backlog_costs_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    lead_times = \
        generate_lead_time(dist=env_configs["lead_time_dist"], num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,config_name=env_configs["config_name"])
    prod_capacities = \
        generate_prod_capacity(dist=env_configs['prod_capacity_dist'], num_data=num_total_agents, config_name=env_configs["config_name"])
    init_inventories = \
        generate_init_inventories(dist=env_configs["init_inventory_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    multiple = env_configs.get("init_assets_multiple", 10)
    init_assets = generate_init_assets(
        dist=env_configs.get("init_assets_dist", ("cost_based",)),
        num_stages=num_stages,
        num_agents_per_stage=num_agents_per_stage,
        order_costs=order_costs.reshape(num_stages, num_agents_per_stage),
        prod_costs=prod_costs.reshape(num_stages, num_agents_per_stage),
        config_name=env_configs["config_name"],
        multiple=multiple  # ← 用配置里的值
    )

    # profit_rates = \
    #     generate_profit_rates(dist=env_configs["profit_rate_dist"], num_data=num_total_agents, config_name=env_configs["config_name"])
    demand_fn = Demand_fn(dist=env_configs["demand_fn"])
    stage_names = env_configs["stage_names"]
    llm_agents = env_configs["llm_agents"]
    state_format = env_configs["state_format"]
    env_no_backlog = env_configs["env_no_backlog"]
    if env_no_backlog:
        for m in range(num_stages):
            for x in range(num_agents_per_stage):
                if not ((m, x) in llm_agents):
                    init_inventories[m*num_agents_per_stage+x] = int(1e5)
    enable_graph_change = env_configs["enable_graph_change"]
    enable_price_change = env_configs["enable_price_change"]
    emergent_events = defaultdict(list)
    for (t, ee) in env_configs["emergent_events"]:
        emergent_events[t].append(ee)
    shut_seq = env_configs["shut_seq"]
    rec_seq = env_configs["rec_seq"]

    return {
        'num_stages': num_stages,
        'num_periods': num_periods,
        'num_agents_per_stage': num_agents_per_stage,
        "demand_dist": env_configs["demand_fn"][0],
        'init_inventories': init_inventories, # num_stages * num_agents_per_stage
        'init_assets': init_assets, 
        'lead_times': lead_times, # num_stages * num_agents_per_stage * num_agents_per_stage
        'demand_fn': demand_fn,
        'prod_capacities': prod_capacities,
        'sale_prices': sale_prices,
        'order_costs': order_costs,
        "prod_costs": prod_costs, 
        'backlog_costs': backlog_costs,
        'holding_costs': holding_costs,
        'supply_relations': supply_relations,
        "demand_relations": demand_relations,
        'stage_names': stage_names,
        "llm_agents": llm_agents,
        "state_format": state_format, 
        "env_no_backlog": env_no_backlog, 
        "enable_graph_change": enable_graph_change,
        "enable_price_change": enable_price_change, 
        "emergent_events": emergent_events,
        "shut_seq": shut_seq,
        "rec_seq": rec_seq,  
    }
    

