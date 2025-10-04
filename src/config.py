# src/config.py
# Minimal, runnable config module with msbs44 and get_env_configs compatible with your env.py.

import numpy as np
from utils import save_dict_to_json
from data_simulation import (
    generate_lead_time, generate_prod_capacity, generate_cost_price,
    generate_sup_dem_relations, generate_holding_costs, generate_backlog_costs,
    generate_init_inventories, Demand_fn, generate_init_assets,
)

np.random.seed(0)

# ----- env_configs: include only the one you currently use (msbs44) -----
env_configs = {
    "msbs44": {
        "config_name": "msbs44",
        "sup_dem_relation_type": "full",
        "num_init_suppliers": 2,
        "num_init_customers": 2,
        "num_agents_per_stage": 4,
        "num_periods": 10,
        "num_stages": 4,
        "stage_names": ['retailer', "wholesaler", 'distributor', 'manufacturer'],

        # Resurrection / reflection (kept for compatibility)
        "enable_resurrection": False,
        "enable_regular_reflection_with_resurrection": True,
        "enable_bankruptcy_reflection": False,
        "max_bankruptcies": 1,

        "revive_policy": "fix",
        "revive_assets": 2500,
        "revive_inventory": 50,
        "reset_suppliers_on_revive": True,
        "stop_on_exhausted_resurrections": True,

        # Distributions
        "init_inventory_dist": ("uniform", 10, 15),
        "price_cost_dist": "uniform",
        "lead_time_dist": ("constant", 2),
        "prod_capacity_dist": ("uniform", 6666, 6666),
        "demand_fn": ("sin_demand", 12, 4, 12, 0, 0),  # legacy global demand

        # If you want per-retailer demand, uncomment and edit the list below:
         "demand_fn_stage0_list": [
             ("sin_demand", 12, 4, 12, 0, 0.0),
             ("sin_demand", 12, 4, 12, 0, 0.5),
             ("sin_demand", 12, 4, 12, 0, 1.0),
             ("sin_demand", 12, 4, 12, 0, 2.0),
         ],

        "holding_costs_dist": "constant",
        "backlog_costs_dist": "constant",
        "profit_rate_dist": ("uniform", 0, 1),
        "init_assets_dist": ("cost_based",),

        # Scalers
        "init_assets_multiple": 6666,
        "init_inventory_multiplier": 1,
        "init_assets_multiplier": 6666,

        # LLM placement & toggles
        "llm_agents": [(i, j) for i in range(4) for j in range(4)],
        "enable_graph_change": False,
        "enable_price_change": False,
        "state_format": "base",
        "env_no_backlog": False,
    },
}


def get_env_configs(env_configs: dict):
    """
    Expand a high-level config entry into a full environment config dict.
    Compatible with your current env.py expectations, including demand_fns_stage0.
    """
    save_dict_to_json(data=env_configs, save_path=env_configs['config_name'])

    num_stages = env_configs["num_stages"]
    num_agents_per_stage = env_configs["num_agents_per_stage"]
    num_periods = env_configs["num_periods"]
    num_total_agents = num_stages * num_agents_per_stage

    # Relations
    supply_relations, demand_relations = generate_sup_dem_relations(
        type=env_configs["sup_dem_relation_type"],
        num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,
        num_suppliers=env_configs["num_init_suppliers"], num_customers=env_configs["num_init_customers"]
    )

    # Costs & prices
    order_costs, sale_prices, prod_costs = generate_cost_price(
        prod_cost_dist=env_configs["price_cost_dist"],
        profit_rate_dist=env_configs["profit_rate_dist"],
        num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,
        config_name=env_configs["config_name"]
    )

    # Per-agent cost parameters
    holding_costs = generate_holding_costs(
        dist=env_configs["holding_costs_dist"],
        num_data=num_total_agents,
        config_name=env_configs["config_name"]
    )
    backlog_costs = generate_backlog_costs(
        dist=env_configs["backlog_costs_dist"],
        num_data=num_total_agents,
        config_name=env_configs["config_name"]
    )

    # Lead time & capacity
    lead_times = generate_lead_time(
        dist=env_configs["lead_time_dist"],
        num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,
        config_name=env_configs["config_name"]
    )
    prod_capacities = generate_prod_capacity(
        dist=env_configs['prod_capacity_dist'],
        num_data=num_total_agents,
        config_name=env_configs["config_name"]
    )

    # Initial inventories
    init_inventories = generate_init_inventories(
        dist=env_configs["init_inventory_dist"],
        num_data=num_total_agents,
        config_name=env_configs["config_name"]
    )

    # Initial assets
    multiple = env_configs.get("init_assets_multiple", 10)
    init_assets = generate_init_assets(
        dist=env_configs.get("init_assets_dist", ("cost_based",)),
        num_stages=num_stages, num_agents_per_stage=num_agents_per_stage,
        order_costs=order_costs.reshape(num_stages, num_agents_per_stage),
        prod_costs=prod_costs.reshape(num_stages, num_agents_per_stage),
        config_name=env_configs["config_name"],
        multiple=multiple
    )

    # Optional multipliers
    inv_mul = env_configs.get("init_inventory_multiplier", 1)
    ass_mul = env_configs.get("init_assets_multiplier", 1.0)
    if inv_mul != 1:
        init_inventories = (np.asarray(init_inventories, dtype=int) * int(inv_mul)).astype(int).tolist()
    if ass_mul != 1.0:
        init_assets = (np.asarray(init_assets, dtype=float) * float(ass_mul)).astype(float)

    # Demand function(s)
    demand_fn = Demand_fn(dist=env_configs["demand_fn"])
    demand_fns_stage0 = None
    try:
        per_list = env_configs.get("demand_fn_stage0_list", None)
        if isinstance(per_list, (list, tuple)) and len(per_list) == num_agents_per_stage:
            demand_fns_stage0 = [Demand_fn(dist=tuple(spec)) for spec in per_list]
    except Exception:
        demand_fns_stage0 = None

    stage_names = env_configs["stage_names"]
    llm_agents = env_configs["llm_agents"]
    state_format = env_configs["state_format"]

    # No-backlog shortcut for non-LLM agents (kept for compatibility)
    env_no_backlog = env_configs.get("env_no_backlog", False)
    if env_no_backlog:
        for m in range(num_stages):
            for x in range(num_agents_per_stage):
                if not ((m, x) in (llm_agents or [])):
                    init_inventories[m * num_agents_per_stage + x] = int(1e5)

    enable_graph_change = env_configs.get("enable_graph_change", False)
    enable_price_change = env_configs.get("enable_price_change", False)

    return {
        'num_stages': num_stages,
        'num_periods': num_periods,
        'num_agents_per_stage': num_agents_per_stage,
        "demand_dist": env_configs["demand_fn"][0],

        'init_inventories': init_inventories,
        'init_assets': init_assets,
        'lead_times': lead_times,
        'demand_fn': demand_fn,
        'demand_fns_stage0': demand_fns_stage0,
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

        # Resurrection / reflection toggles
        "enable_resurrection": env_configs.get("enable_resurrection", False),
        "max_bankruptcies": env_configs.get("max_bankruptcies", 5),
        "revive_policy": env_configs.get("revive_policy", "initial_assets_initinv_plus_backlog"),
        "reset_suppliers_on_revive": env_configs.get("reset_suppliers_on_revive", False),
        "stop_on_exhausted_resurrections": env_configs.get("stop_on_exhausted_resurrections", True),
        "revive_assets": env_configs.get("revive_assets", 1000.0),
        "revive_inventory": env_configs.get("revive_inventory", 50),
        "enable_regular_reflection_with_resurrection": env_configs.get(
            "enable_regular_reflection_with_resurrection", False
        ),
        "enable_bankruptcy_reflection": env_configs.get("enable_bankruptcy_reflection", True),

        # Order cap for mas_model
        "order_cap_ratio": env_configs.get("order_cap_ratio", 2.0),
    }
