import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
import re
import json
import dgl
from typing import Callable
from typing import List


def save_string_to_file(data: str, save_path: str, t: int, round: int, reward: int):
    print("Saving data to: ", f"results/{save_path}/chat_summary_round{round}_period{t}_reward{reward}.txt")
    with open(f"results/{save_path}/chat_summary_round{round}_period{t}_reward{reward}.txt", 'w') as f:
        f.write(data)

def save_dict_to_json(data: dict, save_path: str):
    print("Saving config to: ", f"env/{save_path}/config.json")
    with open(f"env/{save_path}/config.json", 'w') as f:
        json.dump(data, f)

def clear_dir(dir_path: str):
    # Clear the directory
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))

def save_array(data: np.ndarray, save_path: str):
    print("Saving data to: ", save_path)
    np.save(save_path, data)


def extract_pairs(input_string: str) -> dict[str, int]:
    """
    从字符串中抽取形如 agent0=8 / agent0:8 / ("agent0", 8) 等键值对
    返回 dict，键固定 'agent0'、'agent1'...
    """
    # ① 允许可选的外层括号与双引号
    # ② 分隔符 [:,=] 任意其一
    pattern = r'[("\[]?\s*"?([a-zA-Z0-9_]+)"?\s*[:=,]\s*([0-9]+)\s*[)\]]?'

    matches = re.findall(pattern, input_string)
    # [('agent0', '8'), ('agent1', '5'), ...] → {'agent0': 8, ...}
    return {agent: int(value) for agent, value in matches}


def parse_stage_agent_id(stage_agent_id_name: str):
    # Extract stage and agent from the string
    id_name = stage_agent_id_name.replace("agent_", "").replace("stage_", "")
    stage, agent = id_name.split("_")

    return int(stage), int(agent)

# Create a multipartite graph
def draw_multipartite_graph(env, t: int, save_prefix: str):

    num_stages = env.num_stages
    num_agents_per_stage = env.num_agents_per_stage
    sup_rel = env.supply_relations
    dem_rel = env.demand_relations
    save_path = f'results/{save_prefix}/'

    M = nx.DiGraph()

    # Add nodes for each set
    stage_agents = []
    for m in range(num_stages):
        stage_agents = []
        for x in range(num_agents_per_stage):
            stage_agents.append(f"s{m}a{x}")
        M.add_nodes_from(stage_agents, layer=num_stages-m)  # Add set A nodes

    # Add edges between the sets
    edges = []
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] == 1:
                    src = f"s{m+1}a{i}"
                    tgt = f"s{m}a{x}"
                    edges.append((src, tgt))
    M.add_edges_from(edges)

    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="layer")

    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = ["gold", "violet", "limegreen", "darkorange", "black"]
    colors = [stage_colors[m] for m in range(num_stages) for x in range(num_agents_per_stage)]
    # mask closed agents
    for m in range(num_stages):
        for x in range(num_agents_per_stage):
            if env.running_agents[m][x] == 0:
                colors[m*num_agents_per_stage+x] = "black"

    plt.figure(figsize=(25, 20))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
    plt.title("Multipartite Graph")
    plt.savefig(os.path.join(save_path, f"supply_chain_period_{t}.jpg"), format="jpg")


def draw_material_flow(env, t: int, save_prefix: str):
    num_stages = env.num_stages
    num_agents_per_stage = env.num_agents_per_stage
    sup_rel = env.arriving_orders[:, :, :, t]
    save_path = f'results/{save_prefix}/'

    M = nx.DiGraph()

    # Add nodes for each set
    stage_agents = []
    for m in range(num_stages):
        stage_agents = []
        for x in range(num_agents_per_stage):
            stage_agents.append(f"s{m}a{x}")
        M.add_nodes_from(stage_agents, layer=num_stages-m)  # Add set A nodes

    # Add edges between the sets
    edges = []
    edge_labels = {}
    for m in range(num_stages-1):
        for x in range(num_agents_per_stage):
            for i in range(num_agents_per_stage):
                if sup_rel[m][x][i] > 0:
                    src = f"s{m+1}a{i}"
                    tgt = f"s{m}a{x}"
                    edges.append((src, tgt))
                    edge_labels[(src, tgt)] = sup_rel[m][x][i]
    M.add_edges_from(edges)


    # Define positions for the multipartite layout
    pos = nx.multipartite_layout(M, subset_key="layer")

    # Draw the multipartite graph
    # stage_colors = plt.cm.plasma(np.linspace(0, 1, 4))
    stage_colors = ["gold", "violet", "limegreen", "darkorange",]
    colors = [stage_colors[m] for m in range(num_stages) for _ in range(num_agents_per_stage)]

    plt.figure(figsize=(25, 20))
    nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
    nx.draw_networkx_edge_labels(G=M, pos=pos, edge_labels=edge_labels)
    plt.title("Material Flow Graph")
    plt.savefig(os.path.join(save_path, f"material_flow_period_{t}.jpg"), format="jpg")



def visualize_state(env, rewards: dict, t: int, save_prefix: str):
    
    state_dict = env.state_dict
    num_stages = env.num_stages
    num_agents_per_stage = env.num_agents_per_stage
    lt_max = env.max_lead_time
    save_path = f'results/{save_prefix}/'
    df = pd.DataFrame({
        "stage": {},
        "agent_idx": {}, 
        "profits": {}, 
        "prod_capacity": {},
        "inventory": {},
        "sales_price": {},       
        "backlog_cost": {},
        "holding_cost": {},
        "backlog": {}, 
        "upstream_backlog": {},
        "suppliers": {},
        "customers": {},
        "order_cost": {},
        "prod_cost": {},
        "recent_sales": {},
        "lead_time": {},
        "deliveries": {},
    })
    for stage in range(num_stages):
        for agent in range(num_agents_per_stage):

            df = pd.concat([df, pd.DataFrame({
                    'stage': [stage], 
                    "agent_idx": [agent],
                    "prod_capacity": [state_dict[f'stage_{stage}_agent_{agent}'][0]],
                    'sales_price': [state_dict[f'stage_{stage}_agent_{agent}'][1]],
                    'order_cost': [state_dict[f'stage_{stage}_agent_{agent}'][2]],
                    'backlog_cost': [state_dict[f'stage_{stage}_agent_{agent}'][3]],
                    'holding_cost': [state_dict[f'stage_{stage}_agent_{agent}'][4]],
                    'lead_time': [state_dict[f'stage_{stage}_agent_{agent}'][5]],
                    'inventory': [state_dict[f'stage_{stage}_agent_{agent}'][6]],
                    'backlog': [state_dict[f'stage_{stage}_agent_{agent}'][7]],
                    'upstream_backlog': [state_dict[f'stage_{stage}_agent_{agent}'][8]],
                    "suppliers": [state_dict[f'stage_{stage}_agent_{agent}'][9]],
                    "customers": [state_dict[f'stage_{stage}_agent_{agent}'][10]],
                    'recent_sales': [state_dict[f'stage_{stage}_agent_{agent}'][11]],
                    'deliveries': [state_dict[f'stage_{stage}_agent_{agent}'][12]],
                    'prod_cost': [state_dict[f'stage_{stage}_agent_{agent}'][13]], 
                    'profits': [rewards.get(f'stage_{stage}_agent_{agent}', None)]
                    })], ignore_index=True)
    
    df = df.groupby(by=['stage', 'agent_idx']).apply(lambda x: x).reset_index(drop=True)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"env_period_{t}.csv"), index=False)
    draw_multipartite_graph(env=env, t=t, save_prefix=save_prefix)
    draw_material_flow(env=env, t=t, save_prefix=save_prefix)

def random_relations(n_cand: int, n_relation: int):

    return np.random.choice(a=n_cand, size=n_relation, replace=False)


def get_state_description(state: dict, past_req_orders: list, G: nx.Graph, state_format: str, enable_graph_change: bool, agent_name: str=None):
    if state_format == 'base':
        return get_base_description(state=state, past_req_orders=past_req_orders)
    elif state_format == "GraphML":
        return get_GraphML_description(G=G, agent_name=agent_name, enable_graph_change=enable_graph_change, state=state)
    else:
        raise AssertionError(f"{state_format} state description method not implemented yet")


def get_GraphML_description(agent_name: str, G: nx.DiGraph, enable_graph_change: bool, state: dict):

    # Convert to GraphML format
    # print(G.nodes())
    # print(G.edges())
    if enable_graph_change:
        upstream_nodes = [up for up in G.successors(agent_name) if G.nodes[up].get("stage")==G.nodes[agent_name].get("stage")+1]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        connected_nodes = upstream_nodes + customer_nodes + [agent_name]
        sub_graph = G.subgraph(connected_nodes)
    else:
        # Get nodes that have a "suppliers" relation with the given agent
        supplier_nodes = [supplier for node, supplier in G.edges(agent_name) if G.edges[node, supplier].get('customer')]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        sub_graph = G.subgraph(supplier_nodes + customer_nodes + [agent_name])
    graphml_str = '\n'.join(list(nx.generate_graphml(sub_graph, named_key_ids=True, prettyprint=True))[12:])

    recent_sales = f"\nPrevious Sales (in the recent round(s), from old to new): {state['sales']}\n"
    return graphml_str + recent_sales


def get_base_description(state, past_req_orders):

    suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==1])
    non_suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state['suppliers']) if state['suppliers'][i]==0])
    lead_times = " round(s); ".join([f"from agent{i}: {state['lead_times'][i]}" for i, _ in enumerate(state['lead_times'])])
    order_costs = " unit(s); ".join([f"from agent{i}: {state['order_costs'][i]}" for i, _ in enumerate(state['order_costs'])])
    prod_cost = state["prod_cost"]
    # get the arriving deliveries from the upstream in this round
    arriving_delieveries = []
    for i, _ in enumerate(state['suppliers']):
        if state['suppliers'][i] == 1:
            arriving_delieveries.append(f"from agent{i}: {state['deliveries'][i][-state['lead_times'][i]:]}")
    arriving_delieveries = "; ".join(arriving_delieveries)

    # get the requested orders from downstreams in this round
    req_orders = []
    if len(past_req_orders) == 0:
        req_orders = "None"
    else:
        for i, _ in enumerate(past_req_orders):
            if past_req_orders[i] != 0:
                req_orders.append(f"from agent{i}: {past_req_orders[i]}")
        req_orders = " ".join(req_orders)
    # print("req orders", req_orders)

    return (
        f" - Lead Time: {lead_times} round(s)\n"
        f" - Order costs: {order_costs} unit(s)\n"
        f" - Production costs: {prod_cost} unit(s)\n"
        f" - Inventory Level: {state['inventory']} unit(s)\n"
        f" - Production capacity: {state['prod_capacity']} unit(s)\n"
        f" - Current Backlog (you owing to the downstream): {state['backlog']} unit(s)\n"
        f" - Upstream Backlog (your upstream owing to you): {state['upstream_backlog']} unit(s)\n"
        f" - Previous Sales (in the recent round(s), from old to new): {state['sales']}\n"
        f" - In the last round, you placed orders to upstream suppliers: {req_orders}\n"
        f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_delieveries}\n"
        f" - Your upstream suppliers are: {suppliers}\n" 
        f" - Other available upstream suppliers in the environment are: {non_suppliers}\n"
    )


def get_demand_description(demand_fn: Callable) -> str:
    
    if demand_fn.dist == "constant_demand":
        mean = demand_fn.mean
        return f"The expected demand at the retailer (stage 0) is a constant {mean} units for all rounds."
    elif demand_fn.dist == "uniform_demand":
        lb = demand_fn.lb
        ub = demand_fn.ub
        return f"The expected demand at the retailer (stage 0) is a discrete uniform distribution U{lb, ub} for all rounds."
    elif demand_fn.dist == "seasonal_demand":
        return f"The expected demand at the retailer (stage 0) is a discrete uniform distribution U{0, 4} for the first 4 rounds, " \
            "and a discrete uniform distribution U{5, 8} for the last 8 rounds."
    elif demand_fn.dist == "normal_demand":
        mu = demand_fn.mean
        std = demand_fn.std
        return f"The expected demand at the retailer (stage 0) is a normal distribution N({mu}, {std}), " \
            "truncated at 0, for all 12 rounds."
    elif demand_fn.dist == "dyn_poisson_demand":
        mean = demand_fn.mean
        return f"The expected demand at the retailer (stage 0) is a poisson distribution P(lambda={mean}), and the lambda is increasingly bigger."
    else:
        raise KeyError(f"Error: {demand_fn} not implemented.")
  


def update_sup_action(sup_action: list, rm_match: str, add_match: str):
    
    remove_sup = rm_match.replace(" ", "")                
    if remove_sup != "":
        remove_sup = remove_sup.replace("agent", "").replace('"', "")
        try:
            remove_sup = [int(ind) for ind in remove_sup.split(",")]
            for ind in remove_sup:
                sup_action[ind] = 0
        except: # if the string format is invalid
            pass
    add_sup = add_match.replace(" ", "")   
    if add_sup != "":
        add_sup = add_sup.replace("agent", "").replace('"', "")
        try:
            add_sup = [int(ind) for ind in add_sup.split(",")]
            for ind in add_sup:
                sup_action[ind] = 1
        except:
            pass
    return sup_action

def parse_order_list(raw_str: str,
                     num_suppliers: int,
                     default_key_prefix: str = "agent") -> np.ndarray:
    """
    解析 Task3 返回的 order 字符串 → ndarray(order_qty, shape=(num_suppliers,))
    支持:
      1. [("agent0": 8), ("agent1": 3)]
      2. [8, 3]
      3. 8, 3
    解析不到时返回全 0。
    """
    # A. 键值对形式
    kv_pairs = re.findall(r'"?agent[_ ]?(\d+)"?\s*[:=]\s*([0-9]+)', raw_str)
    if kv_pairs:
        vec = np.zeros(num_suppliers, dtype=int)
        for idx, qty in kv_pairs:
            idx = int(idx)
            if idx < num_suppliers:
                vec[idx] = int(qty)
        return vec

    # B. 方括号纯数字
    m = re.search(r'\[(.*?)\]', raw_str, re.DOTALL)
    nums = re.findall(r'\d+', m.group(1)) if m else re.findall(r'\d+', raw_str)

    if nums:
        nums = [int(x) for x in nums]
        nums = (nums + [0] * num_suppliers)[:num_suppliers]  # 补/截断
        return np.array(nums, dtype=int)

    # fallback
    return np.zeros(num_suppliers, dtype=int)
    