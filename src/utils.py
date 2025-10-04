import os
import re
import json
import traceback
from os.path import abspath, exists
from typing import List, Callable

import numpy as np
import pandas as pd
import networkx as nx

# Use headless backend to render plots on servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dgl  # kept if other modules rely on it


# ------------------------------
# Basic I/O helpers
# ------------------------------

def _resolve_out_dir(save_prefix: str) -> str:
    """
    Turn save_prefix into a concrete directory path:
    - absolute path: use as-is
    - startswith 'results/': use as-is
    - otherwise: treat as 'results/<save_prefix>'
    """
    if os.path.isabs(save_prefix):
        out_dir = save_prefix
    elif save_prefix.startswith("results/") or save_prefix.startswith("results" + os.sep):
        out_dir = save_prefix
    else:
        out_dir = os.path.join("results", save_prefix)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[VIZDBG] _resolve_out_dir -> {abspath(out_dir)}")
    return out_dir

def save_string_to_file(data: str, save_path: str, t: int, round: int, reward: int):
    out = f"results/{save_path}/chat_summary_round{round}_period{t}_reward{reward}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("Saving data to:", out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(data)


def save_dict_to_json(data: dict, save_path: str):
    out = f"env/{save_path}/config.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print("Saving config to:", out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)


def clear_dir(dir_path: str):
    # Danger: this removes files under the directory (non-recursive)
    if not os.path.isdir(dir_path):
        return
    for file in os.listdir(dir_path):
        fp = os.path.join(dir_path, file)
        try:
            os.remove(fp)
        except Exception:
            pass


def save_array(data: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Saving data to:", save_path)
    np.save(save_path, data)


# ------------------------------
# Small parsers / utils
# ------------------------------
def extract_pairs(input_string: str) -> dict[str, int]:
    """
    Extract key-value pairs from strings like: agent0=8 / agent0:8 / ("agent0", 8)
    Returns a dict {agent_name: int_value}.
    """
    pattern = r'[("\[]?\s*"?([a-zA-Z0-9_]+)"?\s*[:=,]\s*([0-9]+)\s*[)\]]?'
    matches = re.findall(pattern, input_string)
    return {agent: int(value) for agent, value in matches}


def parse_stage_agent_id(stage_agent_id_name: str):
    # Convert "stage_1_agent_3" -> (1, 3)
    id_name = stage_agent_id_name.replace("agent_", "").replace("stage_", "")
    stage, agent = id_name.split("_")
    return int(stage), int(agent)


# ------------------------------
# Visualization helpers
# ------------------------------
def _stage_colors(num_stages: int, num_agents_per_stage: int) -> list[str]:
    # Cycle colors to avoid index error when stages > palette size
    base = ["gold", "violet", "limegreen", "darkorange", "deepskyblue", "tomato", "orchid"]
    return [base[m % len(base)] for m in range(num_stages) for _ in range(num_agents_per_stage)]


def draw_multipartite_graph(env, t: int, save_prefix: str):
    out_dir = _resolve_out_dir(save_prefix)
    num_stages = int(getattr(env, "num_stages", 0))
    num_agents_per_stage = int(getattr(env, "num_agents_per_stage", 0))
    sup_rel = getattr(env, "supply_relations", None)

    print(f"[VIZDBG] draw_multipartite_graph: stages={num_stages}, agents_per_stage={num_agents_per_stage}")
    try:
        if sup_rel is None:
            print("[VIZDBG] supply_relations is None")
        else:
            arr = np.asarray(sup_rel, dtype=object)
            print(f"[VIZDBG] supply_relations type={type(sup_rel)}, shape={getattr(arr, 'shape', None)}")
    except Exception as e:
        print("[VIZDBG] supply_relations shape check failed:", repr(e))

    M = nx.DiGraph()
    for m in range(num_stages):
        stage_nodes = [f"s{m}a{x}" for x in range(num_agents_per_stage)]
        M.add_nodes_from(stage_nodes, layer=num_stages - m)

    if sup_rel is not None:
        for m in range(num_stages - 1):
            for x in range(num_agents_per_stage):
                for i in range(num_agents_per_stage):
                    try:
                        if int(sup_rel[m][x][i]) == 1:
                            M.add_edge(f"s{m + 1}a{i}", f"s{m}a{x}")
                    except Exception:
                        continue

    pos = nx.multipartite_layout(M, subset_key="layer")
    colors = _stage_colors(num_stages, num_agents_per_stage)
    try:
        running = np.asarray(getattr(env, "running_agents"))
        for m in range(num_stages):
            for x in range(num_agents_per_stage):
                if running[m, x] == 0:
                    colors[m * num_agents_per_stage + x] = "black"
    except Exception:
        pass

    out_path = os.path.join(out_dir, f"supply_chain_period_{t}.png")
    try:
        plt.figure(figsize=(25, 20))
        nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
        plt.title("Multipartite Graph")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[viz][SAVE] multipartite -> {abspath(out_path)}; exists={exists(out_path)}")
    except Exception:
        print("[viz][ERR] multipartite save failed:\n" + traceback.format_exc())

def draw_material_flow(env, t: int, save_prefix: str):
    """
    Draw the realized arrivals between stages at period t, if available.
    Falls back to an empty graph if data is missing or out of range.
    This version adds debug logs and strong error handling.
    """
    out_dir = _resolve_out_dir(save_prefix)
    os.makedirs(out_dir, exist_ok=True)

    num_stages = int(getattr(env, "num_stages", 0))
    num_agents_per_stage = int(getattr(env, "num_agents_per_stage", 0))
    print(f"[VIZDBG] draw_material_flow: stages={num_stages}, agents_per_stage={num_agents_per_stage}, t_in={t}")

    # ---- Try accessing arriving_orders safely ----
    sup_rel_t = None
    tt = None
    try:
        ao = getattr(env, "arriving_orders", None)
        print(f"[VIZDBG] arriving_orders type={type(ao)}")
        if isinstance(ao, np.ndarray):
            print(f"[VIZDBG] arriving_orders shape={ao.shape}")  # (S, A, A, T+1)
            if ao.ndim == 4 and ao.shape[3] > 0:
                # clamp t into valid range
                tt = max(0, min(int(t), ao.shape[3] - 1))
                sup_rel_t = ao[:, :, :, tt]
                print(f"[VIZDBG] using arrivals slice at t_clamped={tt} with slice shape={sup_rel_t.shape}")
            else:
                print("[VIZDBG] arriving_orders is ndarray but ndim != 4 or last dim == 0")
        else:
            print("[VIZDBG] arriving_orders is not an ndarray or missing")
    except Exception:
        sup_rel_t = None
        print("[viz][ERR] arriving_orders access failed:\n" + traceback.format_exc())

    # ---- Build graph ----
    M = nx.DiGraph()

    # Nodes per stage
    for m in range(num_stages):
        stage_nodes = [f"s{m}a{x}" for x in range(num_agents_per_stage)]
        M.add_nodes_from(stage_nodes, layer=num_stages - m)

    # Edges with labels = arrival qty
    edges = []
    edge_labels = {}
    try:
        if sup_rel_t is not None:
            for m in range(num_stages - 1):
                for x in range(num_agents_per_stage):
                    for i in range(num_agents_per_stage):
                        try:
                            qty = int(sup_rel_t[m][x][i])
                        except Exception:
                            qty = 0
                        if qty > 0:
                            src = f"s{m + 1}a{i}"
                            tgt = f"s{m}a{x}"
                            edges.append((src, tgt))
                            edge_labels[(src, tgt)] = qty
        else:
            print("[VIZDBG] sup_rel_t is None; will draw an empty flow graph.")
    except Exception:
        print("[viz][ERR] building edges failed:\n" + traceback.format_exc())

    M.add_edges_from(edges)
    print(f"[VIZDBG] draw_material_flow: edges_built={len(edges)} (t_used={tt if tt is not None else 'N/A'})")

    # Layout & draw
    try:
        pos = nx.multipartite_layout(M, subset_key="layer")
    except Exception:
        print("[viz][ERR] multipartite_layout failed, fallback to spring_layout:\n" + traceback.format_exc())
        pos = nx.spring_layout(M, seed=0)

    colors = _stage_colors(num_stages, num_agents_per_stage)

    # Use PNG; verify on disk right after save
    out_path = os.path.join(out_dir, f"material_flow_period_{t}.png")
    try:
        plt.figure(figsize=(25, 20))
        nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
        if edge_labels:
            nx.draw_networkx_edge_labels(G=M, pos=pos, edge_labels=edge_labels)
        title_t = f"{tt}" if tt is not None else f"{t}"
        plt.title(f"Material Flow Graph (t={title_t})")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        ok = exists(out_path)
        size = os.path.getsize(out_path) if ok else -1
        print(f"[viz][SAVE] material_flow -> {abspath(out_path)}; exists={ok}; size={size} bytes")
    except Exception:
        print("[viz][ERR] material_flow save failed:\n" + traceback.format_exc())



def visualize_state(env, rewards: dict, t: int, save_prefix: str):
    """
    Dump per-agent snapshot CSV and draw two graphs for the given period t.
    Returns a dict containing saved paths.
    """
    out_dir = _resolve_out_dir(save_prefix)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[VIZDBG] visualize_state: t={t}, out_dir={abspath(out_dir)}")

    state_dict = getattr(env, "state_dict", {}) or {}
    num_stages = int(getattr(env, "num_stages", 0))
    num_agents_per_stage = int(getattr(env, "num_agents_per_stage", 0))
    print(f"[VIZDBG] visualize_state: state_keys={len(state_dict)}, stages={num_stages}, agents_per_stage={num_agents_per_stage}")

    cols = [
        "stage", "agent_idx", "profits", "prod_capacity", "inventory",
        "sales_price", "backlog_cost", "holding_cost", "backlog", "upstream_backlog",
        "suppliers", "customers", "order_cost", "prod_cost", "recent_sales",
        "lead_time", "deliveries"
    ]
    df = pd.DataFrame(columns=cols)

    try:
        for stage in range(num_stages):
            for agent in range(num_agents_per_stage):
                key = f"stage_{stage}_agent_{agent}"
                row = {
                    "stage": stage,
                    "agent_idx": agent,
                    "prod_capacity": None,
                    "sales_price": None,
                    "order_cost": None,
                    "backlog_cost": None,
                    "holding_cost": None,
                    "lead_time": None,
                    "inventory": None,
                    "backlog": None,
                    "upstream_backlog": None,
                    "suppliers": None,
                    "customers": None,
                    "recent_sales": None,
                    "deliveries": None,
                    "prod_cost": None,
                    "profits": rewards.get(key, None) if isinstance(rewards, dict) else None,
                }
                try:
                    s = state_dict[key]
                    row["prod_capacity"]   = s[0]
                    row["sales_price"]     = s[1]
                    row["order_cost"]      = s[2]
                    row["backlog_cost"]    = s[3]
                    row["holding_cost"]    = s[4]
                    row["lead_time"]       = s[5]
                    row["inventory"]       = s[6]
                    row["backlog"]         = s[7]
                    row["upstream_backlog"]= s[8]
                    row["suppliers"]       = s[9]
                    row["customers"]       = s[10]
                    row["recent_sales"]    = s[11]
                    row["deliveries"]      = s[12]
                    row["prod_cost"]       = s[13]
                except Exception:
                    pass
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except Exception:
        print("[viz][ERR] building dataframe failed:\n" + traceback.format_exc())

    try:
        df = df.groupby(by=["stage", "agent_idx"]).apply(lambda x: x).reset_index(drop=True)
    except Exception:
        print("[viz][WARN] groupby reset failed; continue without it")

    csv_path = os.path.join(out_dir, f"env_period_{t}.csv")
    try:
        df.to_csv(csv_path, index=False)
        ok = exists(csv_path)
        size = os.path.getsize(csv_path) if ok else -1
        print(f"[viz][SAVE] env_csv -> {abspath(csv_path)}; exists={ok}; size={size} bytes")
    except Exception:
        print("[viz][ERR] save csv failed:\n" + traceback.format_exc())

    imgs = []
    try:
        draw_multipartite_graph(env=env, t=t, save_prefix=save_prefix)
        imgs.append(os.path.join(out_dir, f"supply_chain_period_{t}.png"))
    except Exception:
        print("[viz][ERR] draw_multipartite_graph wrapper:\n" + traceback.format_exc())

    try:
        draw_material_flow(env=env, t=t, save_prefix=save_prefix)
        imgs.append(os.path.join(out_dir, f"material_flow_period_{t}.png"))
    except Exception:
        print("[viz][ERR] draw_material_flow wrapper:\n" + traceback.format_exc())

    return {"csv": csv_path, "imgs": imgs}



# ------------------------------
# Graph description helpers
# ------------------------------
def random_relations(n_cand: int, n_relation: int):
    return np.random.choice(a=n_cand, size=n_relation, replace=False)


def get_state_description(state: dict, past_req_orders: list, G: nx.Graph, state_format: str,
                          enable_graph_change: bool, agent_name: str = None):
    if state_format == "base":
        return get_base_description(state=state, past_req_orders=past_req_orders)
    elif state_format == "GraphML":
        return get_GraphML_description(G=G, agent_name=agent_name, enable_graph_change=enable_graph_change, state=state)
    else:
        raise AssertionError(f"{state_format} state description method not implemented yet")


def get_GraphML_description(agent_name: str, G: nx.DiGraph, enable_graph_change: bool, state: dict):
    # Select a subgraph around the agent
    if enable_graph_change:
        upstream_nodes = [up for up in G.successors(agent_name)
                          if G.nodes[up].get("stage") == G.nodes[agent_name].get("stage") + 1]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        connected_nodes = upstream_nodes + customer_nodes + [agent_name]
        sub_graph = G.subgraph(connected_nodes)
    else:
        supplier_nodes = [supplier for node, supplier in G.edges(agent_name) if G.edges[node, supplier].get("customer")]
        customer_nodes = [customer for node, customer in G.edges(agent_name) if G.edges[node, customer].get("supplier")]
        sub_graph = G.subgraph(supplier_nodes + customer_nodes + [agent_name])

    graphml_str = "\n".join(list(nx.generate_graphml(sub_graph, named_key_ids=True, prettyprint=True))[12:])
    recent_sales = f"\nPrevious Sales (in the recent round(s), from old to new): {state['sales']}\n"
    return graphml_str + recent_sales


def get_base_description(state, past_req_orders):
    suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state["suppliers"]) if state["suppliers"][i] == 1])
    non_suppliers = "; ".join([f"agent{i}" for i, _ in enumerate(state["suppliers"]) if state["suppliers"][i] == 0])
    lead_times = " round(s); ".join([f"from agent{i}: {state['lead_times'][i]}" for i, _ in enumerate(state["lead_times"])])
    order_costs = " unit(s); ".join([f"from agent{i}: {state['order_costs'][i]}" for i, _ in enumerate(state["order_costs"])])
    prod_cost = state["prod_cost"]

    arriving_deliveries = []
    for i, _ in enumerate(state["suppliers"]):
        if state["suppliers"][i] == 1:
            arriving_deliveries.append(f"from agent{i}: {state['deliveries'][i][-state['lead_times'][i]:]}")
    arriving_deliveries = "; ".join(arriving_deliveries)

    req_orders = []
    if len(past_req_orders) == 0:
        req_orders = "None"
    else:
        for i, _ in enumerate(past_req_orders):
            if past_req_orders[i] != 0:
                req_orders.append(f"from agent{i}: {past_req_orders[i]}")
        req_orders = " ".join(req_orders)

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
        f" - Arriving Deliveries (in this and the next round(s), from near to far): {arriving_deliveries}\n"
        f" - Your upstream suppliers are: {suppliers}\n"
        f" - Other available upstream suppliers in the environment are: {non_suppliers}\n"
    )


def get_demand_description(demand_fn):
    try:
        from data_simulation import Demand_fn as _DF
    except Exception:
        _DF = None

    if _DF is not None and isinstance(demand_fn, _DF):
        name = demand_fn.dist
        if name == "uniform_demand":
            return f"uniform_demand(lb={demand_fn.lb}, ub={demand_fn.ub})"
        if name == "normal_demand":
            return f"normal_demand(mean={demand_fn.mean}, std={demand_fn.std})"
        if name == "constant_demand":
            return f"constant_demand(mean={demand_fn.mean})"
        if name == "poisson_demand":
            return f"poisson_demand(mean={demand_fn.mean})"
        if name == "dyn_poisson_demand":
            return f"dyn_poisson_demand(base_mean={demand_fn.mean})"
        if name == "sin_demand":
            return (
                f"sin_demand(mean={demand_fn.mean}, amplitude={demand_fn.amp}, "
                f"period={demand_fn.period_len}, phase={demand_fn.phase}, "
                f"noise_std={demand_fn.noise_std})"
            )
        if name == "cos_demand":
            return (
                f"cos_demand(mean={demand_fn.mean}, amplitude={demand_fn.amp}, "
                f"period={demand_fn.period_len}, phase={demand_fn.phase}, "
                f"noise_std={demand_fn.noise_std})"
            )
        raise KeyError(f"Error: {name} not implemented.")

    if isinstance(demand_fn, str):
        return demand_fn

    if isinstance(demand_fn, (list, tuple)) and len(demand_fn) >= 1:
        name = demand_fn[0]
        args = demand_fn[1:]
        if name == "uniform_demand" and len(args) == 2:
            lb, ub = args
            return f"uniform_demand(lb={lb}, ub={ub})"
        if name == "normal_demand" and len(args) == 2:
            mean, std = args
            return f"normal_demand(mean={mean}, std={std})"
        if name == "constant_demand" and len(args) == 1:
            (mean,) = args
            return f"constant_demand(mean={mean})"
        if name == "poisson_demand" and len(args) == 1:
            (mean,) = args
            return f"poisson_demand(mean={mean})"
        if name == "dyn_poisson_demand" and len(args) == 1:
            (mean,) = args
            return f"dyn_poisson_demand(base_mean={mean})"
        if name in ("sin_demand", "cos_demand") and len(args) == 5:
            mean, amp, period, phase, noise = args
            return f"{name}(mean={mean}, amplitude={amp}, period={period}, phase={phase}, noise_std={noise})"
        return str(demand_fn)

    raise KeyError(f"Error: {demand_fn} not implemented.")


def update_sup_action(sup_action: list, rm_match: str, add_match: str):
    remove_sup = rm_match.replace(" ", "")
    if remove_sup != "":
        remove_sup = remove_sup.replace("agent", "").replace('"', "")
        try:
            remove_sup = [int(ind) for ind in remove_sup.split(",")]
            for ind in remove_sup:
                sup_action[ind] = 0
        except Exception:
            pass
    add_sup = add_match.replace(" ", "")
    if add_sup != "":
        add_sup = add_sup.replace("agent", "").replace('"', "")
        try:
            add_sup = [int(ind) for ind in add_sup.split(",")]
            for ind in add_sup:
                sup_action[ind] = 1
        except Exception:
            pass
    return sup_action


def parse_order_list(raw_str: str, num_suppliers: int, default_key_prefix: str = "agent") -> np.ndarray:
    kv_pairs = re.findall(r'"?agent[_ ]?(\d+)"?\s*[:=]\s*([0-9]+)', raw_str)
    if kv_pairs:
        vec = np.zeros(num_suppliers, dtype=int)
        for idx, qty in kv_pairs:
            idx = int(idx)
            if idx < num_suppliers:
                vec[idx] = int(qty)
        return vec

    m = re.search(r"\[(.*?)\]", raw_str, re.DOTALL)
    nums = re.findall(r"\d+", m.group(1)) if m else re.findall(r"\d+", raw_str)

    if nums:
        nums = [int(x) for x in nums]
        nums = (nums + [0] * num_suppliers)[:num_suppliers]
        return np.array(nums, dtype=int)

    # Fallback
    return np.zeros(num_suppliers, dtype=int)

    