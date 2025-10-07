import os
import re
import json
import traceback
from os.path import abspath, exists
from typing import List, Callable
from pathlib import Path
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

# def draw_multipartite_graph(env, t: int, save_prefix: str):
#     out_dir = _resolve_out_dir(save_prefix)
#     num_stages = int(getattr(env, "num_stages", 0))
#     num_agents_per_stage = int(getattr(env, "num_agents_per_stage", 0))
#     sup_rel = getattr(env, "supply_relations", None)
#
#     print(f"[VIZDBG] draw_multipartite_graph: stages={num_stages}, agents_per_stage={num_agents_per_stage}")
#     try:
#         if sup_rel is None:
#             print("[VIZDBG] supply_relations is None")
#         else:
#             arr = np.asarray(sup_rel, dtype=object)
#             print(f"[VIZDBG] supply_relations type={type(sup_rel)}, shape={getattr(arr, 'shape', None)}")
#     except Exception as e:
#         print("[VIZDBG] supply_relations shape check failed:", repr(e))
#
#     M = nx.DiGraph()
#     for m in range(num_stages):
#         stage_nodes = [f"s{m}a{x}" for x in range(num_agents_per_stage)]
#         M.add_nodes_from(stage_nodes, layer=num_stages - m)
#
#     if sup_rel is not None:
#         for m in range(num_stages - 1):
#             for x in range(num_agents_per_stage):
#                 for i in range(num_agents_per_stage):
#                     try:
#                         if int(sup_rel[m][x][i]) == 1:
#                             M.add_edge(f"s{m + 1}a{i}", f"s{m}a{x}")
#                     except Exception:
#                         continue
#
#     pos = nx.multipartite_layout(M, subset_key="layer")
#     colors = _stage_colors(num_stages, num_agents_per_stage)
#     try:
#         running = np.asarray(getattr(env, "running_agents"))
#         for m in range(num_stages):
#             for x in range(num_agents_per_stage):
#                 if running[m, x] == 0:
#                     colors[m * num_agents_per_stage + x] = "black"
#     except Exception:
#         pass
#
#     out_path = os.path.join(out_dir, f"supply_chain_period_{t}.png")
#     try:
#         plt.figure(figsize=(25, 20))
#         nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
#         plt.title("Multipartite Graph")
#         plt.tight_layout()
#         plt.savefig(out_path)
#         plt.close()
#         print(f"[viz][SAVE] multipartite -> {abspath(out_path)}; exists={exists(out_path)}")
#     except Exception:
#         print("[viz][ERR] multipartite save failed:\n" + traceback.format_exc())

# def draw_material_flow(env, t: int, save_prefix: str):
#     """
#     Draw the realized arrivals between stages at period t, if available.
#     Falls back to an empty graph if data is missing or out of range.
#     This version adds debug logs and strong error handling.
#     """
#     out_dir = _resolve_out_dir(save_prefix)
#     os.makedirs(out_dir, exist_ok=True)
#
#     num_stages = int(getattr(env, "num_stages", 0))
#     num_agents_per_stage = int(getattr(env, "num_agents_per_stage", 0))
#     print(f"[VIZDBG] draw_material_flow: stages={num_stages}, agents_per_stage={num_agents_per_stage}, t_in={t}")
#
#     # ---- Try accessing arriving_orders safely ----
#     sup_rel_t = None
#     tt = None
#     try:
#         ao = getattr(env, "arriving_orders", None)
#         print(f"[VIZDBG] arriving_orders type={type(ao)}")
#         if isinstance(ao, np.ndarray):
#             print(f"[VIZDBG] arriving_orders shape={ao.shape}")  # (S, A, A, T+1)
#             if ao.ndim == 4 and ao.shape[3] > 0:
#                 # clamp t into valid range
#                 tt = max(0, min(int(t), ao.shape[3] - 1))
#                 sup_rel_t = ao[:, :, :, tt]
#                 print(f"[VIZDBG] using arrivals slice at t_clamped={tt} with slice shape={sup_rel_t.shape}")
#             else:
#                 print("[VIZDBG] arriving_orders is ndarray but ndim != 4 or last dim == 0")
#         else:
#             print("[VIZDBG] arriving_orders is not an ndarray or missing")
#     except Exception:
#         sup_rel_t = None
#         print("[viz][ERR] arriving_orders access failed:\n" + traceback.format_exc())
#
#     # ---- Build graph ----
#     M = nx.DiGraph()
#
#     # Nodes per stage
#     for m in range(num_stages):
#         stage_nodes = [f"s{m}a{x}" for x in range(num_agents_per_stage)]
#         M.add_nodes_from(stage_nodes, layer=num_stages - m)
#
#     # Edges with labels = arrival qty
#     edges = []
#     edge_labels = {}
#     try:
#         if sup_rel_t is not None:
#             for m in range(num_stages - 1):
#                 for x in range(num_agents_per_stage):
#                     for i in range(num_agents_per_stage):
#                         try:
#                             qty = int(sup_rel_t[m][x][i])
#                         except Exception:
#                             qty = 0
#                         if qty > 0:
#                             src = f"s{m + 1}a{i}"
#                             tgt = f"s{m}a{x}"
#                             edges.append((src, tgt))
#                             edge_labels[(src, tgt)] = qty
#         else:
#             print("[VIZDBG] sup_rel_t is None; will draw an empty flow graph.")
#     except Exception:
#         print("[viz][ERR] building edges failed:\n" + traceback.format_exc())
#
#     M.add_edges_from(edges)
#     print(f"[VIZDBG] draw_material_flow: edges_built={len(edges)} (t_used={tt if tt is not None else 'N/A'})")
#
#     # Layout & draw
#     try:
#         pos = nx.multipartite_layout(M, subset_key="layer")
#     except Exception:
#         print("[viz][ERR] multipartite_layout failed, fallback to spring_layout:\n" + traceback.format_exc())
#         pos = nx.spring_layout(M, seed=0)
#
#     colors = _stage_colors(num_stages, num_agents_per_stage)
#
#     # Use PNG; verify on disk right after save
#     out_path = os.path.join(out_dir, f"material_flow_period_{t}.png")
#     try:
#         plt.figure(figsize=(25, 20))
#         nx.draw(M, pos, with_labels=True, node_color=colors, node_size=100, font_size=12, edge_color="gray", alpha=1)
#         if edge_labels:
#             nx.draw_networkx_edge_labels(G=M, pos=pos, edge_labels=edge_labels)
#         title_t = f"{tt}" if tt is not None else f"{t}"
#         plt.title(f"Material Flow Graph (t={title_t})")
#         plt.tight_layout()
#         plt.savefig(out_path)
#         plt.close()
#         ok = exists(out_path)
#         size = os.path.getsize(out_path) if ok else -1
#         print(f"[viz][SAVE] material_flow -> {abspath(out_path)}; exists={ok}; size={size} bytes")
#     except Exception:
#         print("[viz][ERR] material_flow save failed:\n" + traceback.format_exc())



def visualize_state(env, rewards: dict, t: int, save_prefix: str):
    """
    Save per-agent snapshot CSV for period t.
    This function no longer renders legacy full-structure graphs.
    Returns: {"csv": <csv_path>}
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
                    row["prod_capacity"]    = s[0]
                    row["sales_price"]      = s[1]
                    row["order_cost"]       = s[2]
                    row["backlog_cost"]     = s[3]
                    row["holding_cost"]     = s[4]
                    row["lead_time"]        = s[5]
                    row["inventory"]        = s[6]
                    row["backlog"]          = s[7]
                    row["upstream_backlog"] = s[8]
                    row["suppliers"]        = s[9]
                    row["customers"]        = s[10]
                    row["recent_sales"]     = s[11]
                    row["deliveries"]       = s[12]
                    row["prod_cost"]        = s[13]
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

    # No legacy full-graph rendering here anymore.
    return {"csv": csv_path}



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


def plot_episode_stage_charts(ep_dir: str | Path, dpi: int = 200):
    ep_dir = Path(ep_dir).resolve()
    csv_path = ep_dir / "records" / "demand_inventory_backlog_assets.csv"
    if not csv_path.exists():
        print(f"[SKIP] {ep_dir.name}: CSV not found {csv_path}")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[SKIP] matplotlib import failed: {e}")
        return

    df = pd.read_csv(csv_path)
    needed = ["period", "stage", "agent", "order", "demand"]
    for c in needed:
        if c not in df.columns:
            print(f"[SKIP] {ep_dir.name}: missing column {c}")
            return
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["period","stage","agent"]).copy()

    stages = sorted(df["stage"].unique())
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown"]

    for s in stages:
        g = df[df["stage"] == s].groupby("period", as_index=False).agg(
            order=("order","sum"),
            demand=("demand","sum"),
        )
        plt.figure(figsize=(12,6))
        plt.plot(g["period"], g["order"], "-", lw=2, color=colors[s % len(colors)], label=f"Stage {s} Order")
        if s == 0:
            plt.plot(g["period"], g["demand"], "--", lw=2, color="black", label=f"Stage {s} Demand")
        else:
            plt.plot(g["period"], g["demand"], "--", lw=2, color="black", alpha=0.7,
                     label=f"Stage {s} Demand (from downstream orders)")
        plt.xlabel("Period"); plt.ylabel("Quantity")
        plt.title(f"{ep_dir.name} | Stage {s} (aggregated)")
        plt.grid(alpha=0.3); plt.legend()
        out_path = ep_dir / f"stage{s}_agg.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=dpi); plt.close()
        print(f"[Saved] {out_path}")

        g_stage = df[df["stage"] == s]
        agents = sorted(g_stage["agent"].unique())
        fig, axes = plt.subplots(len(agents), 1, figsize=(12, 3*len(agents)), sharex=True)
        if len(agents) == 1: axes = [axes]
        for j, a in enumerate(agents):
            g_sa = g_stage[g_stage["agent"] == a]
            axes[j].plot(g_sa["period"], g_sa["order"], "-", lw=1.8, label=f"Agent {a} Order")
            if s == 0:
                axes[j].plot(g_sa["period"], g_sa["demand"], "--", lw=1.5, alpha=0.7, label=f"Agent {a} Demand")
            axes[j].set_title(f"Stage {s} Agent {a}")
            axes[j].grid(alpha=0.3); axes[j].legend(fontsize=8)
        plt.suptitle(f"{ep_dir.name} | Stage {s} (per-agent)", fontsize=14)
        out_path = ep_dir / f"stage{s}_peragent.png"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")

def _safe_episode_orders_csv(ep_dir: str | Path) -> Path:
    """
    Return path to the per-edge orders CSV for an episode, or raise FileNotFoundError.
    """
    ep_dir = Path(ep_dir).resolve()
    csv_path = ep_dir / "records" / "orders_per_edge.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"orders_per_edge.csv not found at: {csv_path}")
    return csv_path

def _load_orders_df(ep_dir: str | Path) -> pd.DataFrame:
    """
    Load <ep_dir>/records/orders_per_edge.csv and standardize to columns
    (buyer -> supplier):
        period, from_stage, from_agent, to_stage, to_agent, order

    Supported input schemas:

    A) Compact (current writer):
       ['period','stage','agent','to_supplier','qty']
       Meaning: buyer=(stage,agent); supplier=(stage+1,to_supplier); order=qty.

    B) Expanded (already standardized but direction may vary):
       ['period','from_stage','from_agent','to_stage','to_agent','order']
       If it is supplier->buyer (from = upstream = to+1), we flip to buyer->supplier.

    Returns
    -------
    pd.DataFrame with int dtype for all required columns, filtered to:
      - cross-stage edges where to_stage == from_stage + 1
      - positive order quantities (order > 0)
    """
    ep_dir = Path(ep_dir).resolve()
    csv_path = ep_dir / "records" / "orders_per_edge.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"orders_per_edge.csv not found at: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # ---- Case A: compact writer schema
    if {"period","stage","agent","to_supplier","qty"}.issubset(df_raw.columns):
        df = pd.DataFrame({
            "period":      pd.to_numeric(df_raw["period"], errors="coerce"),
            "from_stage":  pd.to_numeric(df_raw["stage"],  errors="coerce"),       # buyer (downstream)
            "from_agent":  pd.to_numeric(df_raw["agent"],  errors="coerce"),
            "to_stage":    pd.to_numeric(df_raw["stage"],  errors="coerce") + 1,   # supplier (upstream)
            "to_agent":    pd.to_numeric(df_raw["to_supplier"], errors="coerce"),
            "order":       pd.to_numeric(df_raw["qty"], errors="coerce"),
        })

    # ---- Case B: expanded schema (possibly wrong direction)
    elif {"period","from_stage","from_agent","to_stage","to_agent","order"}.issubset(df_raw.columns):
        df = df_raw[["period","from_stage","from_agent","to_stage","to_agent","order"]].copy()
        for c in ["period","from_stage","from_agent","to_stage","to_agent","order"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Flip only if it is supplier->buyer: from_stage == to_stage + 1
        mask_supplier_to_buyer = (df["from_stage"] == df["to_stage"] + 1)
        if mask_supplier_to_buyer.any():
            fs = df.loc[mask_supplier_to_buyer, "from_stage"].copy()
            fa = df.loc[mask_supplier_to_buyer, "from_agent"].copy()
            df.loc[mask_supplier_to_buyer, "from_stage"] = df.loc[mask_supplier_to_buyer, "to_stage"]
            df.loc[mask_supplier_to_buyer, "from_agent"] = df.loc[mask_supplier_to_buyer, "to_agent"]
            df.loc[mask_supplier_to_buyer, "to_stage"]   = fs
            df.loc[mask_supplier_to_buyer, "to_agent"]   = fa
    else:
        raise KeyError(
            "orders_per_edge.csv missing required columns. "
            "Supported schemas:\n"
            "  A) period,stage,agent,to_supplier,qty\n"
            "  B) period,from_stage,from_agent,to_stage,to_agent,order\n"
            f"Got columns: {list(df_raw.columns)}"
        )

    # Cast, drop invalids
    df = df.dropna(subset=["period","from_stage","from_agent","to_stage","to_agent","order"]).astype(int)

    # Keep only cross-stage buyer->immediate-upstream edges
    df = df[df["to_stage"] == df["from_stage"] + 1].copy()

    # Keep only positive orders
    df = df[df["order"] > 0].copy()

    return df


# ---------------------------------------------------------------------
# 2) Plot one period: draw buyer->supplier edges
# ---------------------------------------------------------------------
def render_txn_graph_for_period_from_csv(
    ep_dir: str | Path,
    t: int,
    dpi: int = 200,
) -> Path | None:
    """
    Render a directed graph of orders for period t with edges:
        buyer (downstream) -> supplier (upstream)

    Output: <ep_dir>/viz/txn_period_<t>.png

    Returns the image path if any edge exists in that period; otherwise None.
    """
    ep_dir = Path(ep_dir).resolve()
    out_dir = ep_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"txn_period_{t}.png"

    try:
        df = _load_orders_df(ep_dir)
    except Exception as e:
        print(f"[SKIP] render_txn_graph_for_period_from_csv: {e}")
        return None

    dft = df[(df["period"] == int(t)) & (df["order"] > 0)].copy()
    if dft.empty:
        print(f"[TXN] period={t}: no positive orders; skip figure.")
        return None

    # Aggregate multiple orders between the same pair within the same period
    dft_agg = (
        dft.groupby(["from_stage", "from_agent", "to_stage", "to_agent"], as_index=False)
           .agg(order=("order", "sum"))
    )

    # Build graph: buyer -> supplier
    G = nx.DiGraph()
    for _, row in dft_agg.iterrows():
        buyer_s,  buyer_a  = int(row.from_stage), int(row.from_agent)
        supply_s, supply_a = int(row.to_stage),   int(row.to_agent)
        q = int(row.order)

        buyer    = f"s{buyer_s}a{buyer_a}"
        supplier = f"s{supply_s}a{supply_a}"

        if not G.has_node(buyer):
            # layer = stage keeps lower stage to the left in multipartite layout
            G.add_node(buyer, layer=buyer_s, stage=buyer_s, agent=buyer_a)
        if not G.has_node(supplier):
            G.add_node(supplier, layer=supply_s, stage=supply_s, agent=supply_a)

        G.add_edge(buyer, supplier, weight=q)

    # Layout
    try:
        pos = nx.multipartite_layout(G, subset_key="layer")
    except Exception:
        pos = nx.spring_layout(G, seed=0)

    # Style
    palette = ["gold", "violet", "limegreen", "deepskyblue", "tomato", "orchid", "cadetblue"]
    node_colors = [palette[G.nodes[n]["stage"] % len(palette)] for n in G.nodes()]
    widths = [1.0 + 0.02 * G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(18, 12))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=280,
        font_size=10,
        edge_color="gray",
        width=widths,
        alpha=0.95,
        arrows=True,
        arrowsize=16,
    )
    edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title(f"Orders (downstream → upstream), period={t}")
    plt.savefig(img_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {img_path}")
    return img_path


# ---------------------------------------------------------------------
# 3) Plot all periods
# ---------------------------------------------------------------------
def render_all_period_graphs_from_csv(
    ep_dir: str | Path,
    dpi: int = 200,
) -> list[Path]:
    """
    For a given episode directory, generate a PNG per period with any positive order.
    Returns the list of image paths that were created.
    """
    ep_dir = Path(ep_dir).resolve()
    imgs = []
    try:
        df = _load_orders_df(ep_dir)
    except Exception as e:
        print(f"[SKIP] render_all_period_graphs_from_csv: {e}")
        return imgs

    periods = sorted(df["period"].unique())
    for t in periods:
        p = render_txn_graph_for_period_from_csv(ep_dir, int(t), dpi=dpi)
        if p is not None:
            imgs.append(p)
    return imgs


# ---------------------------------------------------------------------
# 4) Degree stats over the whole episode (buyer->supplier edges)
# ---------------------------------------------------------------------
def compute_and_save_degree_stats_from_txn(
    ep_dir: str | Path,
    dpi: int = 200,
) -> Path:
    """
    Build a directed graph with edges BUYER -> SUPPLIER (orders) across ALL periods,
    then compute degree statistics per node.

    Meanings:
      - out_degree_*  : for buyers, how many suppliers they ordered from (and how much).
      - in_degree_*   : for suppliers, how many buyers ordered from them (and how much).

    Saves:
      <ep_dir>/degree_stats_txn.csv
      <ep_dir>/degree_hist_txn.png  (histogram of total_degree_binary)
    """
    ep_dir = Path(ep_dir).resolve()
    out_csv = ep_dir / "degree_stats_txn.csv"
    out_png = ep_dir / "degree_hist_txn.png"

    df = _load_orders_df(ep_dir)
    if df.empty:
        pd.DataFrame(columns=[
            "node","stage","agent_idx",
            "in_degree_binary","out_degree_binary","total_degree_binary",
            "in_degree_weighted","out_degree_weighted","total_degree_weighted",
        ]).to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv} (empty)")
        return out_csv

    # Build episode-wide graph (accumulate weights)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        buyer_s,  buyer_a  = int(r["from_stage"]), int(r["from_agent"])
        supply_s, supply_a = int(r["to_stage"]),   int(r["to_agent"])
        q = int(r["order"])

        buyer    = f"s{buyer_s}a{buyer_a}"
        supplier = f"s{supply_s}a{supply_a}"

        if not G.has_node(buyer):
            G.add_node(buyer, stage=buyer_s, agent_idx=buyer_a)
        if not G.has_node(supplier):
            G.add_node(supplier, stage=supply_s, agent_idx=supply_a)

        if G.has_edge(buyer, supplier):
            G[buyer][supplier]["weight"] += q
        else:
            G.add_edge(buyer, supplier, weight=q)

    # Compute degrees
    rows = []
    for n in sorted(G.nodes()):
        stage = G.nodes[n]["stage"]
        agent = G.nodes[n]["agent_idx"]
        in_bin = G.in_degree(n)
        out_bin = G.out_degree(n)
        in_w = sum(G[u][n]["weight"] for u in G.predecessors(n))
        out_w = sum(G[n][v]["weight"] for v in G.successors(n))
        rows.append({
            "node": n,
            "stage": stage,
            "agent_idx": agent,
            "in_degree_binary": in_bin,
            "out_degree_binary": out_bin,
            "total_degree_binary": in_bin + out_bin,
            "in_degree_weighted": in_w,
            "out_degree_weighted": out_w,
            "total_degree_weighted": in_w + out_w,
        })

    out_df = pd.DataFrame(rows).sort_values(["stage","agent_idx"]).reset_index(drop=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # Optional: histogram of binary total degree
    try:
        plt.figure(figsize=(8,5))
        plt.hist(out_df["total_degree_binary"].values, bins=20)
        plt.xlabel("Binary total degree")
        plt.ylabel("Num nodes")
        plt.title("Degree distribution (buyer→supplier edges)")
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_png}")
    except Exception as e:
        print(f"[WARN] degree histogram failed: {e}")

    return out_csv
def list_order_split_violations(ep_dir: str | Path, max_suppliers: int = 3) -> pd.DataFrame:
    """
    List per-period violations where a buyer placed positive orders to more than `max_suppliers` upstream suppliers.
    Assumes _load_orders_df() returns standardized buyer->supplier edges:
      columns: period, from_stage, from_agent, to_stage, to_agent, order
      with only cross-stage, positive orders kept.
    """
    ep_dir = Path(ep_dir).resolve()
    try:
        df = _load_orders_df(ep_dir)
    except Exception as e:
        print(f'[SKIP] list_order_split_violations: {e}')
        return pd.DataFrame()

    # Count distinct suppliers per (period, buyer)
    grp = (
        df.groupby(["period", "from_stage", "from_agent"])
          .agg(
              suppliers_nunique=("to_agent", "nunique"),
              total_qty=("order", "sum")
          )
          .reset_index()
    )
    viol = grp[grp["suppliers_nunique"] > int(max_suppliers)].copy()
    if viol.empty:
        print(f"[RULE] No violations found (>{max_suppliers} suppliers).")
        return viol

    # Optional: save
    out_csv = ep_dir / f"records/split_violations_gt{max_suppliers}.csv"
    viol.to_csv(out_csv, index=False)
    print(f"[RULE] Violations saved -> {out_csv}")

    return viol


def summarize_order_split_by_buyer(ep_dir: str | Path) -> pd.DataFrame:
    """
    Episode-level summary: for each buyer, how many distinct suppliers they used in the whole episode
    and how many periods they exceeded various thresholds (2, 3, 4...).
    """
    ep_dir = Path(ep_dir).resolve()
    try:
        df = _load_orders_df(ep_dir)
    except Exception as e:
        print(f'[SKIP] summarize_order_split_by_buyer: {e}')
        return pd.DataFrame()

    # per-period distinct suppliers
    per_period = (
        df.groupby(["period", "from_stage", "from_agent"])
          .agg(suppliers_nunique=("to_agent", "nunique"), total_qty=("order", "sum"))
          .reset_index()
    )

    # episode-level distinct suppliers
    episode_distinct = (
        df.groupby(["from_stage", "from_agent"])
          .agg(
              suppliers_distinct=("to_agent", "nunique"),
              periods_active=("period", "nunique"),
              total_qty=("order", "sum")
          )
          .reset_index()
    )

    # how many periods buyer exceeded thresholds
    for k in [2, 3, 4]:
        col = f"periods_gt_{k}"
        tmp = (
            per_period[per_period["suppliers_nunique"] > k]
            .groupby(["from_stage", "from_agent"])
            .size()
            .rename(col)
        )
        episode_distinct = episode_distinct.merge(tmp, on=["from_stage", "from_agent"], how="left")

    episode_distinct = episode_distinct.fillna(0).astype({
        "periods_active": int,
        "suppliers_distinct": int,
        "total_qty": int,
        "periods_gt_2": int,
        "periods_gt_3": int,
        "periods_gt_4": int
    })

    out_csv = ep_dir / "records/order_split_summary_by_buyer.csv"
    episode_distinct.to_csv(out_csv, index=False)
    print(f"[RULE] Buyer-level split summary saved -> {out_csv}")

    return episode_distinct