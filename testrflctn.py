#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
testrflctn.py — Validate the reflection pipeline outside the main simulation.

Usage:
  python testrflctn.py \
    --csv results/records/demand_inventory_backlog_assets.csv \
    --save_dir results/test_memory \
    --reflect_every 9 --min_obs 6
"""

import argparse, os, sys, json, re
import pandas as pd
import numpy as np

# ---------- helper: locate src/ for memory_module ----------
def add_src_to_path():
    candidates = ["src", ".", "code", "scripts"]
    for c in candidates:
        p = os.path.join(c, "memory_module.py")
        if os.path.exists(p):
            sys.path.insert(0, c)
            return c
    return None

# ---------- a mock LLM that always returns well-formed JSON rules ----------
def mock_llm(prompt: str) -> str:
    """
    Parse the 'Data: [...]' JSON embedded in the prompt, compute simple stats,
    and return a well-formed JSON rules object. This tests the *parsing & save*
    paths in ExperienceMemory.reflect, independent of real LLM behavior.
    """
    try:
        m = re.search(r"Data:\s*(\[[\s\S]*\])", prompt)
        data = json.loads(m.group(1)) if m else []
    except Exception:
        data = []

    try:
        demand = np.array([d["state"]["demand"] for d in data], float)
        assets = np.array([d["state"]["assets"] for d in data], float)
        backlog = np.array([d["state"]["backlog"] for d in data], float)
        inventory = np.array([d["state"]["inventory"] for d in data], float)
        order = np.array([d["order"] for d in data], float)
        reward = np.array([d["reward"] for d in data], float)
    except Exception:
        demand = assets = backlog = inventory = order = reward = np.array([])

    rules = []
    if demand.size > 0:
        mean_d = float(np.mean(demand))
        mean_b = float(np.mean(backlog)) if backlog.size else 0.0
        mean_i = float(np.mean(inventory)) if inventory.size else 0.0

        asset_trend = ""
        if assets.size >= 2:
            asset_trend = "shrinking" if assets[-1] < assets[0] else "growing"

        rules.append({
            "because": f"Typical demand around {mean_d:.1f}, backlog≈{mean_b:.1f}, inventory≈{mean_i:.1f}, assets {asset_trend}.",
            "then": "Order toward demand plus a fraction of backlog when backlog exceeds inventory; avoid multiple zero-order rounds.",
            "note": "Raise orders gradually to avoid bullwhip; do not exceed assets constraint."
        })

        # If backlog rising, add a second rule
        if backlog.size >= 3 and (backlog[-1] > backlog[0] and (backlog[-1] - backlog[0]) > 0.1 * max(1.0, backlog[0])):
            rules.append({
                "because": "Backlog shows an upward trend in the recent window.",
                "then": "Increase orders by demand + 0.5*backlog - inventory (clamped to >=0).",
                "note": ""
            })

    if not rules:
        rules = [{
            "because": "Fallback",
            "then": "Keep orders close to recent demand; prevent stockout by ordering small amounts.",
            "note": ""
        }]

    return json.dumps({"rules": rules}, ensure_ascii=False)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to demand_inventory_backlog_assets.csv")
    parser.add_argument("--save_dir", default="results/test_memory", help="Directory to write rules.json + reflection_raw/")
    parser.add_argument("--reflect_every", type=int, default=9)
    parser.add_argument("--min_obs", type=int, default=6)
    args = parser.parse_args()

    added = add_src_to_path()
    if added is None:
        print("[WARN] Could not locate memory_module.py automatically. Ensure it's importable (in PYTHONPATH).")

    try:
        from memory_module import ExperienceMemory, MemoryConfig
    except Exception as e:
        print("[ERROR] Failed to import memory_module.ExperienceMemory:", e)
        print("       Make sure this script is run from your repo root, or adjust sys.path.")
        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "reflection_raw"), exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"period","stage","agent","demand","inventory","backlog","assets","order"}
    miss = required_cols - set(df.columns)
    if miss:
        print("[ERROR] CSV missing columns:", miss)
        sys.exit(1)

    # Infer stage/agent counts
    S = int(df["stage"].max()) + 1
    A = int(df["agent"].max()) + 1
    print(f"[INFO] Loaded CSV with {len(df)} rows; stages={S}, agents_per_stage={A}")

    mem = ExperienceMemory(
        num_stages=S,
        num_agents_per_stage=A,
        config=MemoryConfig(
            reflect_every=args.reflect_every,
            min_obs_for_reflection=args.min_obs,
            save_dir=args.save_dir,
        )
    )

    # Populate memory with observations
    df_sorted = df.sort_values(["period","stage","agent"]).reset_index(drop=True)
    last_period = int(df_sorted["period"].max())

    for _, row in df_sorted.iterrows():
        kwargs = dict(
            period=int(row["period"]),
            s=int(row["stage"]),
            a=int(row["agent"]),
            demand=float(row["demand"]),
            assets=float(row["assets"]),
            backlog=float(row["backlog"]),
            inventory=float(row["inventory"]),
            order=float(row["order"]),
            reward=0.0,
            order_vec=None,
        )
        # Be tolerant to different signatures
        try:
            mem.add_observation(**kwargs)
        except TypeError:
            kwargs.pop("order_vec", None)
            mem.add_observation(**kwargs)

    # Run reflection per agent (forced) using last available window
    total_rules = 0
    for s in range(S):
        for a in range(A):
            obs_cnt = len(mem.store[(s, a)])
            if obs_cnt < max(3, args.min_obs):
                print(f"[SKIP] s={s} a={a}: only {obs_cnt} observations (< {max(3, args.min_obs)})")
                continue
            try:
                new_rules = mem.reflect(
                    s=s, a=a,
                    llm_fn=mock_llm,
                    period=last_period,
                    reason="forced_test",
                    window_n=min(9, obs_cnt),
                )
                print(f"[REFLECT] s={s} a={a}: produced {len(new_rules)} rule(s)")
                total_rules += len(new_rules)
            except TypeError:
                # Fallback if signature differs
                new_rules = mem.reflect(s, a, mock_llm, last_period, "forced_test", min(9, obs_cnt))
                print(f"[REFLECT] s={s} a={a}: produced {len(new_rules)} rule(s) [positional]")
                total_rules += len(new_rules)
            except Exception as e:
                print(f"[ERR] reflect s={s} a={a}:", e)

    # Save memory (rules.json + mem_*.csv)
    try:
        mem.save()
    except Exception as e:
        print("[WARN] mem.save() failed:", e)

    rules_path = os.path.join(args.save_dir, "rules.json")
    print(f"[DONE] Saved rules to: {rules_path}")
    if os.path.exists(rules_path):
        try:
            obj = json.load(open(rules_path, "r", encoding="utf-8"))
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1200])
        except Exception as e:
            print("[WARN] Could not pretty-print rules.json:", e)

    print(f"[NOTE] Raw reflection logs (prompts/outputs) are in: {os.path.join(args.save_dir, 'reflection_raw')}")

if __name__ == "__main__":
    main()
