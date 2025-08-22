# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional, Callable
import os
import json
import math
import numpy as np
import pandas as pd
import re


@dataclass
class MemoryConfig:
    reflect_every: int = 9
    min_obs_for_reflection: int = 9
    asset_drop_window: int = 3
    asset_drop_pct_per_step: float = 0.12
    asset_total_drop_pct: float = 0.28
    backlog_rise_window: int = 3
    backlog_rise_pct_per_step: float = 0.10
    backlog_min_level: float = 5.0

    k_neighbors: int = 5
    distance_eps: float = 1e-6
    use_dims: Tuple[int, int, int, int] = (0, 1, 2, 3)

    max_rules_per_agent: int = 50
    rule_anchor_std_multiplier: float = 1.0
    max_rules_in_prompt: int = 3

    save_dir: str = "results/memory"


@dataclass
class Observation:
    period: int
    demand: float
    assets: float
    backlog: float
    inventory: float
    order: float
    reward: float
    order_vec: Optional[List[int]] = None

    def state_vector(self) -> np.ndarray:
        return np.array([self.demand, self.assets, self.backlog, self.inventory], dtype=float)


@dataclass
class Rule:
    rule_id: str
    period: int
    stage: int
    agent: int
    reason: str
    insights: List[str]
    anchor_mean: List[float]
    anchor_std: List[float]
    window_periods: List[int]
    meta: Dict[str, Any]

    def anchor_vector(self) -> np.ndarray:
        return np.array(self.anchor_mean, dtype=float)

    def distance(self, q: np.ndarray, dims: Tuple[int, ...]) -> float:
        q = np.asarray(q, dtype=float)
        a = self.anchor_vector()
        sel = np.array(dims, dtype=int)
        diff = q[sel] - a[sel]
        return float(np.sqrt(np.sum(diff * diff)))

    def radius(self, mult: float = 1.0) -> float:
        std = np.array(self.anchor_std, dtype=float)
        return float(mult * np.linalg.norm(std))



class ExperienceMemory:

    def __init__(self, num_stages: int, num_agents_per_stage: int, config: Optional[MemoryConfig] = None):
        self.cfg = config or MemoryConfig()
        self.num_stages = int(num_stages)
        self.num_agents_per_stage = int(num_agents_per_stage)
        self.store: Dict[Tuple[int, int], List[Observation]] = {(s, a): [] for s in range(num_stages) for a in range(num_agents_per_stage)}
        self.rules: Dict[Tuple[int, int], List[Rule]] = {(s, a): [] for s in range(num_stages) for a in range(num_agents_per_stage)}
        os.makedirs(self.cfg.save_dir, exist_ok=True)


    def add_observation(
        self,
        period: int,
        s: int,
        a: int,
        demand: float,
        assets: float,
        backlog: float,
        inventory: float,
        order: float,
        reward: float,
        order_vec: Optional[List[int]] = None,
    ) -> None:
        obs = Observation(
            period=int(period),
            demand=float(demand),
            assets=float(assets),
            backlog=float(backlog),
            inventory=float(inventory),
            order=float(order),
            reward=float(reward),
            order_vec=list(order_vec) if order_vec is not None else None,
        )
        self.store[(s, a)].append(obs)

    def last_n(self, s: int, a: int, n: int) -> List[Observation]:
        return self.store[(s, a)][-n:]


    def _stats_for_scaling(self, s: int, a: int) -> Tuple[np.ndarray, np.ndarray]:
        data = self.store[(s, a)]
        if not data:
            return np.zeros(4), np.ones(4)
        M = np.stack([o.state_vector() for o in data], axis=0)  # (N,4)
        mean = M.mean(axis=0)
        std = M.std(axis=0)
        std[std < 1e-6] = 1.0
        return mean, std

    def neighbors(self, s: int, a: int, q_state: np.ndarray, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = int(k or self.cfg.k_neighbors)
        data = self.store[(s, a)]
        if not data:
            return []
        q = np.asarray(q_state, dtype=float).ravel()
        dims = np.array(self.cfg.use_dims, dtype=int)
        mean, std = self._stats_for_scaling(s, a)
        M = np.stack([o.state_vector() for o in data], axis=0)
        Z = (M - mean) / std
        zq = (q - mean) / std
        diff = Z[:, dims] - zq[dims]
        d = np.sqrt((diff * diff).sum(axis=1))  # (N,)
        order = np.argsort(d)[: min(k, len(d))]
        return [{"obs": data[i], "_dist": float(d[i])} for i in order]

    def suggest_order_vector(
        self, s: int, a: int, q_state: np.ndarray, suppliers_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        neigh = self.neighbors(s, a, q_state, self.cfg.k_neighbors)
        vecs: List[np.ndarray] = []
        ws: List[float] = []
        ds: List[float] = []
        for n in neigh:
            ov = n["obs"].order_vec
            if ov is None:
                continue
            vecs.append(np.asarray(ov, dtype=float))
            dist = float(n["_dist"])
            rew = max(0.0, float(n["obs"].reward))
            w = (1.0 / (self.cfg.distance_eps + dist)) * (1.0 + 0.1 * rew)
            ws.append(w)
            ds.append(dist)
        if not vecs:
            return None

        if len(vecs) >= 3:
            std_sum = np.std(np.vstack(vecs), axis=0).sum()
            if std_sum < 1e-3:
                idx = int(np.argmax(ws))
                out = np.asarray(vecs[idx]).astype(int)
                if suppliers_mask is not None and len(suppliers_mask) == len(out):
                    out = out * suppliers_mask
                return out

        L = max(len(v) for v in vecs)
        mat = np.zeros((len(vecs), L), dtype=float)
        for i, v in enumerate(vecs):
            mat[i, : len(v)] = v
        w = np.asarray(ws, dtype=float)
        w /= max(1e-12, w.sum())
        avg = (w.reshape(-1, 1) * mat).sum(axis=0)
        out = np.maximum(0, np.rint(avg)).astype(int)
        if suppliers_mask is not None and len(suppliers_mask) == len(out):
            out = out * suppliers_mask
        return out


    def _scheduled_reflection(self, period: int, s: int, a: int) -> bool:
        if len(self.store[(s, a)]) < self.cfg.min_obs_for_reflection:
            return False
        if self.cfg.reflect_every <= 0:
            return False
        return (period + 1) % self.cfg.reflect_every == 0

    def _asset_drop_trigger(self, s: int, a: int) -> bool:
        win = self.cfg.asset_drop_window
        if win <= 1:
            return False
        data = self.last_n(s, a, win)
        if len(data) < win:
            return False
        assets = [o.assets for o in data]
        if assets[0] <= 0:
            return False
        total_drop = (assets[0] - assets[-1]) / max(1e-6, assets[0])
        if total_drop < self.cfg.asset_total_drop_pct:
            return False
        for i in range(1, len(assets)):
            prev, cur = assets[i - 1], assets[i]
            if prev <= 0:
                return False
            step_drop = (prev - cur) / prev
            if step_drop < self.cfg.asset_drop_pct_per_step:
                return False
        return True

    def _backlog_rise_trigger(self, s: int, a: int) -> bool:
        win = self.cfg.backlog_rise_window
        if win <= 1:
            return False
        data = self.last_n(s, a, win)
        if len(data) < win:
            return False
        backlogs = [o.backlog for o in data]
        if max(backlogs) < self.cfg.backlog_min_level:
            return False
        for i in range(1, len(backlogs)):
            prev, cur = backlogs[i - 1], backlogs[i]
            if prev <= 0:
                if (cur - prev) < self.cfg.backlog_min_level:
                    return False
            else:
                step_rise = (cur - prev) / prev
                if step_rise < self.cfg.backlog_rise_pct_per_step:
                    return False
        return True

    def should_reflect(self, s: int, a: int, period: int) -> Optional[str]:
        if self._asset_drop_trigger(s, a):
            return "asset_drop"
        if self._backlog_rise_trigger(s, a):
            return "backlog_rise"
        if self._scheduled_reflection(period, s, a):
            return "scheduled"
        return None

    def _prepare_reflection_window(self, s: int, a: int, max_n: int = 9) -> List[Observation]:
        return self.last_n(s, a, max_n)

    def _window_stats(self, window: List[Observation]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        if not window:
            return np.zeros(4), np.ones(4), []
        M = np.stack([o.state_vector() for o in window], axis=0)
        mean = M.mean(axis=0)
        std = M.std(axis=0)
        std[std < 1e-6] = 1.0
        periods = [o.period for o in window]
        return mean, std, periods

    def reflect(
        self,
        s: int,
        a: int,
        llm_fn: Callable[[str], str],
        period: int,
        reason: str,
        window_n: int = 9,
    ) -> List[Dict[str, Any]]:
        window = self._prepare_reflection_window(s, a, window_n)
        if len(window) < self.cfg.min_obs_for_reflection:
            return []

        mean, std, w_periods = self._window_stats(window)
        payload = [
            {
                "period": o.period,
                "state": {
                    "demand": o.demand,
                    "assets": o.assets,
                    "backlog": o.backlog,
                    "inventory": o.inventory,
                },
                "order": o.order,
                "reward": o.reward,
            }
            for o in window
        ]
        prompt = (
            "You are a supply-chain coach. From the recent rounds' data (state, order, reward), "
            "derive 1-3 high-level, general rules that improve future ordering decisions. "
            "Each rule should be short, concrete, and testable.\n"
            "Return strictly in JSON as:\n"
            "{\n"
            '  "rules": [\n'
            '    {"because":"...", "then":"...", "note":"(optional)"}\n'
            "  ]\n"
            "}\n"
            f"Data: {json.dumps(payload)}\n"
            "Focus on balancing backlog vs inventory, avoiding stockouts and over-ordering, "
            "and reacting to demand/asset changes.\n"
        )

        raw = llm_fn(prompt)

        try:
            log_dir = os.path.join(self.cfg.save_dir, "reflection_raw")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"p{period}_s{s}_a{a}.txt"), "w", encoding="utf-8") as f:
                f.write(str(raw))
        except Exception:
            pass

        rules_out: List[Dict[str, Any]] = []
        obj = None
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except Exception:
                obj = None
        if isinstance(obj, dict):
            cand = None
            for key in ("rules", "insights"):
                if key in obj and isinstance(obj[key], list):
                    cand = obj[key]
                    break
            if cand is None:
                cand = []
            for r in cand:
                if isinstance(r, dict):
                    because = str(r.get("because", "")).strip()
                    then = str(r.get("then", "")).strip()
                    note = str(r.get("note", "")).strip() if "note" in r else ""
                    if because or then or note:
                        rules_out.append({"because": because, "then": then, "note": note})
                elif isinstance(r, str) and r.strip():
                    rules_out.append({"because": "", "then": r.strip(), "note": ""})

        if not rules_out and isinstance(raw, str):
            for m in re.finditer(r"^-\s+(.+)$", raw, flags=re.M):
                txt = m.group(1).strip()
                if txt:
                    rules_out.append({"because": "", "then": txt, "note": ""})
        if not rules_out:
            rules_out.append({
                "because": "Fallback: no valid JSON parsed.",
                "then": "If backlog is rising and assets are shrinking in similar states, raise orders toward demand plus a portion of backlog.",
                "note": str(raw)[:200] if isinstance(raw, str) else "",
            })

        anchor_mean = mean.tolist()
        anchor_std = std.tolist()
        new_rules: List[Rule] = []
        for idx, rr in enumerate(rules_out):
            rid = f"{s}-{a}-{period}-{idx}"
            rule = Rule(
                rule_id=rid,
                period=period,
                stage=s,
                agent=a,
                reason=reason,
                insights=[x for x in [rr.get("because", ""), rr.get("then", ""), rr.get("note", "")] if x],
                anchor_mean=anchor_mean,
                anchor_std=anchor_std,
                window_periods=w_periods,
                meta={"raw_len": len(str(raw)) if isinstance(raw, str) else 0},
            )
            new_rules.append(rule)

        cur = self.rules[(s, a)]
        cur.extend(new_rules)
        if len(cur) > self.cfg.max_rules_per_agent:
            self.rules[(s, a)] = cur[-self.cfg.max_rules_per_agent :]

        return [asdict(r) for r in new_rules]

    def retrieve_applicable_rules(
        self,
        s: int,
        a: int,
        q_state: np.ndarray,
        top_k: int = 3,
        max_radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:

        q = np.asarray(q_state, dtype=float).ravel()
        dims = tuple(range(len(q)))
        out: List[Tuple[float, Rule, float]] = []
        for r in self.rules[(s, a)]:
            d = r.distance(q, dims=dims)
            r_radius = r.radius(mult=self.cfg.rule_anchor_std_multiplier) if max_radius is None else float(max_radius)
            if r_radius < 1e-6:
                r_radius = 0.5
            if d <= r_radius:
                out.append((d, r, r_radius))
        out.sort(key=lambda x: x[0])
        out = out[: min(top_k, len(out))]
        return [
            {
                "rule_id": r.rule_id,
                "reason": r.reason,
                "insights": r.insights,
                "distance": float(d),
                "radius": float(rad),
                "anchor_mean": r.anchor_mean,
                "anchor_std": r.anchor_std,
            }
            for (d, r, rad) in out
        ]


    def save(self) -> None:
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        for (s, a), data in self.store.items():
            if not data:
                continue
            df = pd.DataFrame(
                [
                    {
                        "period": o.period,
                        "demand": o.demand,
                        "assets": o.assets,
                        "backlog": o.backlog,
                        "inventory": o.inventory,
                        "order": o.order,
                        "reward": o.reward,
                        "order_vec": json.dumps(o.order_vec) if o.order_vec is not None else "",
                    }
                    for o in data
                ]
            )
            df.to_csv(os.path.join(self.cfg.save_dir, f"mem_s{s}_a{a}.csv"), index=False)

        out = {}
        for (s, a), rules in self.rules.items():
            out[f"{s}-{a}"] = [
                {
                    "rule_id": r.rule_id,
                    "period": r.period,
                    "stage": r.stage,
                    "agent": r.agent,
                    "reason": r.reason,
                    "insights": r.insights,
                    "anchor_mean": r.anchor_mean,
                    "anchor_std": r.anchor_std,
                    "window_periods": r.window_periods,
                    "meta": r.meta,
                }
                for r in rules
            ]
        with open(os.path.join(self.cfg.save_dir, "rules.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        # observations
        for s in range(self.num_stages):
            for a in range(self.num_agents_per_stage):
                path = os.path.join(self.cfg.save_dir, f"mem_s{s}_a{a}.csv")
                if not os.path.exists(path):
                    continue
                try:
                    df = pd.read_csv(path)
                    data: List[Observation] = []
                    for _, row in df.iterrows():
                        try:
                            ov_raw = row.get("order_vec", "")
                            ov = json.loads(ov_raw) if isinstance(ov_raw, str) and ov_raw else None
                        except Exception:
                            ov = None
                        data.append(
                            Observation(
                                period=int(row["period"]),
                                demand=float(row["demand"]),
                                assets=float(row["assets"]),
                                backlog=float(row["backlog"]),
                                inventory=float(row["inventory"]),
                                order=float(row["order"]),
                                reward=float(row["reward"]),
                                order_vec=ov,
                            )
                        )
                    self.store[(s, a)] = data
                except Exception:
                    pass
        # rules
        rules_path = os.path.join(self.cfg.save_dir, "rules.json")
        if os.path.exists(rules_path):
            try:
                raw = json.load(open(rules_path, "r", encoding="utf-8"))
                for key, arr in raw.items():
                    try:
                        s, a = map(int, key.split("-"))
                    except Exception:
                        continue
                    lst: List[Rule] = []
                    for obj in arr:
                        try:
                            lst.append(
                                Rule(
                                    rule_id=str(obj["rule_id"]),
                                    period=int(obj["period"]),
                                    stage=int(obj["stage"]),
                                    agent=int(obj["agent"]),
                                    reason=str(obj["reason"]),
                                    insights=list(obj.get("insights", [])),
                                    anchor_mean=list(obj.get("anchor_mean", [0, 0, 0, 0])),
                                    anchor_std=list(obj.get("anchor_std", [1, 1, 1, 1])),
                                    window_periods=list(obj.get("window_periods", [])),
                                    meta=dict(obj.get("meta", {})),
                                )
                            )
                        except Exception:
                            continue
                    self.rules[(s, a)] = lst
            except Exception:
                pass

