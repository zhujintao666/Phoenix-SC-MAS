from typing import List, Dict, Any
import os, csv, re, time, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from autogen import ConversableAgent
except Exception:
    ConversableAgent = object


from memory_module import ExperienceMemory, MemoryConfig

def _sanitize_llm_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return cfg
    d = dict(cfg)
    if "request_timeout" in d:
        d.setdefault("timeout", d.pop("request_timeout"))
    cl = d.get("config_list")
    if isinstance(cl, list):
        new_cl = []
        for item in cl:
            if isinstance(item, dict) and "request_timeout" in item:
                item = dict(item)
                item.setdefault("timeout", item.pop("request_timeout"))
            new_cl.append(item)
        d["config_list"] = new_cl
    return d

def _deep_sanitize_llm_config(obj):
    if isinstance(obj, dict):
        if "request_timeout" in obj:
            val = obj.pop("request_timeout")
            obj.setdefault("timeout", val)
        for k in list(obj.keys()):
            _deep_sanitize_llm_config(obj[k])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            _deep_sanitize_llm_config(obj[i])


def _has_request_timeout(obj) -> bool:
    if isinstance(obj, dict):
        if "request_timeout" in obj:
            return True
        return any(_has_request_timeout(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_request_timeout(x) for x in obj)
    return False


def create_agents(stage_names: List[str], num_agents_per_stage: int, llm_config) -> List[ConversableAgent]:
    agents: List[ConversableAgent] = []
    num_stages = len(stage_names)
    safe_llm_config = _sanitize_llm_config(llm_config)

    for stage, stage_name in enumerate(stage_names):
        ag = ConversableAgent(
            name=f"{stage_name.capitalize()}Agent_0",
            system_message=(
                f"You play a crucial role in a {num_stages}-stage supply chain as the stage {stage + 1} ({stage_name}). "
                "Your goal is to minimize the total cost by managing inventory and orders effectively."
            ),
            llm_config=safe_llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )
        agents.append(ag)

        for sa_ind in range(1, num_agents_per_stage):
            ag2 = ConversableAgent(
                name=f"{stage_name.capitalize()}Agent_{sa_ind}",
                system_message=ag.system_message,
                llm_config=safe_llm_config,
                code_execution_config=False,
                human_input_mode="NEVER",
            )
            agents.append(ag2)

    for ag in agents:
        try:
            if hasattr(ag, "llm_config") and isinstance(ag.llm_config, dict):
                _deep_sanitize_llm_config(ag.llm_config)
        except Exception:
            pass

    try:
        print(
            "[AGENTS] llm_config sanitized; any request_timeout left?",
            any(_has_request_timeout(getattr(ag, "llm_config", {})) for ag in agents)
        )
    except Exception:
        pass

    return agents

def run_simulation(
    im_env,
    user_proxy,
    stage_agents,
    config_name: str,
    round: int = 0,
    enable_memory: bool = True,
    enable_knn_suggest: bool | None = None,
    run_tag: str | None = None,
    return_meta: bool = False,
    plot_after: bool = True,
    stop_on_first_bankruptcy: bool = False,
    enable_resurrection: bool = False,
    revive_assets: float = 1000.0,
    revive_inventory: int = 50,
    reset_suppliers_on_revive: bool = True,
    max_bankruptcies: int = 5,
    revive_policy: str = "initial_assets_initinv_plus_backlog",
    stop_on_exhausted_resurrections: bool = True,
    enable_regular_reflection_with_resurrection: bool = False,
    enable_bankruptcy_reflection: bool = True,
):
    import os, re, json, time, csv, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from copy import deepcopy
    from form_msg import generate_msg
    from memory_module import ExperienceMemory, MemoryConfig

    if enable_knn_suggest is None:
        enable_knn_suggest = enable_memory

    ts = run_tag if run_tag else time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("results", config_name, ts)
    dir_chat = os.path.join(run_dir, "chat_summaries")
    dir_mem  = os.path.join(run_dir, "memory")
    dir_rec  = os.path.join(run_dir, "records")
    dir_viz  = os.path.join(run_dir, "viz")
    dir_plots= os.path.join(run_dir, "plots")
    for d in (run_dir, dir_chat, dir_mem, dir_rec, dir_viz, dir_plots):
        os.makedirs(d, exist_ok=True)
    print(f"[RUN] save root -> {run_dir}")

    OBS_PATH = os.path.join(dir_rec, "demand_inventory_backlog_assets.csv")
    OBS_COLS = ["period", "stage", "agent", "demand", "inventory", "backlog", "assets", "order"]
    _obs_f = open(OBS_PATH, "w", newline="", encoding="utf-8")
    _obs_writer = csv.DictWriter(_obs_f, fieldnames=OBS_COLS)
    _obs_writer.writeheader()

    LEDGER_PATH = os.path.join(dir_rec, "ledger.csv")
    LEDGER_COLS = [
        "period", "stage", "agent",
        "assets_start", "revenue", "order_cost", "holding_cost", "backlog_cost",
        "profit", "bankruptcy_adjustment", "identity_diff", "assets_end",
        "deduct_reasons", "bankrupt", "revived"
    ]
    _ledger_f = open(LEDGER_PATH, "w", newline="", encoding="utf-8")
    _ledger_writer = csv.DictWriter(_ledger_f, fieldnames=LEDGER_COLS)
    _ledger_writer.writeheader()

    RESUR_PATH = os.path.join(dir_rec, "resurrection_log.csv")
    RESUR_COLS = [
        "period", "stage", "agent",
        "bankruptcy_count", "over_limit", "policy",
        "assets_end_before", "init_assets", "init_inventory", "backlog_t",
        "revive_assets", "revive_inventory", "new_assets", "new_inventory",
        "note"
    ]
    _resur_f = open(RESUR_PATH, "w", newline="", encoding="utf-8")
    _resur_writer = csv.DictWriter(_resur_f, fieldnames=RESUR_COLS)
    _resur_writer.writeheader()

    CLEAR_HISTORY = True
    WARMUP_PERIODS = 6
    MAX_MSG_CHARS = 4000
    EMERGENCY_K = 2

    def _sha_arr(arr):
        try:
            arr = np.ascontiguousarray(np.array(arr)).view(np.uint8)
            return hashlib.sha256(arr).hexdigest()[:10]
        except Exception:
            return "na"

    def _write_chat_summary(period: int, text: str, reward_sum: float):
        fn = os.path.join(dir_chat, f"round{round}_period{period:03d}_reward{int(reward_sum)}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        return fn

    def parse_orders_and_reason(raw_text: str, n_expected: int):
        txt = (raw_text or "").strip()
        if not txt:
            raise ValueError("empty reply")
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and "orders" in obj:
                arr = obj["orders"]
                if isinstance(arr, list) and len(arr) == n_expected and all(isinstance(x, (int, float)) for x in arr):
                    vec = np.array([max(0, int(x)) for x in arr], dtype=int)
                    reason = ""
                    for k in ("why", "reason", "rationale"):
                        if isinstance(obj.get(k, ""), str):
                            reason = obj[k].strip(); break
                    return vec, reason
        except Exception:
            pass
        m_blk = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt, flags=re.I)
        if m_blk:
            block = m_blk.group(1)
            try:
                obj = json.loads(block)
                if isinstance(obj, dict) and "orders" in obj and isinstance(obj["orders"], list):
                    arr = obj["orders"]
                    if len(arr) == n_expected and all(isinstance(x, (int, float)) for x in arr):
                        vec = np.array([max(0, int(x)) for x in arr], dtype=int)
                        reason = ""
                        for k in ("why", "reason", "rationale"):
                            if isinstance(obj.get(k, ""), str):
                                reason = obj[k].strip(); break
                        return vec, reason
            except Exception:
                pass
        for m in re.finditer(r'"orders"\s*:\s*\[([^\]]*)\]', txt, flags=re.I):
            arr_txt = m.group(1)
            nums = [p.strip() for p in arr_txt.split(",") if p.strip()]
            try:
                arr = [int(x) for x in nums]
            except Exception:
                continue
            if len(arr) == n_expected and all(x >= 0 for x in arr):
                reason = ""
                near = txt[max(0, m.end()): m.end() + 300]
                m2 = re.search(r'"(?:why|reason|rationale)"\s*:\s*"([^"]{0,400})"', near, flags=re.I)  # 修复小拼写
                if m2: reason = m2.group(1).strip()
                return np.array(arr, dtype=int), reason
        cand = []
        for m in re.finditer(r'\[([0-9,\s]+)\]', txt):
            arr_txt = m.group(1)
            nums = [p.strip() for p in arr_txt.split(",") if p.strip()]
            try:
                arr = [int(x) for x in nums]
            except Exception:
                continue
            if len(arr) == n_expected and all(x >= 0 for x in arr):
                cand.append((np.array(arr, int), (m.start(1), m.end(1)), 10 + m.start()))
        if cand:
            cand.sort(key=lambda x: x[2], reverse=True)
            vec, span, _ = cand[0]
            reason = txt[:span[0]].strip()
            return vec, reason
        raise ValueError("No parsable orders found.")

    def _eff_lead_time(env, stage_id, agent_id, sup_vec):
        try:
            lt = getattr(env, "lead_time", None)
            if lt is None:
                return 1
            arr = np.asarray(lt)
            if arr.ndim == 0:
                return int(max(1, int(arr)))
            if arr.ndim == 1:
                idx = agent_id if agent_id < arr.shape[0] else -1
                return int(max(1, int(arr[idx])))
            if arr.ndim == 2:
                return int(max(1, int(arr[stage_id, agent_id])))
            vec = np.asarray(arr[stage_id, agent_id]).reshape(-1)
            if vec.size == 0:
                return 1
            sup = np.asarray(sup_vec).astype(int).reshape(-1)
            L = min(len(sup), len(vec))
            effs = vec[:L][sup[:L] > 0]
            if effs.size == 0:
                effs = vec[:L]
            return int(max(1, int(np.median(effs))))
        except Exception:
            return 1

    def resurrect_agent(env, s: int, a: int, t_end: int,
                        target_assets: float, target_inventory: int,
                        reset_suppliers: bool = False):
        """
        Set the bankrupt agent to a stable 'revived' state:
        - assets := target_assets
        - inventory := target_inventory (at end-of-period); backlog := 0
        - running_agents := 1
        - do NOT reset supplier links by default to avoid topology shock
        Return (new_assets, new_inventory) for logging.
        """
        # assets
        try:
            env.assets[s, a] = float(target_assets)
        except Exception:
            pass

        # inventory / backlog：尽量同时更新 'inventory' 或带时间轴的 'inventories'
        new_inv = None
        try:
            env.inventory[s, a] = int(target_inventory)
            new_inv = int(env.inventory[s, a])
        except Exception:
            pass
        try:
            env.backlogs[s, a, t_end] = 0  # ← 关键补丁
        except Exception:
            pass

        try:
            env.inventories[s, a, t_end] = int(target_inventory)
            new_inv = int(env.inventories[s, a, t_end])
        except Exception:
            pass

        try:
            env.running_agents[s, a] = 1
        except Exception:
            pass

        if reset_suppliers:
            try:
                if hasattr(env, "sc_graph") and hasattr(env.sc_graph, "ensure_min_suppliers"):
                    env.sc_graph.ensure_min_suppliers(s, a)
            except Exception:
                pass

        try:
            new_assets = float(env.assets[s, a])
        except Exception:
            new_assets = float("nan")
        if new_inv is None:
            try:
                _sd = env.parse_state()
                k = f"stage_{s}_agent_{a}"
                new_inv = int(_sd.get(k, {}).get("inventory", 0))
            except Exception:
                new_inv = 0

        return new_assets, int(new_inv)

    def _collect_last_orders(history_dict, period, key, L, vec_len):
        last_vecs = []
        inbound_vec = np.zeros(vec_len, dtype=int)
        for lag in range(1, L+1):
            hist = history_dict.get(period - lag + 1, {})
            pv = np.asarray(hist.get(key, np.zeros(vec_len, dtype=int)), dtype=int)
            if pv.shape[0] != vec_len:
                vv = np.zeros(vec_len, dtype=int)
                vv[:min(vec_len, pv.shape[0])] = pv[:min(vec_len, pv.shape[0])]
                pv = vv
            last_vecs.append(pv)
            if lag == L:
                inbound_vec = pv.copy()
        last_totals = [int(v.sum()) for v in last_vecs]
        return last_vecs, last_totals, inbound_vec, int(inbound_vec.sum())

    im_env.reset()
    num_stages = im_env.num_stages
    num_agents_per_stage = im_env.num_agents_per_stage
    llm_agent_set = im_env.llm_agent_set
    enable_graph_change = im_env.enable_graph_change
    enable_price_change = im_env.enable_price_change

    print("[CFG] seed=", getattr(im_env, "seed", None))
    try:
        print("[CFG] prices=", _sha_arr(getattr(im_env, "sale_prices", [])))
        print("[CFG] costs=",  _sha_arr(getattr(im_env, "order_costs", [])))
        print("[CFG] lead_time=", _sha_arr(getattr(im_env, "lead_time", [])))
        print("[CFG] init_assets=", _sha_arr(getattr(im_env, "assets", [])))
        print("[CFG] init_inventory=", _sha_arr(getattr(im_env, "inventory", [])))
    except Exception as _e:
        print("[CFG] fingerprint error:", _e)
    # === Baselines: initial assets & initial inventories (for revive policy) ===
    S, A = im_env.num_stages, im_env.num_agents_per_stage
    bankrupt_cum = np.zeros((S, A), dtype=int)
    revived_cum = np.zeros((S, A), dtype=int)
    try:
        init_assets0 = np.asarray(getattr(im_env, "init_assets"), dtype=float).reshape(S, A)
    except Exception:
        init_assets0 = np.asarray(getattr(im_env, "assets"), dtype=float).reshape(S, A).copy()

    try:
        init_inventories0 = np.asarray(getattr(im_env, "init_inventories"), dtype=int).reshape(S, A)
    except Exception:
        try:
            _state0 = im_env.parse_state()
        except TypeError:
            _state0 = im_env.parse_state(getattr(im_env, "state_dict", {}))
        init_inventories0 = np.zeros((S, A), dtype=int)
        for s in range(S):
            for a in range(A):
                k = f"stage_{s}_agent_{a}"
                init_inventories0[s, a] = int(_state0.get(k, {}).get("inventory", 0))

    bankrupt_counts = np.zeros((S, A), dtype=int)

    try:
        from utils import visualize_state
        visualize_state(env=im_env, rewards={}, t=-1, save_prefix=os.path.join(config_name, ts, "viz"))
    except Exception:
        pass

    mem = ExperienceMemory(
        num_stages=num_stages,
        num_agents_per_stage=num_agents_per_stage,
        config=MemoryConfig(
            reflect_every=6,
            min_obs_for_reflection=6,
            save_dir=dir_mem,
        )
    )
    mem.cfg.k_neighbors = 5

    try:
        from autogen import ConversableAgent
    except Exception:
        ConversableAgent = object

    def _deep_sanitize(d):
        if isinstance(d, dict):
            if "request_timeout" in d:
                d.setdefault("timeout", d.pop("request_timeout"))
            for k in list(d.keys()):
                _deep_sanitize(d[k])
        elif isinstance(d, list):
            for i in range(len(d)):
                _deep_sanitize(d[i])

    try:
        base_ref_cfg = deepcopy(getattr(stage_agents[0], "llm_config", {}))
    except Exception:
        base_ref_cfg = {}
    _deep_sanitize(base_ref_cfg)
    reflection_agent = ConversableAgent(
        name="ReflectionAgent",
        system_message=(
            "You are a supply-chain coach. Given recent observations (demand, assets, backlog, inventory), "
            "orders and rewards, derive 1-3 general, testable rules to improve ordering. "
            'Return JSON like: {"rules":[{"because":"...", "then":"...", "note":"..."}]}'
        ),
        llm_config=base_ref_cfg,
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    def _reflection_llm_fn(prompt: str) -> str:
        res = user_proxy.initiate_chat(
            reflection_agent,
            message={"content": prompt},
            summary_method="last_msg",
            max_turns=1,
            clear_history=True,
        )
        txt = ""
        if getattr(res, "chat_history", None):
            txt = (res.chat_history[-1].get("content") or "").strip()
        if not txt:
            txt = (res.summary or "").strip()
        return txt

    records = []
    api_cost = 0.0
    all_action_order_dicts: Dict[int, Dict[str, np.ndarray]] = {}
    all_action_price_dicts = {}
    all_action_sup_dicts = {}
    all_action_dem_dicts = {}
    episode_reward = 0.0

    for period in range(im_env.num_periods):
        terminate_on_exhaust = False
        exhaust_info = None

        total_chat_summary = ""

        try:
            state_dict = im_env.parse_state()
        except TypeError:
            state_dict = im_env.parse_state(getattr(im_env, "state_dict", {}))

        past_req_orders = all_action_order_dicts.get(period, dict())
        try:
            im_env.sc_graph.update_graph(state_dict=state_dict, past_req_orders=past_req_orders)
        except Exception:
            pass

        action_order_dict = {}
        action_price_dict = {}
        action_sup_dict = {}
        action_dem_dict = {}

        emergent_events = getattr(im_env, "emergent_events", {}).get(period, [])
        shutdown_list = None
        recovery_list = None
        for event in emergent_events:
            if event == "demand_surge":
                print("There is a sudden demand surge.")
                try: im_env.create_demand_surge()
                except Exception: pass
            if event == "sudden_shutdown":
                print("There is a sudden shutdown event.")
                shutdown_list = getattr(im_env, "shut_seq", {}).get(period, [])
                for s_id0, a_id0 in shutdown_list:
                    try: state_dict = im_env.create_shutdown_event(s_id0, a_id0, state_dict)
                    except Exception: pass
            if event == "recovery":
                print("Here is a recovery event.")
                recovery_list = getattr(im_env, "rec_seq", {}).get(period, [])
                for (s_id0, a_id0) in recovery_list:
                    try: im_env.create_recovery_event(s_id0, a_id0)
                    except Exception: pass

        try:
            running_prev = np.array(im_env.running_agents, dtype=int).copy()
        except Exception:
            running_prev = np.ones((num_stages, num_agents_per_stage), dtype=int)

        for stage_id in range(num_stages):
            for agent_id in range(num_agents_per_stage):
                key = f"stage_{stage_id}_agent_{agent_id}"

                try:
                    if running_prev[stage_id, agent_id] == 0:
                        print(f"[SKIP] period={period} {key} is shutdown (start-of-period)")
                        action_sup_dict[key] = np.zeros(num_agents_per_stage, dtype=int)
                        action_order_dict[key] = np.zeros(num_agents_per_stage, dtype=int)
                        action_price_dict[key] = 0
                        action_dem_dict[key] = 0
                        continue
                except Exception:
                    pass

                if stage_id == 0:
                    downstream_need = int(state_dict.get(key, {}).get("demand", 0))
                else:
                    downstream_need = 0
                    for down_agent in range(num_agents_per_stage):
                        down_key = f"stage_{stage_id-1}_agent_{down_agent}"
                        vec = action_order_dict.get(down_key, np.zeros(num_agents_per_stage, dtype=int))
                        if agent_id < len(vec):
                            downstream_need += int(vec[agent_id])

                if (stage_id, agent_id) in llm_agent_set:
                    stage_state = state_dict.get(key, {})
                    pr_orders = past_req_orders.get(key, [])

                    cur_demand = int(stage_state.get("demand", 0)) if stage_id == 0 else int(downstream_need)
                    cur_assets = float(getattr(im_env, "assets")[stage_id, agent_id]) if hasattr(im_env, "assets") else float(stage_state.get("assets", 0.0))
                    cur_backlog = int(stage_state.get("backlog", 0))
                    cur_inventory = int(stage_state.get("inventory", 0))
                    state_vec = np.array([cur_demand, cur_assets, cur_backlog, cur_inventory], dtype=float)

                    sup_vec = np.asarray(stage_state.get("suppliers", np.zeros(num_agents_per_stage)), dtype=int)

                    if stage_id == num_stages - 1:
                        sup_vec = np.ones(num_agents_per_stage, dtype=int)
                    else:
                        sup_vec = (sup_vec > 0).astype(int)
                        try:
                            sup_alive_mask = np.asarray(running_prev[stage_id + 1], dtype=int)
                            Lg = min(len(sup_vec), len(sup_alive_mask))
                            sup_vec[:Lg] = sup_vec[:Lg] * (sup_alive_mask[:Lg] > 0).astype(int)
                        except Exception:
                            pass

                    if stage_id < num_stages - 1 and sup_vec.sum() == 0 and (downstream_need > 0 or cur_backlog > 0):
                        print(
                            f"[WARN][SUPPLY-CUTOFF] p={period} {key}: need={downstream_need}, backlog={cur_backlog}, but no suppliers available")

                    L_eff = _eff_lead_time(im_env, stage_id, agent_id, sup_vec)
                    last_vecs, last_totals, inbound_vec, inbound_total = _collect_last_orders(
                        history_dict=all_action_order_dicts,
                        period=period, key=key, L=L_eff, vec_len=num_agents_per_stage
                    )

                    lead_card = [
                        "LeadTimeContext:",
                        f"- lead_time = {L_eff} periods",
                        f"- recent orders totals (t-1 → t-{L_eff}): {last_totals}",
                        f"- inbound_next_vec (≈ orders at t-{L_eff}): {inbound_vec.tolist()} (total={inbound_total})",
                    ]
                    lead_orders = [
                        "LeadTimeOrders:",
                        f"- recent_vectors (t-1 → t-{L_eff}): { [v.tolist() for v in last_vecs] }",
                    ]

                    mem_hint_lines = []
                    if enable_memory and enable_knn_suggest and period >= WARMUP_PERIODS:
                        knn_vec = mem.suggest_order_vector(
                            s=stage_id, a=agent_id, q_state=state_vec, suppliers_mask=sup_vec
                        )
                        if knn_vec is not None:
                            mem_hint_lines.append(
                                f"MemoryHint: similar past states suggest orders ≈ {knn_vec.astype(int).tolist()} (masked by suppliers)."
                            )

                    rule_lines = []
                    if enable_memory:
                        if enable_resurrection:
                            all_rules = mem.rules.get((stage_id, agent_id), [])
                            insights = []
                            for r in all_rules:
                                if isinstance(r, dict) and r.get("insights"):
                                    for ins in r["insights"]:
                                        if isinstance(ins, str) and ins.strip():
                                            insights.append(ins.strip())
                            seen = set()
                            uniq_insights = []
                            for ins in insights:
                                if ins not in seen:
                                    seen.add(ins)
                                    uniq_insights.append(ins)
                            N = 8
                            if uniq_insights:
                                rule_lines.append("LearnedRules (global):")
                                for ins in uniq_insights[:N]:
                                    rule_lines.append(f"- {ins}")
                        else:
                            rules = mem.retrieve_applicable_rules(stage_id, agent_id, state_vec, top_k=3)
                            if rules:
                                rule_lines.append("ApplicableRules:")
                                for r in rules:
                                    insight = r["insights"][-1] if r.get("insights") else ""
                                    rule_lines.append(f"- {insight}  [dist≈{r.get('distance', 0):.2f}]")

                    op_hints = [
                        "Operational reminders:",
                        "- Consider lead time: plan for demand over next L periods, not only current demand.",
                        "- If backlog exists, clear it gradually across L; avoid one-shot overshoot.",
                    ]
                    prefix_parts = ["You MUST use the following context when deciding orders."]
                    prefix_parts += lead_card + lead_orders + mem_hint_lines + rule_lines + op_hints
                    prefix = "\n".join(prefix_parts) + "\n"

                    base_msg, _state_info = generate_msg(
                        shutdown_list=shutdown_list,
                        recovery_list=recovery_list,
                        enable_graph_change=enable_graph_change,
                        stage_id=stage_id,
                        cur_agent_id=agent_id,
                        stage_state=stage_state,
                        im_env=im_env,
                        enable_price_change=enable_price_change,
                        action_order_dict=action_order_dict,
                        past_req_orders=pr_orders,
                        period=period,
                    )

                    body_full = ''.join(base_msg)
                    suffix = (
                        f'\nReturn ONE JSON object on a single line:\n'
                        f'{{"orders":[q1,q2,...,q{num_agents_per_stage}], "why":"<short reason>"}}\n'
                        f'If you cannot return JSON, it is OK to output text; I will extract the numbers.\n'
                    )
                    budget = MAX_MSG_CHARS - len(prefix) - len(suffix)
                    if budget < 0: budget = 0
                    body = body_full[:budget]
                    msg0 = prefix + body + suffix

                    chat_result = user_proxy.initiate_chat(
                        stage_agents[stage_id*num_agents_per_stage + agent_id],
                        message={'content': msg0},
                        summary_method='last_msg',
                        max_turns=1,
                        clear_history=CLEAR_HISTORY,
                    )
                    reply_text = ""
                    try:
                        if getattr(chat_result, "chat_history", None):
                            last = chat_result.chat_history[-1]
                            reply_text = (last.get("content") or "").strip()
                        if not reply_text:
                            reply_text = (chat_result.summary or "").strip()
                    except Exception:
                        reply_text = (chat_result.summary or "").strip()
                    try:
                        api_cost_local = chat_result.cost['usage_including_cached_inference']['total_cost']
                    except Exception:
                        api_cost_local = 0.0

                    nonlocal_api_cost = locals().get("api_cost", 0.0)
                    nonlocal_api_cost += api_cost_local
                    globals()["api_cost"] = nonlocal_api_cost

                    try:
                        parsed_vec, reason_text = parse_orders_and_reason(reply_text, num_agents_per_stage)
                    except Exception as e:
                        print(f"[LLM][PARSE-FAIL] period={period} {key}: {e}")
                        parsed_vec = np.zeros(num_agents_per_stage, dtype=int)
                        reason_text = ""

                    raw_vec = np.array([max(0, int(x)) for x in np.asarray(parsed_vec).ravel()], dtype=int)
                    if len(raw_vec) != num_agents_per_stage:
                        raw_vec = np.zeros(num_agents_per_stage, dtype=int)

                    if stage_id < num_stages - 1:
                        final_vec = (np.maximum(0, raw_vec) * sup_vec).astype(int)
                    else:
                        final_vec = np.maximum(0, raw_vec).astype(int)

                    total_chat_summary += (
                        f"=== {key} | period={period} ===\n"
                        f"[PROMPT SENT]\n{msg0}\n"
                        f"[REPLY]\n{reply_text or '[EMPTY]'}\n"
                        f"[PARSED_RAW]{raw_vec.tolist()} | [MASK]{sup_vec.astype(int).tolist()} | "
                        f"[PARSED_FINAL]{final_vec.tolist()} | reason={reason_text}\n\n"
                    )

                    safe_txt = (reply_text or "")[:200].replace("\n", " ")
                    print(
                        f"[DBG][LLM] p={period} {key} raw={raw_vec.tolist()} "
                        f"mask={sup_vec.astype(int).tolist()} final={final_vec.tolist()} msg='{safe_txt}'"
                    )

                    parsed_vec = final_vec

                    action_order_dict[key] = parsed_vec
                    action_sup_dict[key] = sup_vec
                    action_price_dict.setdefault(key, 0)
                    action_dem_dict.setdefault(key, 0)

                else:
                    print(f"[SKIP] period={period} {key} is non-LLM (policy proxy)")
                    try:
                        action_sup_dict, action_order_dict, action_price_dict = im_env.no_backlog_env_proxy(
                            stage_id=stage_id,
                            agent_id=agent_id,
                            action_order_dict=action_order_dict,
                            action_sup_dict=action_sup_dict,
                            action_price_dict=action_price_dict,
                        )
                    except Exception:
                        action_sup_dict[key] = np.zeros(num_agents_per_stage, dtype=int)
                        action_order_dict[key] = np.zeros(num_agents_per_stage, dtype=int)
                        action_price_dict[key] = 0
                    action_price_dict.setdefault(key, 0)
                    action_dem_dict.setdefault(key, 0)

        assets_start_mat = np.asarray(im_env.assets, dtype=float).copy()
        next_states, rewards, terminations, truncations, infos = im_env.step(
            order_dict=action_order_dict,
            sup_dict=action_sup_dict,
            dem_dict=action_dem_dict,
            price_dict=action_price_dict,
        )

        try:
            next_state_dict = im_env.parse_state(next_states)
        except TypeError:
            next_state_dict = im_env.parse_state()

        # === end-of-period index ===
        t_end = int(getattr(im_env, "period", period + 1))

        bankrupt_now = []
        try:
            assets_end_mat = np.asarray(im_env.assets, dtype=float)
            idx = np.argwhere(assets_end_mat <= 0.0)
            for pair in idx:
                bankrupt_now.append((int(pair[0]), int(pair[1])))
        except Exception:
            pass

        bankrupt_set = set(bankrupt_now)

        orders_vec_dict = {
            k: (np.asarray(v, dtype=int) if isinstance(v, (list, np.ndarray)) else np.zeros(num_agents_per_stage,
                                                                                            dtype=int))
            for k, v in action_order_dict.items()
        }

        for s in range(num_stages):
            for a in range(num_agents_per_stage):
                k = f"stage_{s}_agent_{a}"

                if s == 0:
                    try:
                        demand_t = int(getattr(im_env, "demands")[t_end])
                    except Exception:
                        demand_t = 0
                else:
                    demand_t = 0
                    for down_agent in range(num_agents_per_stage):
                        down_key = f"stage_{s - 1}_agent_{down_agent}"
                        vec = orders_vec_dict.get(down_key, np.zeros(num_agents_per_stage, dtype=int))
                        if a < len(vec):
                            demand_t += int(vec[a])

                # end-of-period states
                inventory_t = int(next_state_dict.get(k, {}).get("inventory", 0))
                backlog_t = int(next_state_dict.get(k, {}).get("backlog", 0))
                try:
                    assets_end = float(im_env.assets[s, a])
                except Exception:
                    assets_end = float(next_state_dict.get(k, {}).get("assets", 0.0))

                # my orders total (this period placed)
                my_vec = orders_vec_dict.get(k, np.zeros(num_agents_per_stage, dtype=int))
                order_total = int(np.sum(my_vec))

                # revenue/cost decomposition (match env.step)
                try:
                    sale_price = float(im_env.sale_prices[s, a])
                except Exception:
                    sale_price = 0.0
                try:
                    sales_t = int(getattr(im_env, "sales")[s, a, t_end])
                except Exception:
                    sales_t = 0
                revenue = sale_price * sales_t

                try:
                    unit_costs = np.asarray(im_env.order_costs[s, :], dtype=float)  # (X,)
                    arrivals = np.asarray(im_env.arriving_orders[s, a, :, t_end], dtype=int)  # (X,)
                    order_cost = float(np.sum(unit_costs * arrivals))
                except Exception:
                    order_cost = 0.0

                try:
                    holding_cost = float(im_env.holding_costs[s, a]) * float(inventory_t)
                except Exception:
                    holding_cost = 0.0
                try:
                    backlog_cost = float(im_env.backlog_costs[s, a]) * float(backlog_t)
                except Exception:
                    backlog_cost = 0.0

                profit_t = float(revenue - order_cost - holding_cost - backlog_cost)

                # assets_start snapshot
                try:
                    assets_start = float(assets_start_mat[s, a])
                except Exception:
                    assets_start = float(assets_end - profit_t)

                # explicit clamp delta
                expected_end = assets_start + profit_t
                if (assets_end <= 0.0 + 1e-12) and (expected_end < 0.0):
                    bankruptcy_adjustment = float(assets_end - expected_end)
                else:
                    bankruptcy_adjustment = 0.0

                # readable reasons
                reasons = []
                if order_cost > 0:  reasons.append(f"order_cost-{int(order_cost)}")
                if holding_cost > 0: reasons.append(f"holding_cost-{int(holding_cost)}")
                if backlog_cost > 0: reasons.append(f"backlog_penalty-{int(backlog_cost)}")
                deduct_reasons = ";".join(reasons)

                # flags
                is_bankrupt = bool(assets_end <= 0.0 + 1e-12)
                will_revive = bool(
                    enable_resurrection and ((s, a) in bankrupt_set) and (bankrupt_counts[s, a] < max_bankruptcies)
                )
                if is_bankrupt:
                    bankrupt_cum[s, a] += 1

                # write OBS (end-of-period)
                _obs_writer.writerow({
                    "period": t_end,
                    "stage": int(s),
                    "agent": int(a),
                    "demand": int(demand_t),
                    "inventory": int(inventory_t),
                    "backlog": int(backlog_t),
                    "assets": float(assets_end),
                    "order": int(order_total),
                })

                # write LEDGER (end-of-period)
                identity_diff = float(assets_end - (assets_start + profit_t + bankruptcy_adjustment))
                _ledger_writer.writerow({
                    "period": t_end,
                    "stage": int(s),
                    "agent": int(a),
                    "assets_start": float(assets_start),
                    "revenue": float(revenue),
                    "order_cost": float(order_cost),
                    "holding_cost": float(holding_cost),
                    "backlog_cost": float(backlog_cost),
                    "profit": float(profit_t),
                    "bankruptcy_adjustment": float(bankruptcy_adjustment),
                    "identity_diff": float(identity_diff),
                    "assets_end": float(assets_end),
                    "deduct_reasons": deduct_reasons,
                    "bankrupt": int(bankrupt_cum[s, a]),
                    "revived": int(revived_cum[s, a]),

                })

                # write MEMORY (end-of-period, unified t)
                if enable_memory:
                    order_vec_list = my_vec.astype(int).tolist()
                    reward_t = float(rewards.get(k, 0.0)) if isinstance(rewards, dict) else 0.0
                    mem.add_observation(
                        period=t_end, s=s, a=a,
                        demand=float(demand_t),
                        assets=float(assets_end),
                        backlog=float(backlog_t),
                        inventory=float(inventory_t),
                        order=float(order_total),
                        reward=float(reward_t),
                        order_vec=order_vec_list,
                    )

        # flush both
        _obs_f.flush()
        _ledger_f.flush()

        # --- bankruptcy-triggered reflection & resurrection ---
        if enable_resurrection and bankrupt_now:
            for (s_bk, a_bk) in bankrupt_now:
                bankrupt_counts[s_bk, a_bk] += 1
                cur_cnt = int(bankrupt_counts[s_bk, a_bk])
                over_limit = (cur_cnt > int(max_bankruptcies))

                if enable_memory and enable_bankruptcy_reflection:
                    obs_cnt = len(mem.store[(s_bk, a_bk)])
                    window_n = obs_cnt
                    if window_n > 0:
                        _ = mem.reflect(
                            s=s_bk, a=a_bk,
                            llm_fn=_reflection_llm_fn,
                            period=t_end,
                            reason="bankruptcy_all_history",
                            window_n=window_n
                        )
                        mem.save()

                try:
                    assets_end_before = float(im_env.assets[s_bk, a_bk])
                except Exception:
                    assets_end_before = 0.0

                k_bk = f"stage_{s_bk}_agent_{a_bk}"
                try:
                    backlog_t = int(next_state_dict.get(k_bk, {}).get("backlog", 0))
                except Exception:
                    try:
                        backlog_t = int(getattr(im_env, "backlog")[s_bk, a_bk])
                    except Exception:
                        backlog_t = 0

                policy = str(revive_policy or "").lower()
                if policy == "initial_assets_initinv_plus_backlog":
                    target_assets = float(init_assets0[s_bk, a_bk])
                    target_inventory = int(max(0, int(init_inventories0[s_bk, a_bk]) + int(backlog_t)))
                elif policy in ("fix", "const", "constant"):  # ← 新增显式分支
                    target_assets = float(revive_assets)
                    target_inventory = int(revive_inventory)
                else:
                    target_assets = float(revive_assets)
                    target_inventory = int(revive_inventory)

                if over_limit:
                    try:
                        _resur_writer.writerow({
                            "period": int(t_end),
                            "stage": int(s_bk),
                            "agent": int(a_bk),
                            "bankruptcy_count": cur_cnt,
                            "over_limit": 1,
                            "policy": policy,
                            "assets_end_before": float(assets_end_before),
                            "init_assets": float(init_assets0[s_bk, a_bk]),
                            "init_inventory": int(init_inventories0[s_bk, a_bk]),
                            "backlog_t": int(backlog_t),
                            "revive_assets": float("nan"),
                            "revive_inventory": float("nan"),
                            "new_assets": float("nan"),
                            "new_inventory": float("nan"),
                            "note": "exceed_max_bankruptcies",
                        })
                        _resur_f.flush()
                    except Exception:
                        pass

                    if stop_on_exhausted_resurrections and (not terminate_on_exhaust):
                        terminate_on_exhaust = True
                        exhaust_info = {
                            "period": int(t_end),
                            "stage": int(s_bk),
                            "agent": int(a_bk),
                            "bankruptcy_count": cur_cnt
                        }
                    continue

                new_assets, new_inventory = resurrect_agent(
                    im_env, s=s_bk, a=a_bk, t_end=t_end,
                    target_assets=target_assets,
                    target_inventory=target_inventory,
                    reset_suppliers=reset_suppliers_on_revive
                )
                revived_cum[s_bk, a_bk] += 1

                try:
                    _resur_writer.writerow({
                        "period": int(t_end),
                        "stage": int(s_bk),
                        "agent": int(a_bk),
                        "bankruptcy_count": cur_cnt,
                        "over_limit": 0,
                        "policy": policy,
                        "assets_end_before": float(assets_end_before),
                        "init_assets": float(init_assets0[s_bk, a_bk]),
                        "init_inventory": int(init_inventories0[s_bk, a_bk]),
                        "backlog_t": int(backlog_t),
                        "revive_assets": float(target_assets),
                        "revive_inventory": int(target_inventory),
                        "new_assets": float(new_assets),
                        "new_inventory": int(new_inventory),
                        "note": "auto-resurrect on bankruptcy",
                    })
                    _resur_f.flush()
                except Exception:
                    pass

            early_stop_flag = False

        use_scheduled_reflection = enable_memory and (
                enable_regular_reflection_with_resurrection or (not enable_resurrection)
        )
        if use_scheduled_reflection:
            bankrupt_set_this_period = set(bankrupt_now)
            skip_scheduled_for_bankrupt_this_period = enable_bankruptcy_reflection
            def _last_k_obs(s, a, k):
                try:
                    arr = mem.last_n(s, a, max(k, 6))
                except Exception:
                    arr = mem.store.get((s, a), [])
                try:
                    arr = sorted(arr, key=lambda o: o.period)
                except Exception:
                    pass
                return arr[-k:] if len(arr) >= k else arr

            def _mono_down(xs):
                return len(xs) >= 2 and all(a >= b for a, b in zip(xs, xs[1:])) and any(a > b for a, b in zip(xs, xs[1:]))

            def _mono_up(xs):
                return len(xs) >= 2 and all(a <= b for a, b in zip(xs, xs[1:])) and any(a < b for a, b in zip(xs, xs[1:]))

            scheduled_window = ((period + 1) % mem.cfg.reflect_every) == 0

            for s in range(num_stages):
                for a in range(num_agents_per_stage):
                    if skip_scheduled_for_bankrupt_this_period and (s, a) in bankrupt_set_this_period:
                        continue
                    obs_cnt = len(mem.store[(s, a)])
                    has_min_obs = (obs_cnt >= mem.cfg.min_obs_for_reflection)

                    recent = _last_k_obs(s, a, EMERGENCY_K)
                    assets_seq  = [o.assets  for o in recent]
                    backlog_seq = [o.backlog for o in recent]

                    emergency_reason = ""
                    if len(recent) >= EMERGENCY_K:
                        if _mono_down(assets_seq):  emergency_reason = f"assets_down_{EMERGENCY_K}"
                        elif _mono_up(backlog_seq): emergency_reason = f"backlog_up_{EMERGENCY_K}"

                    if not emergency_reason and len(recent) >= 2:
                        prev, cur = recent[-2], recent[-1]
                        drop_ratio = (prev.assets - cur.assets) / max(1.0, abs(prev.assets))
                        if drop_ratio >= 0.30:
                            emergency_reason = "assets_drop_30pct"

                    do_scheduled = scheduled_window and has_min_obs
                    do_emergency = (emergency_reason != "")

                    print("[REFLECT][CHK]", dict(
                        p=period, s=s, a=a,
                        scheduled=scheduled_window, has_min_obs=has_min_obs,
                        emergency=emergency_reason,
                        assets_seq=assets_seq, backlog_seq=backlog_seq
                    ))

                    if not (do_scheduled or do_emergency):
                        continue

                    reason = "scheduled" if do_scheduled else emergency_reason
                    window_n = mem.cfg.min_obs_for_reflection if do_scheduled else EMERGENCY_K
                    window_n = min(window_n, obs_cnt)

                    try:
                        before = len(mem.rules[(s, a)])
                    except Exception:
                        before = 0

                    try:
                        print("[REFLECT][CALL]", dict(p=period, s=s, a=a, reason=reason, window_n=window_n))
                        new_rules = mem.reflect(
                            s=s, a=a,
                            llm_fn=_reflection_llm_fn,
                            period=period,
                            reason=reason,
                            window_n=window_n,
                        )
                        after = len(mem.rules[(s, a)])
                        added = after - before
                        print("[REFLECT][DONE]", dict(returned=len(new_rules), added=added))
                        try:
                            mem.save()
                        except Exception as e:
                            print(f"[MEM][WARN] save after reflect failed: {e}")
                    except Exception as e:
                        print(f"[REFLECT][ERR] p={period} s={s} a={a}: {e}")

        early_stop_flag = False
        if stop_on_first_bankruptcy and (not enable_resurrection):
            try:
                if hasattr(im_env, "assets"):
                    assets_arr = np.asarray(im_env.assets, dtype=float)
                else:
                    assets_arr = np.full((num_stages, num_agents_per_stage), np.nan, dtype=float)
                    for s in range(num_stages):
                        for a in range(num_agents_per_stage):
                            k = f"stage_{s}_agent_{a}"
                            assets_arr[s, a] = float(next_state_dict.get(k, {}).get("assets", np.nan))

                where_dead = np.argwhere(np.isfinite(assets_arr) & (assets_arr <= 0))
                if where_dead.size > 0:
                    s0, a0 = int(where_dead[0][0]), int(where_dead[0][1])
                    first_bankruptcy_info = {
                        "period": int(period),
                        "stage": s0,
                        "agent": a0,
                        "assets": float(assets_arr[s0, a0]),
                    }
                    try:
                        pd.DataFrame([first_bankruptcy_info]).to_csv(
                            os.path.join(run_dir, "bankruptcy_log.csv"), index=False
                        )
                    except Exception as _e:
                        print("[WARN] fail to save early bankruptcy_log:", _e)
                    print("[EARLY-STOP] first bankruptcy:", first_bankruptcy_info)
                    early_stop_flag = True
            except Exception as _e:
                print("[WARN] early-stop check failed:", _e)
                early_stop_flag = False

        llm_agent_rewards = {}
        round_reward_sum = 0.0
        for (s_id, a_id) in llm_agent_set:
            kk = f"stage_{s_id}_agent_{a_id}"
            if isinstance(rewards, dict) and kk in rewards:
                llm_agent_rewards[kk] = rewards[kk]
                round_reward_sum += rewards[kk]
        episode_reward += round_reward_sum
        print(f"period = {period}")
        print(f"llm_agent_rewards = {llm_agent_rewards}")
        print(f"round_reward_sum = {round_reward_sum}")

        try:
            from utils import visualize_state
            visualize_state(env=im_env, rewards=rewards, t=period, save_prefix=os.path.join(config_name, ts, "viz"))
        except Exception:
            pass

        try:
            _ = _write_chat_summary(period, total_chat_summary, round_reward_sum)
        except Exception as e:
            print(f"[WARN] fail to save chat summary: {e}")

        all_action_order_dicts[period + 1] = action_order_dict
        all_action_sup_dicts[period + 1] = action_sup_dict
        all_action_dem_dicts[period + 1] = action_dem_dict
        all_action_price_dicts[period + 1] = action_price_dict

        if stop_on_first_bankruptcy and (not enable_resurrection) and early_stop_flag:
            break
        if terminate_on_exhaust and stop_on_exhausted_resurrections:
            print("[EARLY-STOP] exceeded max bankruptcies; terminating.", exhaust_info)
            try:
                stop_fn = os.path.join(run_dir, "stop_exhausted_resurrections.csv")
                with open(stop_fn, "w", newline="", encoding="utf-8-sig") as f:
                    wr = csv.DictWriter(f, fieldnames=["period", "stage", "agent", "bankruptcy_count"])
                    wr.writeheader()
                    wr.writerow(exhaust_info)
                print("[Saved]", stop_fn)
            except Exception as _e:
                print("[WARN] fail to save stop_exhausted_resurrections:", _e)
            break

    try:
        mem.save()
    except Exception as e:
        print(f"[MEM][WARN] final save failed: {e}")

    print(f"episode_reward = {episode_reward}")
    print(f"api_cost = {api_cost}")
    print("[MEM] saved to:", dir_mem)
    print("=" * 80)

    try:
        _obs_f.close()
        print("[Saved]", OBS_PATH)
    except Exception as e:
        print("[WARN] close obs file failed:", e)

    try:
        _ledger_f.close()
        print("[Saved]", LEDGER_PATH)
    except Exception as e:
        print("[WARN] close ledger file failed:", e)

    try:
        _resur_f.close()
        print("[Saved]", RESUR_PATH)
    except Exception as e:
        print("[WARN] close resurrection file failed:", e)
    saved_bankruptcy = False
    if hasattr(im_env, "bankruptcy_log") and len(getattr(im_env, "bankruptcy_log", [])) > 0:
        try:
            df_bk = pd.DataFrame(im_env.bankruptcy_log, columns=["period", "stage", "agent", "assets"])
            df_bk.to_csv(os.path.join(run_dir, "bankruptcy_log.csv"), index=False)
            print("[Saved]", os.path.join(run_dir, "bankruptcy_log.csv"))
            saved_bankruptcy = True
        except Exception as e:
            print("[WARN] fail to save bankruptcy_log:", e)

    if not saved_bankruptcy and len(records):
        try:
            df = pd.DataFrame(records, columns=["period", "stage", "agent", "demand", "inventory", "backlog", "assets", "order"])
            df["assets"] = pd.to_numeric(df["assets"], errors="coerce")
            dead = df[df["assets"] <= 0].sort_values(["stage", "agent", "period"])
            firsts = dead.groupby(["stage", "agent"], as_index=False).first()[["period", "stage", "agent", "assets"]]
            if not firsts.empty:
                firsts.to_csv(os.path.join(run_dir, "bankruptcy_log.csv"), index=False)
                print("[Saved-derived]", os.path.join(run_dir, "bankruptcy_log.csv"))
        except Exception as e:
            print("[WARN] fail to derive bankruptcy_log:", e)

    if plot_after:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            obs = pd.read_csv(OBS_PATH)
            led = pd.read_csv(LEDGER_PATH)
            try:
                resu = pd.read_csv(RESUR_PATH)
            except Exception:
                resu = pd.DataFrame(columns=["period", "stage", "agent"])

            for df in (obs, led, resu):
                if len(df) == 0: continue
                for c in set(["period", "stage", "agent"]).intersection(df.columns):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            # per-agent plots
            for (s, a), g_led in led.groupby(["stage", "agent"]):
                g_led = g_led.sort_values("period").copy()
                g_led["bk_inc"] = g_led["bankrupt"].diff().fillna(g_led["bankrupt"]).astype("Int64")
                g_led["rv_inc"] = g_led["revived"].diff().fillna(g_led["revived"]).astype("Int64")
                g_obs = obs[(obs["stage"] == s) & (obs["agent"] == a)].sort_values("period")
                g_res = resu[(resu["stage"] == s) & (resu["agent"] == a)].sort_values("period")

                if g_led.empty or g_obs.empty:
                    continue

                # Figure 1: assets identity & events
                plt.figure(figsize=(9, 4))
                plt.plot(g_led["period"], g_led["assets_start"], marker="o", label="assets_start")
                plt.plot(g_led["period"], g_led["assets_end"], marker="o", label="assets_end")
                bad = g_led[g_led["identity_diff"].abs() > 1e-6]
                if not bad.empty:
                    plt.scatter(bad["period"], bad["assets_end"], s=36, label="identity_diff≠0")

                for p in g_led.loc[g_led["bk_inc"] > 0, "period"].astype(int).tolist():
                    plt.axvline(p, color="r", linestyle="-", alpha=0.5, linewidth=1)
                for p in g_led.loc[g_led["rv_inc"] > 0, "period"].astype(int).tolist():
                    plt.axvline(p, color="g", linestyle="--", alpha=0.7, linewidth=1)

                plt.title(f"Agent s{s}-a{a} | assets & events")
                plt.xlabel("period");
                plt.ylabel("assets");
                plt.legend()
                out = os.path.join(dir_plots, f"s{s}_a{a}_assets_events.png")
                plt.tight_layout();
                plt.savefig(out);
                plt.close()

                # Figure 2: ops panel
                plt.figure(figsize=(9, 4))
                ax1 = plt.gca()
                ax1.plot(g_obs["period"], g_obs["demand"], label="demand")
                ax1.plot(g_obs["period"], g_obs["order"], label="order")
                ax1.set_xlabel("period");
                ax1.set_ylabel("flow")

                ax2 = ax1.twinx()
                ax2.plot(g_obs["period"], g_obs["inventory"], label="inventory", linestyle="--")
                ax2.plot(g_obs["period"], g_obs["backlog"], label="backlog", linestyle="--")
                ax2.set_ylabel("stock")

                for p in g_led.loc[g_led["bk_inc"] > 0, "period"].astype(int).tolist():
                    ax1.axvline(p, color="r", linestyle="-", alpha=0.3, linewidth=1)
                for p in g_led.loc[g_led["rv_inc"] > 0, "period"].astype(int).tolist():
                    ax1.axvline(p, color="g", linestyle="--", alpha=0.4, linewidth=1)

                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                plt.legend(h1 + h2, l1 + l2, loc="upper left")
                plt.title(f"Agent s{s}-a{a} | ops panel")
                out = os.path.join(dir_plots, f"s{s}_a{a}_ops.png")
                plt.tight_layout();
                plt.savefig(out);
                plt.close()

            print(f"[Saved] enhanced plots -> {dir_plots}")
        except Exception as e:
            print("[WARN] plotting failed:", e)

    if return_meta:
        return {"episode_reward": float(episode_reward), "api_cost": float(api_cost), "run_dir": run_dir}
    return float(episode_reward)














