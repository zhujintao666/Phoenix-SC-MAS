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

    CLEAR_HISTORY = True
    WARMUP_PERIODS = 0
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

    try:
        from utils import visualize_state
        visualize_state(env=im_env, rewards={}, t=-1, save_prefix=os.path.join(config_name, ts, "viz"))
    except Exception:
        pass

    mem = ExperienceMemory(
        num_stages=num_stages,
        num_agents_per_stage=num_agents_per_stage,
        config=MemoryConfig(
            reflect_every=8,
            min_obs_for_reflection=8,
            save_dir=dir_mem,
        )
    )
    mem.cfg.k_neighbors = 3

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
                    sup_vec = (sup_vec > 0).astype(int)
                    if stage_id < num_stages - 1:
                        try:
                            sup_alive_mask = np.asarray(running_prev[stage_id + 1], dtype=int)
                            Lg = min(len(sup_vec), len(sup_alive_mask))
                            sup_vec[:Lg] = (sup_vec[:Lg] * (sup_alive_mask[:Lg] > 0).astype(int))
                        except Exception:
                            pass
                    if sup_vec.sum() == 0 and (downstream_need > 0 or cur_backlog > 0):
                        print(f"[WARN][SUPPLY-CUTOFF] p={period} {key}: need={downstream_need}, backlog={cur_backlog}, but no suppliers available")

                    L_eff = _eff_lead_time(im_env, stage_id, agent_id, sup_vec)
                    last_vecs, last_totals, inbound_vec, inbound_total = _collect_last_orders(
                        history_dict=all_action_order_dicts,
                        period=period, key=key, L=L_eff, vec_len=num_agents_per_stage
                    )
                    coverage_proxy = (cur_inventory + inbound_total) / max(1.0, float(cur_demand))

                    lead_card = [
                        "LeadTimeContext:",
                        f"- lead_time = {L_eff} periods",
                        f"- recent orders totals (t-1 → t-{L_eff}): {last_totals}",
                        f"- inbound_next_vec (≈ orders at t-{L_eff}): {inbound_vec.tolist()} (total={inbound_total})",
                        f"- coverage proxy ≈ (inventory + inbound_next_total) / max(1, demand) = {coverage_proxy:.2f}",
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

                    parsed_vec = np.array([max(0, int(x)) for x in np.asarray(parsed_vec).ravel()], dtype=int)
                    if len(parsed_vec) != num_agents_per_stage:
                        parsed_vec = np.zeros(num_agents_per_stage, dtype=int)
                    parsed_vec = (np.maximum(0, parsed_vec) * sup_vec).astype(int)

                    total_chat_summary += (
                        f"=== {key} | period={period} ===\n"
                        f"[PROMPT SENT]\n{msg0}\n"
                        f"[REPLY]\n{reply_text or '[EMPTY]'}\n"
                        f"[PARSED]\norders={parsed_vec.tolist()} | reason={reason_text}\n\n"
                    )
                    safe_txt = (reply_text or "")[:200].replace("\n", " ")
                    print(f"[DBG][LLM] p={period} {key} parsed_orders={parsed_vec.tolist()} raw='{safe_txt}'")

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

        orders_vec_dict = {
            k: (np.asarray(v, dtype=int) if isinstance(v, (list, np.ndarray)) else np.zeros(num_agents_per_stage, dtype=int))
            for k, v in action_order_dict.items()
        }

        for s in range(num_stages):
            for a in range(num_agents_per_stage):
                k = f"stage_{s}_agent_{a}"
                alive_prev = bool(running_prev[s, a] == 1)
                if not alive_prev:
                    continue

                if s == 0:
                    try:
                        demand_t = int(getattr(im_env, "demands", [0]*(period+1))[period])
                    except Exception:
                        demand_t = int(next_state_dict.get(k, {}).get("demand", 0))
                else:
                    demand_t = 0
                    for down_agent in range(num_agents_per_stage):
                        down_key = f"stage_{s-1}_agent_{down_agent}"
                        vec = orders_vec_dict.get(down_key, np.zeros(num_agents_per_stage, dtype=int))
                        if a < len(vec):
                            demand_t += int(vec[a])

                inventory_t = int(next_state_dict.get(k, {}).get("inventory", 0))
                backlog_t   = int(next_state_dict.get(k, {}).get("backlog", 0))
                try:
                    assets_t = float(im_env.assets[s, a])
                except Exception:
                    assets_t = float(next_state_dict.get(k, {}).get("assets", 0.0))

                my_vec = orders_vec_dict.get(k, np.zeros(num_agents_per_stage, dtype=int))
                order_total = int(np.sum(my_vec))
                order_vec_list = my_vec.astype(int).tolist()
                reward_t = float(rewards.get(k, 0.0)) if isinstance(rewards, dict) else 0.0

                mem.add_observation(
                    period=period, s=s, a=a,
                    demand=float(demand_t),
                    assets=float(assets_t),
                    backlog=float(backlog_t),
                    inventory=float(inventory_t),
                    order=float(order_total),
                    reward=float(reward_t),
                    order_vec=order_vec_list,
                )
                records.append([period, s, a, demand_t, inventory_t, backlog_t, assets_t, order_total])

                _obs_writer.writerow({
                    "period": int(period),
                    "stage": int(s),
                    "agent": int(a),
                    "demand": int(demand_t),
                    "inventory": int(inventory_t),
                    "backlog": int(backlog_t),
                    "assets": float(assets_t),
                    "order": int(order_total),
                })
        _obs_f.flush()

        if enable_memory:
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
            df = pd.read_csv(OBS_PATH)
            for c in ["period","stage","agent","demand","inventory","backlog","assets","order"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna().copy()
            df[["period","stage","agent"]] = df[["period","stage","agent"]].astype(int)
            for (s,a), g in df.groupby(["stage","agent"]):
                g = g.sort_values("period")
                plt.figure()
                plt.plot(g["period"], g["demand"],   label="demand")
                plt.plot(g["period"], g["inventory"],label="inventory")
                plt.plot(g["period"], g["backlog"],  label="backlog")
                plt.plot(g["period"], g["assets"],   label="assets")
                plt.plot(g["period"], g["order"],    label="order")
                plt.xlabel("period"); plt.ylabel("value"); plt.legend()
                out = os.path.join(dir_plots, f"stage{s}_agent{a}.png")
                plt.tight_layout(); plt.savefig(out); plt.close()
            print(f"[Saved] agent plots -> {dir_plots}")
        except Exception as e:
            print("[WARN] plotting failed:", e)

    if return_meta:
        return {"episode_reward": float(episode_reward), "api_cost": float(api_cost), "run_dir": run_dir}
    return float(episode_reward)












