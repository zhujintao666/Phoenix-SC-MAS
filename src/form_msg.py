from utils import get_state_description, get_demand_description
from env import InventoryManagementEnv

PROMPT_SET = "strict_capped.v1"
# [HIGH PRIORITY] The following Task3 + GoldenRule are the recovered versions that performed best.

task1_msg = (
    "Task1: Do you want to remove any upstream suppliers?\n"
    "Consider: lead time, per-unit order cost, MOQ, reliability/capacity, and the impact on safety stock & holding cost.\n"
    "Return EXACTLY one JSON list (e.g., [0,1] or []).\n"
)

task2_msg = (
    "Task2: Do you want to add new supplier(s) from available options?\n"
    "Consider: lead time reduction (lower holding cost), cost, capacity/reliability, MOQs, and cash.\n"
    "Return EXACTLY one JSON list (e.g., [2,3] or []).\n"
)

# [HIGH PRIORITY] Recovered core: strict JSON + horizon/base-stock math + hard caps
task3_msg = (
    "Task3: Decide the order quantity to place with each supplier for THIS ROUND only.\n"
    "Return EXACTLY ONE JSON object on a single line (no extra text):\n"
    '  {"orders":[q1,q2,...,qN], "why":"<1-2 short sentences>"}\n'
    "\nHard numeric caps (you MUST satisfy ALL):\n"
    "  - Non-negative INTEGERS only; array length N = your upstream supplier count (fixed order).\n"
    "  - Production cap: total orders ≤ this-round production capacity.\n"
    "  - Supplier cap: respect each supplier capacity; total ≤ sum of supplier capacities; respect MOQs.\n"
    "  - Cash cap: keep a cash buffer; define\n"
    "      unit_total_cost = unit_order_cost + unit_production_cost (+ shipping if any)\n"
    "      cash_cap = floor((assets_now * (1 - 0.20)) / unit_total_cost)  # keep ≥20% assets as buffer\n"
    "    and require total orders ≤ cash_cap.\n"
    "  - Ramp cap (anti-spike): total orders ≤ ceil(last_total_orders * 1.30).\n"
    "\nPlan over effective lead time L (avoid double-counting inbound):\n"
    "  inbound_L_total = sum(arrivals_in_next_L_periods)\n"
    "  coverage_gap = max(0, ExpectedDemand_over_next_L + CurrentBacklog - OnHandInventory - inbound_L_total)\n"
    "Signal to follow (horizon rule):\n"
    "  target = clamp_{≥0}(coverage_gap), then cap by production/supplier/cash/ramp limits above.\n"
    "\nAllocation hint:\n"
    "  Prefer lower lead time and lower unit cost; smooth across rounds (avoid one-shot spikes).\n"
    "If some numbers are missing, be conservative: keep the ≥20% cash buffer and respect ramp cap.\n"
)

task4_msg = (
    "Task4: Set product price for THIS ROUND.\n"
    'Return EXACTLY ONE JSON: {"price": P, "why":"<1-2 short sentences>"} or {"prices":[p1,...], "why":"..."}\n'
    "Ensure price ≥ (upstream order cost + production cost + a margin). Consider h/p trade-offs and competitor pricing if available.\n"
)

gold_rule_msg = (
    "\nStrict output rules:\n"
    "- For Task3, output ONLY the JSON on one line; no explanations before/after.\n"
    "- Do NOT reveal your chain-of-thought; think silently and return final answer only.\n"
    "- The JSON MUST satisfy all caps (production, supplier, cash, ramp). If target exceeds caps, reduce to the caps.\n"
    '- If you order zero, still return full-length zeros: {"orders":[0,0,...,0]}.\n'
    "- Keep 'why' ≤2 short sentences (e.g., which cap bound your decision: cash/production/supplier/ramp/coverage).\n"
    "\n[HIGH PRIORITY] Golden rule (horizon-based base-stock): Over effective lead time L, target\n"
    "  NewOrders ≈ clamp_{≥0}( ExpectedDemand_over_next_L + CurrentBacklog + SafetyStock(L)\n"
    "                           - OnHandInventory - InboundOverNext_L ).\n"
    "Notes:\n"
    "- ExpectedDemand_over_next_L from recent sales trends.\n"
    "- InboundOverNext_L = deliveries scheduled within L (arrivals window).\n"
    "- SafetyStock(L) reflects holding cost (h) vs stockout/backlog penalty (p); if estimates exist, use a higher service level when p >> h\n"
    "  (newsvendor-style critical fractile ≈ p/(p+h)); otherwise use a conservative buffer.\n"
    "- Account for holding cost: avoid inventory far above the base-stock target when h is high.\n"
    "- Respect constraints: supplier/production capacity, MOQ, and maintain the cash buffer.\n"
    "- Smooth orders over rounds; do NOT place large one-shot orders.\n"
    "\nInterpretation: order_costs/holding_cost/backlog_cost are per-unit (per period for holding/backlog).\n"
)

least_lead_time = (
    "Task: Which upstream compan(ies) has the least lead time to you? "
    "Provide the answer in brackets (e.g., [agent4])."
)
lowest_order_cost = (
    "Task: Which upstream compan(ies) has the lowest order cost? "
    "Provide the answer in brackets (e.g., [agent5])."
)
expected_demand = (
    "Task: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]). \n"
)


def generate_msg(im_env: InventoryManagementEnv, shutdown_list: list, recovery_list: list, enable_graph_change: bool,
                 enable_price_change: bool, action_order_dict: dict,
                 past_req_orders: list, stage_state: dict, period: int, stage_id: int, cur_agent_id: int):
    agent_name = f"stage_{stage_id}_agent_{cur_agent_id}"
    message = (
        f"Now this is the round {period}, "
        f"and you are {agent_name} at the stage {stage_id}: {im_env.stage_names[stage_id]} in the supply chain. "
    )

    for event in im_env.emergent_events[im_env.period]:
        if event == "sudden_shutdown":
            shutdown_agents = []
            for (st_stage_id, st_agent_id) in shutdown_list:
                shutdown_agents.append(f"stage_{st_stage_id}_agent_{st_agent_id}")
            shutdown_agents = ", ".join(shutdown_agents)
            shutdown_message = (
                f"There is a sudden shutdown event. "
                f"Those agent(s) are closed since now: {shutdown_agents}\n\n"
            )
            message += shutdown_message
        if event == "recovery":
            recovered_agents = []
            for (rec_stage_id, rec_agent_id) in recovery_list:
                recovered_agents.append(f"stage_{rec_stage_id}_agent_{rec_agent_id}")
            recovered_agents = ", ".join(recovered_agents)
            recover_message = (
                f"There is a recovery event. "
                f"Those agent(s) are re-open since now: {recovered_agents}\n\n"
            )
            message += recover_message
    message += f"Given your current state:\n{get_state_description(state=stage_state, past_req_orders=past_req_orders, G=im_env.sc_graph.G, agent_name=agent_name, state_format=im_env.state_format, enable_graph_change=enable_graph_change)}\n\n"
    if stage_id != 0:
        down_order = []
        for down_agent_id in range(im_env.num_agents_per_stage):
            dr = action_order_dict[f'stage_{stage_id - 1}_agent_{down_agent_id}'][cur_agent_id]
            if dr != 0:
                down_order.append(f"from agent{down_agent_id}: {dr}")
        down_order = "; ".join(down_order)
        downstream_order = f"Your downstream order from the agents at stage {stage_id - 1} for this round is: {down_order}. "
        message += f"{downstream_order}\n"
    assets_now = float(im_env.assets[stage_id, cur_agent_id])
    asset_hint = (
        f"\n**Available assets this round:** {assets_now:.0f}\n"
        f"Order prudently. Base your orders on lead time, backlog, inventory and recent sales; "
        f"avoid one-shot large orders and keep a reasonable cash buffer.\n"
    )
    message += asset_hint

    state_info = message

    # message += get_lead_time_task()
    # message += get_order_cost_task()
    message += get_decision_task(stage=stage_id, im_env=im_env, enable_graph_change=enable_graph_change,
                                 enable_price_change=enable_price_change)
    # message += get_expected_demand_task()

    return message, state_info


def get_lead_time_task():
    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += least_lead_time

    return task_msg


def get_order_cost_task():
    task_msg = "\nPlease answer the question based on your understanding of the given supply chain network.\n"
    task_msg += lowest_order_cost

    return task_msg


def get_expected_demand_task():
    task_msg = "\nTask: What is your estimated demand from downstream in the next round? Provide the answer in brackets (e.g., [10]).\n"

    return task_msg


def get_decision_task(stage: int, im_env, enable_graph_change: bool, enable_price_change: bool):
    task_msg = ""
    num_tasks = 0

    if stage < im_env.num_stages - 1 and enable_graph_change:  # Ask for supplier updates if it is allowed or it is not a manufacturer
        task_msg += f"{task1_msg}\n"
        task_msg += f"{task2_msg}\n"
        num_tasks += 2

    # Ask for order quantity
    # task_msg += f"{expected_demand}\n"
    task_msg += f"{task3_msg}\n"
    num_tasks += 1

    # Ask for price decision
    if enable_price_change:
        task_msg += f"{task4_msg}\n"
        num_tasks += 1

    task_msg = f"There are {num_tasks} task(s) for you to make decision(s). \n\n" + task_msg

    task_msg += f"{gold_rule_msg}\n"

    return task_msg


