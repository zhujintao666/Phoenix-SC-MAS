from utils import get_state_description, get_demand_description
from env import InventoryManagementEnv


task3_msg = (
    "Task3: Decide the order quantity to place with each supplier for THIS ROUND only.\n"
    "Return EXACTLY ONE JSON on a single line (no extra text). Include a very short 'why'.\n"
    '  {"orders":[q1,q2,...,qN], "why":"<≤2 short clauses>"}\n'
    "\nRules:\n"
    "- Use EnvContext as ground truth.\n"
    "- Non-negative INTEGERS; length N = number of upstream suppliers (fixed order).\n"
    "- Base your total on the Golden rule target below; smooth rather than copy the last round.\n"
)

gold_rule_msg = (
    "[Golden rule over lead time L]\n"
    "  target = max(0, forecast_over_L + backlog − on_hand − inbound_within_L) + safety_buffer.\n"
    "Estimation hints:\n"
    "  • forecast_over_L: average or simple trend of recent_sales.\n"
    "  • inbound_within_L: count ONLY arrivals scheduled within L (avoid double-counting).\n"
    "  • safety_buffer: higher when backlog penalty ≫ holding cost; otherwise keep modest.\n"
    "\nDistribution rule (if multiple suppliers):\n"
    "  • You MUST place orders to AT MOST 2 upstream suppliers in this period; set all other supplier quantities to 0.\n"
    "  • Prefer splitting across 1–2 suppliers evenly or proportionally to reliability/cost; avoid sending all to one unless clearly justified.\n"
    "  • Do not output all zeros unless BOTH demand and backlog are strictly zero.\n"
    "\nStrict output rules:\n"
    "- Output ONLY the JSON on one line; no text before/after.\n"
    "- The JSON must be: {\"orders\": [int,...,int], \"why\": \"...\"}.\n"
    "- Length of 'orders' == number of upstream candidates; all entries are non-negative integers.\n"
    "- At most TWO non-zero entries in 'orders'.\n"
    "- Keep 'why' ≤ 2 short clauses (e.g., coverage/trend/buffer).\n"
    "- If zero, still return full-length zeros: {\"orders\":[0,0,...,0]}.\n"
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


def generate_msg(im_env: InventoryManagementEnv, enable_graph_change: bool,
                 enable_price_change: bool, action_order_dict: dict,
                 past_req_orders: list, stage_state: dict, period: int, stage_id: int, cur_agent_id: int):
    agent_name = f"stage_{stage_id}_agent_{cur_agent_id}"
    message = (
        f"Now this is the round {period}, "
        f"and you are {agent_name} at the stage {stage_id}: {im_env.stage_names[stage_id]} in the supply chain. "
    )

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