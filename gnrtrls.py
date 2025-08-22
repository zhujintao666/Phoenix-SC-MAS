

import os, json, time, pathlib
from openai import OpenAI


MODEL_NAME = "deepseek-r1"
JSON_IN   = pathlib.Path("src/random120.json")
JSON_OUT  = pathlib.Path("src/random120_filled.json")


GOLDEN_RULES = """
1. Prioritize Profitability: Always ensure that the income from fulfilling downstream orders outweighs the combined costs of holding, backlog, and order placement.
2. Balance Inventory Levels: Maintain enough stock to meet demand over the lead time but avoid excessive holding costs.
3. Clear Backlogs Quickly: Backlogs incur high costs and reduce income potential. Address them early to improve profitability.
4. Optimize Order Timing and Quantity: Place orders strategically to avoid frequent small orders (high purchase cost) or large orders (high holding cost).
5. Preserve Financial Stability: Ensure that your assets are sufficient to place necessary orders while maintaining cash flow for future turns.
6. Consider Lead Time Variability: Account for the range of lead times to avoid delayed replenishments.
7. Use Safety Stock: Maintain a buffer to handle demand and lead time variability but avoid excessive stockpiling.
"""

PROMPT_TEMPLATE = """Suppose you are the manager of a company in the supply chain game. The game lasts for multiple turns and you can place orders at each turn. Given the inventory level, backlog, downstream demand, lead time, order price and your assets, propose some golden rules to govern your decisions on order placement under different scenarios. The income comes from the fulfilled downstream orders and the costs come from holding cost, purchase cost, and backlog cost.
{rules}
Scenario:
- Demand: {demand}
- Inventory: {inventory}
- Backlog: {backlog}
- Assets (units affordable): {assets}
- Lead time: 2
- Order price: 1
Follow the golden rules and give your action to this scenario following this format. E.g.,[8].
"""


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def query_llm(demand, inventory, backlog, assets) -> str:
    prompt = PROMPT_TEMPLATE.format(
        rules=GOLDEN_RULES.strip(),
        demand=demand, inventory=inventory,
        backlog=backlog, assets=assets
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

def parse_order_qty(reply: str) -> int:
    import re
    match = re.search(r"\d+", reply)
    if match:
        return int(match.group())
    raise ValueError(f"无法解析订单量: {reply}")

def main():
    data = json.loads(JSON_IN.read_text(encoding="utf-8"))

    for i, row in enumerate(data, 1):

        reply = query_llm(row["demand"], row["inventory"], row["backlog"], row["assets"])
        try:
            qty = parse_order_qty(reply)
        except Exception as e:
            print(f"[WARN] idx {i} failed，reply：{reply}\n{e}")
            qty = 0     

        row["order"] = qty
        print(f"{i:03d}/{len(data)} -> order = {qty}")

    JSON_OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("All done! Saved to", JSON_OUT)

if __name__ == "__main__":
    main()
