import os, json, re, pathlib
from openai import OpenAI

MODEL_NAME = "qwen-plus"
JSON_IN = pathlib.Path("src/random120.json")

GOLDEN_RULES = '''
1. Prioritize Profitability: Always ensure that the income from fulfilling downstream orders outweighs the combined costs of holding, backlog, and order placement.
2. Balance Inventory Levels: Maintain enough stock to meet demand over the lead time but avoid excessive holding costs.
3. Clear Backlogs Quickly: Backlogs incur high costs and reduce income potential. Address them early to improve profitability.
4. Optimize Order Timing and Quantity: Place orders strategically to avoid frequent small orders (high purchase cost) or large orders (high holding cost).
5. Preserve Financial Stability: Ensure that your assets are sufficient to place necessary orders while maintaining cash flow for future turns.
6. Consider Lead Time Variability: Account for the range of lead times to avoid delayed replenishments.
7. Use Safety Stock: Maintain a buffer to handle demand and lead time variability but avoid excessive stockpiling.
'''


PROMPT_TEMPLATE = ('''
Suppose you are the manager of a company in the supply chain game. The game lasts for multiple turns and you can place orders at each turn. Given the inventory level, backlog, downstream demand, lead time, order price and your assets, propose some golden rules to govern your decisions on order placement under different scenarios. The income comes from the fulfilled downstream orders and the costs come from holding cost, purchase cost, and backlog cost.
{rules}
Scenario:
- Demand: {demand}
- Inventory: {inventory}
- Backlog: {backlog}
- Assets (units affordable): {assets}
- Lead time: 2
- Order price: 1
Follow the rules and give your action to this scenario.
Return the result in the format [order_qty] only without any analysis or explanation,e.g.[8].
'''
)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def build_prompt(row):
    return PROMPT_TEMPLATE.format(
        rules=GOLDEN_RULES,
        demand=row["demand"],
        inventory=row["inventory"],
        backlog=row["backlog"],
        assets=row["assets"]
    )

def extract_qty(reply: str) -> int | None:
    m = re.search(r"\[(\d+)\]", reply) or re.search(r"\d+", reply)
    return int(m.group(1)) if m else None

def main():
    data = json.loads(JSON_IN.read_text("utf-8"))

    for idx, row in enumerate(data, 1):
        prompt = build_prompt(row)

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"reasoning": "disabled"}
        )
        reply = resp.choices[0].message.content.strip()
        qty = extract_qty(reply)
        row["order"] = qty if qty is not None else -1

        print(f"{idx:03d}/{len(data)} -> order = {qty}")
        # pathlib.Path("src/full_llm_answers.txt").write_text(f"{idx:03d}\n{reply}\n", append=True)

    pathlib.Path("src/random120_filled.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )
    print("All done! Saved to src/random120_filled.json")


if __name__ == "__main__":
    main()