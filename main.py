import os
import sys
import numpy as np
from tqdm.auto import tqdm
from autogen import ConversableAgent

sys.path.append('src')

from env import env_creator
from config import env_configs, get_env_configs
from llm_config import llm_config_list
import utils
from mas_model import create_agents, run_simulation
from utils import get_demand_description, clear_dir


np.random.seed(42)
DRAW_FIGS = False
NUM_EPISODES = 25
ENV_NAME = "llm_sampling"
ENABLE_MEMORY = True
USE_RANDOM_ENV_SEED = True

if not DRAW_FIGS:
    utils.visualize_state = lambda *args, **kwargs: None


config_list = llm_config_list

os.makedirs(f"results/{ENV_NAME}", exist_ok=True)

run_id = utils.time_str() if hasattr(utils, "time_str") else None
if run_id is None:
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")

run_dir = f"results/{ENV_NAME}/{run_id}"
os.makedirs(run_dir, exist_ok=True)

os.makedirs(f"env/{ENV_NAME}", exist_ok=True)
clear_dir(f"env/{ENV_NAME}")

env_config_base = get_env_configs(env_configs=env_configs[ENV_NAME])
print(env_config_base["demand_dist"])
print(get_demand_description(env_config_base["demand_fn"]))


def tune_llm_sampling(stage_agents):
    for ag in stage_agents:
        cfg = getattr(ag, "llm_config", None)
        if not isinstance(cfg, dict):
            continue
        cfg.setdefault("temperature", 0.2)
        cfg.setdefault("max_retries", 2)
        ag.llm_config = cfg

user_proxy = ConversableAgent(
    name="UserProxy",
    llm_config=False,
    human_input_mode="NEVER",
)

rewards = []

for r in tqdm(range(NUM_EPISODES), desc="Episodes"):
    print("\n\n=== New episode starts ===")

    env_config_i = get_env_configs(env_configs=env_configs[ENV_NAME])
    if USE_RANDOM_ENV_SEED:
        try:
            env_config_i["seed"] = int(np.random.randint(1e9))
        except Exception:
            pass
    im_env = env_creator(env_config_i)

    stage_agents = create_agents(
        stage_names=env_config_i["stage_names"],
        num_agents_per_stage=env_config_i['num_agents_per_stage'],
        llm_config={'config_list': config_list},
    )
    tune_llm_sampling(stage_agents)

    config_name = f"{ENV_NAME}/{run_id}"

    reward = run_simulation(
        im_env=im_env,
        user_proxy=user_proxy,
        stage_agents=stage_agents,
        config_name=config_name,
        round=r,
        run_tag=f"{run_id}-ep{r:03d}",
        enable_memory=True,
        enable_knn_suggest=False,
        plot_after=True,  
        return_meta=False
    )

    rewards.append(float(reward))
    print(f"[EPISODE {r}] reward = {reward}")


mean_reward = float(np.mean(rewards)) if len(rewards) else 0.0
std_reward = float(np.std(rewards)) if len(rewards) else 0.0

print(f"Rewards: {rewards}")
print(f"Mean Episode Reward: {mean_reward}")
print(f"Standard Deviation of Episode Reward: {std_reward}")

try:
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"ENABLE_MEMORY = {ENABLE_MEMORY}\n")
        f.write(f"NUM_EPISODES  = {NUM_EPISODES}\n")
        f.write(f"USE_RANDOM_ENV_SEED = {USE_RANDOM_ENV_SEED}\n")
        f.write(f"Rewards: {rewards}\n")
        f.write(f"Mean Episode Reward: {mean_reward}\n")
        f.write(f"Std  Episode Reward: {std_reward}\n")
    print(f"[Saved] {os.path.join(run_dir, 'summary.txt')}")
except Exception as e:
    print("[WARN] failed to save run summary:", e)
