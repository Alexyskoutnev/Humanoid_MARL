import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict

from Humanoid_MARL.agent.ppo.train_torch import train
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import load_reward_config
from Humanoid_MARL import CONFIG_TRAIN, CONFIG_NETWORK, CONFIG_REWARD


def debug_config(config: Dict) -> Dict:
    update_config = {
        "unroll_length": 1,
        "num_minibatches": 1,
        "num_envs": 1,
        "batch_size": 1,
        "unroll_length": 1,
    }
    return config.update(update_config)


def main():
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    print(f"USING GPU {gpu_index}")
    env_name = "humanoids"
    project_name = f"MARL_ppo_{env_name}_or_run"
    # ================ Config ================
    with open(CONFIG_TRAIN, "r") as f:
        config = yaml.safe_load(f)
        if config["debug"]:
            debug_config(config)
    env_config = load_reward_config(CONFIG_REWARD, env_name)
    # ================ Config ================
    # ================ Logging ===============
    if not config["debug"]:
        gpu_info = f"GPU {gpu_index} | env_name {env_name}"
        logger = WandbLogger(project_name, config=config, notes=gpu_info)
        config["logger"] = logger
    # ================ Progress Function ================
    xdata = []
    ydata = []
    eval_sps = []
    train_sps = []
    times = [datetime.now()]

    def progress(num_steps, metrics, path="./data/ppo", name="ppo_training_plot.png"):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"].cpu())
        eval_sps.append(metrics["speed/eval_sps"])
        train_sps.append(metrics["speed/sps"])
        plt.xlim([0, config["num_timesteps"]])
        plt.ylim([0, 10_000])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_name = f"{timestamp}_{name}"
        PLT_SAVE_PATH = os.path.join(path, timestamped_name)
        plt.savefig(PLT_SAVE_PATH)

    # ================ Progress Function ================
    if config["debug"]:
        train(**config, progress_fn=None, env_config=env_config)
    else:
        train(**config, progress_fn=progress, env_config=env_config)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")


if __name__ == "__main__":
    main()
