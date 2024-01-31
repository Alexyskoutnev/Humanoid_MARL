import os
from datetime import datetime
import numpy as np
import wandb
import matplotlib.pyplot as plt

from Humanoid_MARL.agent.ppo.train_torch import train


def main():
    # ================ Config ================
    config = {
        'num_timesteps': 100_000_000,
        'eval_frequency': 10,
        'episode_length': 1000,
        'unroll_length': 10,
        'num_minibatches': 32,
        'num_update_epochs': 8,
        'discounting': 0.97,
        'learning_rate': 3e-3,
        'entropy_cost': 1e-2,
        'num_envs': 2048,
        'batch_size': 512,
        'env_name': "humanoids",
        'render' : False,
        'device' : 'cuda',
        'debug' : False
    }
    # ================ Config ================
    # ================ Logging ===============
    if not config['debug']:
        wandb.init(project="MARL-Humanoid",
                    config=config)
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
        plt.xlim([0, config['num_timesteps']])
        plt.ylim([0, 6000])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_name = f"{timestamp}_{name}"
        PLT_SAVE_PATH = os.path.join(path, timestamped_name)
        plt.savefig(PLT_SAVE_PATH)
    # ================ Progress Function ================
    train(**config, progress_fn=progress)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

if __name__ == "__main__":
    main()