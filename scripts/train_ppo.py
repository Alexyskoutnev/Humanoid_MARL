import os
from datetime import datetime
import numpy as np
import wandb

#Visuals
import mediapy as media
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display
import webbrowser

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
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 2048,
        'batch_size': 512,
        'env_name': "humanoids",
        'render' : True,
        'device' : 'cuda',
    }
    # ================ Config ================
    # ================ Logging ===============
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
        PLT_SAVE_PATH = os.path.join(path, name)
        plt.savefig(PLT_SAVE_PATH)
    # ================ Progress Function ================
    train(**config, progress_fn=progress)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

if __name__ == "__main__":
    main()



