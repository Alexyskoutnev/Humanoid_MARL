import os
from datetime import datetime
import numpy as np

#Visuals
import mediapy as media
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display
import webbrowser

from Humanoid_MARL.agent.ppo.train_torch import train


def main():
    # ================ Config ================
    num_timesteps=50_000_000
    num_evals=10
    episode_length=1000
    normalize_observations=True
    action_repeat=1
    unroll_length=10
    num_minibatches=32
    num_updates_per_batch=8
    discounting=0.97
    learning_rate=3e-4
    entropy_cost=1e-3
    num_envs=2048
    batch_size=1024
    env_name = "humanoids"
    # ================ Config ================

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
        plt.xlim([0, num_timesteps])
        plt.ylim([0, 6000])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        PLT_SAVE_PATH = os.path.join(path, name)
        plt.savefig(PLT_SAVE_PATH)
    # ================ Progress Function ================
    train(env_name=env_name,
          num_envs=num_envs,
          episode_length=episode_length,
          num_timesteps=num_timesteps,
          unroll_length=unroll_length,
          batch_size=batch_size,
          num_minibatches=num_minibatches,
          num_update_epochs=num_updates_per_batch,
          discounting=discounting,
          learning_rate=learning_rate,
          entropy_cost=entropy_cost,
          progress_fn=progress
        )
    
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

if __name__ == "__main__":
    main()



