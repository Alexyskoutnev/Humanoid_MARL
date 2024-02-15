import os
from datetime import datetime
import numpy as np
import wandb
import matplotlib.pyplot as plt

from Humanoid_MARL.agent.ppo.train_torch import train
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import seed_everything

SEED = 1
seed_everything(SEED)

def main():
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"USING GPU {gpu_index}")
    env_name = "humanoids"
    project_name = f"MARL_ppo_{env_name}"
    debug = False
    if not debug:
        config = {
            'num_timesteps': 500_000_000,
            'eval_reward_limit' : 10_000,
            'eval_frequency': 50,
            'episode_length': 1000,
            'unroll_length': 10,
            'num_minibatches': 32,
            'num_update_epochs': 8,
            'discounting': 0.97,
            'learning_rate': 3e-4,
            'entropy_cost': 2e-3,
            'num_envs': 2048,
            'batch_size': 512,
            'env_name': env_name,
            'device' : 'cuda',
            'debug' : False,
            'device_idx' : gpu_index,
            'notebook' : False,
            'model_path' : "./models/20240214_184013_ppo_humanoids.pt",
        }
    else:
        config = {
            'num_timesteps': 200_000,
            'eval_reward_limit' : 10_000,
            'eval_frequency': 10,
            'episode_length': 1000,
            'unroll_length': 1,
            'num_minibatches': 8,
            'num_update_epochs': 1,
            'discounting': 0.97,
            'learning_rate': 3e-4,
            'entropy_cost': 1e-3,
            'num_envs': 4,
            'batch_size': 64,
            'env_name': env_name,
            'device' : 'cuda',
            'debug' : True,
            'device_idx' : gpu_index,
            'notebook' : False,
            'model_path' : None,
        }
    # ================ Config ================
    # ================ Logging ===============
    if not config['debug']:
        logger = WandbLogger(project_name, config=config)
        config['logger'] = logger
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
        plt.ylim([0, 10_000])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_name = f"{timestamp}_{name}"
        PLT_SAVE_PATH = os.path.join(path, timestamped_name)
        plt.savefig(PLT_SAVE_PATH)
    # ================ Progress Function ================
    if debug:
        train(**config, progress_fn=None)
    else:
        train(**config, progress_fn=progress)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    print(f"eval steps/sec: {np.mean(eval_sps)}")
    print(f"train steps/sec: {np.mean(train_sps)}")

if __name__ == "__main__":
    main()