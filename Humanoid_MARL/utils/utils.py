from typing import Dict

import random
import numpy as np
import os
import torch
import yaml
from Humanoid_MARL import CONFIG_TRAIN, CONFIG_NETWORK, CONFIG_REWARD


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_reward_config(path: str, env) -> Dict:
    if env == "humanoid":
        path = os.path.join(CONFIG_REWARD, "reward_humanoid.yaml")
    elif env == "humanoids":
        path = os.path.join(CONFIG_REWARD, "reward_humanoids.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
