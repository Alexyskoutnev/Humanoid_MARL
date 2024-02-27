import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_ROOT)
CONFIG_TRAIN = os.path.join(PACKAGE_ROOT, "config/training.yaml")
CONFIG_NETWORK = os.path.join(PACKAGE_ROOT, "config/network.yaml")
CONFIG_REWARD = os.path.join(PACKAGE_ROOT, "config")
CONFIG_AGENT = os.path.join(PACKAGE_ROOT, "config/agent_config.yaml")
