import os

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_ROOT)

CONFIG_TRAIN_HUMANOID = os.path.join(PACKAGE_ROOT, "config/training_humanoid.yaml")
CONFIG_TRAIN_ANT = os.path.join(PACKAGE_ROOT, "config/training_ant.yaml")
CONFIG_TRAIN_POINT_MASS = os.path.join(PACKAGE_ROOT, "config/training_point_mass.yaml")
CONFIG_TRAIN_LINKED_BALLS = os.path.join(
    PACKAGE_ROOT, "config/training_simple_robot.yaml"
)
CONFIG_TRAIN_SIMPLE_ROBOT = os.path.join(
    PACKAGE_ROOT, "config/training_simple_robot.yaml"
)

CONFIG_NETWORK_HUMANOID = os.path.join(PACKAGE_ROOT, "config/network_humanoid.yaml")
CONFIG_NETWORK_ANT = os.path.join(PACKAGE_ROOT, "config/network_ant.yaml")
CONFIG_NETWORK_POINT_MASS = os.path.join(PACKAGE_ROOT, "config/network_point_mass.yaml")
CONFIG_NETWORK_LINKED_BALLS = os.path.join(
    PACKAGE_ROOT, "config/network_simple_robot.yaml"
)

CONFIG_NETWORK_SIMPLE_ROBOT = os.path.join(
    PACKAGE_ROOT, "config/network_simple_robot.yaml"
)

CONFIG_AGENT_HUMANOID = os.path.join(PACKAGE_ROOT, "config/agent_config_humanoid.yaml")
CONFIG_AGENT_ANT = os.path.join(PACKAGE_ROOT, "config/agent_config_ant.yaml")
CONFIG_AGENT_POINT_MASS = os.path.join(
    PACKAGE_ROOT, "config/agent_config_point_mass.yaml"
)
CONFIG_AGENT_LINKED_BALLS = os.path.join(
    PACKAGE_ROOT, "config/agent_config_simple_robot.yaml"
)
CONFIG_AGENT_SIMPLE_ROBOT = os.path.join(
    PACKAGE_ROOT, "config/agent_config_simple_robot.yaml"
)

CONFIG_REWARD_HUMANOID = os.path.join(PACKAGE_ROOT, "config/reward_humanoids.yaml")
CONFIG_REWARD_ANT = os.path.join(PACKAGE_ROOT, "config/reward_ant.yaml")
CONFIG_REWARD_POINT_MASS = os.path.join(PACKAGE_ROOT, "config/reward_point_mass.yaml")
CONFIG_REWARD_LINKED_BALLS = os.path.join(
    PACKAGE_ROOT, "config/reward_simple_robot.yaml"
)
CONFIG_REWARD_SIMPLE_ROBOT = os.path.join(
    PACKAGE_ROOT, "config/reward_simple_robot.yaml"
)
