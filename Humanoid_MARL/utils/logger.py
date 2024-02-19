from typing import Dict, List
import wandb
import torch


class WandbLogger:
    def __init__(self, project_name="", config: Dict = {}) -> None:
        wandb.init(project=project_name, config=config)

    def log_train(self, info: Dict, rewards: List[float], num_agents: int) -> None:
        if num_agents == 1:
            wandb.log(
                {
                    "forward_reward": torch.mean(info["forward_reward"], dim=0)
                    .cpu()
                    .item(),
                    "reward_linvel": torch.mean(info["reward_linvel"], dim=0)
                    .cpu()
                    .item(),
                    "reward_quadctrl": torch.mean(info["reward_quadctrl"], dim=0)
                    .cpu()
                    .item(),
                    "reward_alive": torch.mean(info["reward_alive"], dim=0)
                    .cpu()
                    .item(),
                    "x_position": torch.mean(info["x_position"], dim=0).cpu().item(),
                    "y_position": torch.mean(info["y_position"], dim=0).cpu().item(),
                    "distance_from_origin": torch.mean(
                        info["distance_from_origin"], dim=0
                    )
                    .cpu()
                    .item(),
                    "training_reward": rewards.cpu().item(),
                    "x_velocity": torch.mean(info["x_velocity"], dim=0).cpu().item(),
                    "y_velocity": torch.mean(info["y_velocity"], dim=0).cpu().item(),
                }
            )
        elif num_agents == 2:
            wandb.log(
                {
                    "forward_reward_h1": torch.mean(info["forward_reward"], dim=0)
                    .cpu()[0]
                    .item(),
                    "forward_reward_h2": torch.mean(info["forward_reward"], dim=0)
                    .cpu()[1]
                    .item(),
                    "reward_linvel_h1": torch.mean(info["reward_linvel"], dim=0)
                    .cpu()[0]
                    .item(),
                    "reward_linvel_h2": torch.mean(info["reward_linvel"], dim=0)
                    .cpu()[1]
                    .item(),
                    "reward_quadctrl_h1": torch.mean(info["reward_quadctrl"], dim=0)
                    .cpu()[0]
                    .item(),
                    "reward_quadctrl_h2": torch.mean(info["reward_quadctrl"], dim=0)
                    .cpu()[1]
                    .item(),
                    "reward_alive_h1": torch.mean(info["reward_alive"], dim=0)
                    .cpu()[0]
                    .item(),
                    "reward_alive_h2": torch.mean(info["reward_alive"], dim=0)
                    .cpu()[1]
                    .item(),
                    "x_position_h1": torch.mean(info["x_position"], dim=0)
                    .cpu()[0]
                    .item(),
                    "x_position_h2": torch.mean(info["x_position"], dim=0)
                    .cpu()[1]
                    .item(),
                    "y_position_h1": torch.mean(info["y_position"], dim=0)
                    .cpu()[0]
                    .item(),
                    "y_position_h2": torch.mean(info["y_position"], dim=0)
                    .cpu()[1]
                    .item(),
                    "distance_from_origin_h1": torch.mean(
                        info["distance_from_origin"], dim=0
                    )
                    .cpu()[0]
                    .item(),
                    "distance_from_origin_h2": torch.mean(
                        info["distance_from_origin"], dim=0
                    )
                    .cpu()[1]
                    .item(),
                    "training_reward_h1": rewards[0].cpu().item(),
                    "training_reward_h2": rewards[1].cpu().item(),
                    "x_velocity_h1": torch.mean(info["x_velocity"], dim=0)
                    .cpu()[0]
                    .item(),
                    "x_velocity_h2": torch.mean(info["x_velocity"], dim=0)
                    .cpu()[1]
                    .item(),
                    "y_velocity_h1": torch.mean(info["y_velocity"], dim=0)
                    .cpu()[0]
                    .item(),
                    "y_velocity_h2": torch.mean(info["y_velocity"], dim=0)
                    .cpu()[1]
                    .item(),
                    "z_position_h1": torch.mean(info["z_position"], dim=0)
                    .cpu()[0]
                    .item(),
                    "z_position_h2": torch.mean(info["z_position"], dim=0)
                    .cpu()[1]
                    .item(),
                }
            )

    def log_eval(
        self, episode_reward: float, sps: float, eval_sps: float, total_loss: float
    ) -> None:
        wandb.log(
            {
                "eval/episode_reward": episode_reward,
                "speed/sps": sps,
                "speed/eval_sps": eval_sps,
                "losses/total_loss": total_loss,
            }
        )

    def log_epoch_loss(self, epoch_loss: float) -> None:
        wandb.log({"losses/epoch_loss": epoch_loss})

    def log_network_loss(
        self, critic_loss: float, actor_loss: float, entropy_loss: float
    ) -> None:
        wandb.log(
            {
                "losses/critic_loss": critic_loss,
                "losses/actor_loss": actor_loss,
                "losses/entropy_loss": entropy_loss,
            }
        )
