from typing import Dict, List
import wandb
import torch


class WandbLogger:
    """Wandb logger for logging metrics and visualizing training progress."""

    def __init__(self, project_name="", config: Dict = {}) -> None:
        wandb.init(project=project_name, config=config)

    def log_train(
        self, info: Dict[str, torch.Tensor], rewards: List[float], num_agents: int
    ) -> None:
        log_data = {}
        for i in range(num_agents):
            agent_prefix = f"_h{i+1}" if num_agents > 1 else ""
            for key, _ in info.items():
                try:
                    log_data.update(
                        {
                            f"{key}{agent_prefix}": torch.mean(info[key], dim=0)
                            .cpu()[i]
                            .item()
                        }
                    )
                except Exception as e:
                    print(f"Error logging {key}{agent_prefix}: {e}")
        wandb.log(log_data)

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
