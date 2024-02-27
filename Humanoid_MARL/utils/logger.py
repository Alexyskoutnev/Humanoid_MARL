from typing import Dict, List, ClassVar
import wandb
import torch


class WandbLogger:
    """Wandb logger for logging metrics and visualizing training progress."""

    _skipped_keys: ClassVar[List[str]] = [
        "first_pipeline_state",
        "truncation",
        "first_obs",
    ]

    def __init__(self, project_name="", config: Dict = {}, notes: str = "") -> None:
        wandb.init(project=project_name, config=config, notes=notes)

    def log_train(
        self, info: Dict[str, torch.Tensor], rewards: List[float], num_agents: int
    ) -> None:
        log_data = {}
        for i in range(num_agents):
            agent_prefix = f"_h{i+1}" if num_agents > 1 else ""
            for key, _ in info.items():
                if key in WandbLogger._skipped_keys:
                    continue
                if key == "steps":
                    log_data.update({key: info[key]})
                else:
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
        self,
        episode_reward: float,
        sps: float,
        eval_sps: float,
        total_loss: float,
        running_mean_reward: float,
    ) -> None:
        wandb.log(
            {
                "eval/episode_reward": episode_reward,
                "speed/sps": sps,
                "speed/eval_sps": eval_sps,
                "losses/total_loss": total_loss,
                "eval/running_mean": running_mean_reward,
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

    def log_running_mean(self, running_mean: float) -> None:
        wandb.log({"eval/running_mean": running_mean})
