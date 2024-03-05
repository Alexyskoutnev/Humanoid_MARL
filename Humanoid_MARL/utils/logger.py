from typing import Dict, List, ClassVar
import wandb
import torch


class WandbLogger:
    """Wandb logger for logging metrics and visualizing training progress."""

    _skipped_keys: ClassVar[List[str]] = [
        "first_pipeline_state",
        "truncation",
        "first_obs",
        "steps",
    ]
    _agent_name_map: ClassVar[Dict[int, str]] = {0: "persuader", 1: "evader"}

    def __init__(self, project_name="", config: Dict = {}, notes: str = "") -> None:
        wandb.init(project=project_name, config=config, notes=notes)

    def log_train(
        self,
        info: Dict[str, torch.Tensor],
        rewards: torch.Tensor,
        num_agents: int,
    ) -> None:
        if num_agents == 1:
            log_data = {}
            log_data["total_reward"] = rewards[0].cpu().item()
            for key, value in info.items():
                if key in WandbLogger._skipped_keys:
                    continue
                try:
                    log_data[key] = torch.mean(value, dim=0).cpu().item()
                except Exception as e:
                    print(f"Error logging {key}: {e}")
            wandb.log(log_data)
        if num_agents == 2:
            log_data = {}
            log_data["total_reward_evader"] = rewards[0][1].cpu().item()
            log_data["total_reward_persuader"] = rewards[0][0].cpu().item()
            for i in range(num_agents):
                agent_prefix = f"_{self._agent_name_map[i]}" if num_agents > 1 else ""
                for key, _ in info.items():
                    if key in WandbLogger._skipped_keys:
                        continue
                    try:
                        log_data[f"{key}{agent_prefix}"] = (
                            torch.mean(info[key], dim=0).cpu()[i].item()
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
