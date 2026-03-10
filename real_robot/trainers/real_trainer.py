import logging
import os

import torch
import wandb
from tqdm import tqdm

from agents.utils.ema import ExponentialMovingAverage
from trainers.base_trainer import BaseTrainer


log = logging.getLogger(__name__)


class RealTrainer(BaseTrainer):
    def __init__(self, *args, checkpoint_every_n_epochs: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_every_n_epochs = int(checkpoint_every_n_epochs)

    def main(self, agent):
        agent.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(agent.parameters(), self.decay_ema, self.device)

        if agent.use_lr_scheduler:
            self.optimizer, self.scheduler = agent.configure_optimizers()
        else:
            self.optimizer = agent.configure_optimizers()

        for num_epoch in tqdm(range(self.epoch)):
            epoch_loss = torch.tensor(0.0).to(self.device)

            for data in self.train_dataloader:
                obs_dict, action, mask = data

                for key in obs_dict.keys():
                    if key == "lang":
                        continue

                    obs_dict[key] = obs_dict[key].to(self.device)

                    if "rgb" not in key and "image" not in key:
                        continue
                    obs_dict[key] = obs_dict[key][:, :self.obs_seq_len].contiguous()

                action = self.scaler.scale_output(action)
                action = action[:, self.obs_seq_len - 1 :, :].contiguous()

                batch_loss = self.train_one_step(agent, obs_dict, action)
                epoch_loss += batch_loss

            epoch_loss = epoch_loss / len(self.train_dataloader)
            wandb.log({"train_loss": epoch_loss.item(), "epoch": num_epoch})
            log.info("Epoch %s: Mean train loss is %s", num_epoch, epoch_loss.item())

            if self.checkpoint_every_n_epochs > 0 and (num_epoch + 1) % self.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(agent, num_epoch + 1)

        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(agent.parameters())
            self.ema_helper.copy_to(agent.parameters())

        agent.store_model_weights(agent.working_dir, sv_name="last_model")
        agent.store_model_scaler(agent.working_dir)

    def _save_checkpoint(self, agent, epoch: int) -> None:
        checkpoint_dir = os.path.join(agent.working_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if hasattr(self, "scheduler"):
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.if_use_ema:
            payload["ema_state_dict"] = self.ema_helper.state_dict()

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(payload, checkpoint_path)
        log.info("Checkpoint saved to: %s", checkpoint_path)
