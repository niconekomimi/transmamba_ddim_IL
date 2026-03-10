import logging
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers), replace=True)
torch.cuda.empty_cache()


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="real_robot_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed_everywhere(cfg.seed)

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode=cfg.wandb.mode,
        config=wandb_config,
    )

    agent = hydra.utils.instantiate(cfg.agents)
    trainer = hydra.utils.instantiate(cfg.trainers)

    agent.get_params()
    trainer.main(agent)

    log.info("Real-robot training done")
    log.info("Artifacts saved in %s", agent.working_dir)

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
