import logging
import random
import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class IdentityScaler:
    """Fallback scaler when model_scaler.pkl is missing."""
    def inverse_scale_output(self, x):
        return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra.main(
    config_path="configs",
    config_name="libero_config.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):

    # -----------------------------------------------------------------------
    # 0. Basic setup
    # -----------------------------------------------------------------------
    set_seed_everywhere(cfg.seed)
    log.info("========== Evaluation-only mode ==========")

    device = cfg.device if torch.cuda.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # 1. Instantiate agent (NO trainer, NO dataset)
    # -----------------------------------------------------------------------
    log.info("Instantiating agent...")
    agent = hydra.utils.instantiate(cfg.agents)
    agent.to(device)
    agent.eval()

    # -----------------------------------------------------------------------
    # 2. Load pretrained model weights
    # -----------------------------------------------------------------------
    # !!! CHANGE THIS PATH !!!
    # Should point to the directory that contains model_state_dict.pth
    ckpt_dir = sim_framework_path(cfg.eval.ckpt_dir)

    log.info(f"Loading model from: {ckpt_dir}")
    agent.load_pretrained_model(ckpt_dir)

    # -----------------------------------------------------------------------
    # 3. Load scaler (or fallback to identity)
    # -----------------------------------------------------------------------
    try:
        agent.load_model_scaler(ckpt_dir)
        log.info("Loaded model scaler.")
    except FileNotFoundError:
        log.warning("model_scaler.pkl not found, using IdentityScaler.")
        agent.set_scaler(IdentityScaler())

    # -----------------------------------------------------------------------
    # 4. Instantiate simulation environment
    # -----------------------------------------------------------------------
    log.info("Instantiating simulation environment...")
    env_sim = hydra.utils.instantiate(cfg.simulation)

    # -----------------------------------------------------------------------
    # 5. Load task embeddings (required by MultiTaskSim)
    # -----------------------------------------------------------------------
    log.info("Loading task embeddings...")

    # Task embeddings are stored in: task_embeddings/{task_suite}.pkl
    # Reuse LiberoDataset just to load the task dict
    try:
        from environments.dataset.libero_dataset import LiberoDataset

        dummy_dataset = LiberoDataset(
            data_directory=cfg.dataset_path,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            max_len_data=cfg.max_len_data,
            window_size=cfg.window_size,
            traj_per_task=1,
        )
        env_sim.get_task_embs(dummy_dataset.tasks)
        log.info("Task embeddings loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load task embeddings: {e}")
        raise RuntimeError(
            "Task embeddings are required for evaluation. "
            "Please check task_embeddings/*.pkl"
        )

    # -----------------------------------------------------------------------
    # 6. Run evaluation
    # -----------------------------------------------------------------------
    log.info("Starting simulation evaluation...")
    env_sim.test_agent(
        agent=agent,
        agent_config=None,
        epoch="eval"
    )

    log.info("========== Evaluation finished ==========")


if __name__ == "__main__":
    main()
