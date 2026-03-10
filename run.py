import logging
import random
import os
import hydra
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import torch

from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)

#注册一个新的OmegaConf解析器，用于在配置文件中执行加法操作
OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()

# 设置多线程数
def set_seed_everywhere(seed):
    #设置随机种子
    torch.manual_seed(seed)
    #如果有GPU可用，设置所有GPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #设置numpy的随机种子
    np.random.seed(seed)
    #设置python内置random模块的随机种子
    random.seed(seed)


@hydra.main(config_path="configs", config_name="libero_config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        # mode="disabled",
        config=wandb.config
    )

    # load vqvae before training the agent: add path to the config file
    # train the agent
    agent = hydra.utils.instantiate(cfg.agents)

    # Record runtime-effective replanning interval (may come from python default)
    try:
        wandb.config.update({"runtime/replan_every": int(getattr(agent, "replan_every"))}, allow_val_change=True)
        wandb.log({"runtime/replan_every": int(getattr(agent, "replan_every"))})
    except Exception as e:
        log.warning("Failed to record replan_every to wandb: %s", e)

    if "trainers" not in cfg or cfg.trainers is None:
        raise ValueError(
            "Config must define 'trainers' to run training. "
            "Use --config-name=libero_config.yaml or robocasa_config.yaml, or add a trainers section."
        )

    trainer = hydra.utils.instantiate(cfg.trainers)

    agent.get_params()
    trainer.main(agent)

    if "simulation" not in cfg or cfg.simulation is None:
        raise ValueError(
            "Config must define 'simulation' to run evaluation rollouts. "
            "Use --config-name=libero_config.yaml or robocasa_config.yaml, or add a simulation section."
        )

    env_sim = hydra.utils.instantiate(cfg.simulation)

    tasks = getattr(trainer.trainset, "tasks", None)
    if tasks is not None:
        try:
            env_sim.get_task_embs(tasks)
        except AttributeError:
            print("env_sim has no 'get_task_embs'; skip.")
    else:
        print("trainset has no 'tasks'; skip task embeddings for this env.")

    env_sim.test_agent(agent, cfg.agents, epoch=cfg.epoch)

    log.info("Training done")
    log.info("state_dict saved in {}".format(agent.working_dir))

    # 收尾：结束 wandb 会话（不要在函数内重新 import）
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
