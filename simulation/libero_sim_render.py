import logging
import os
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import hydra
from tqdm import tqdm

import wandb

from .base_sim import BaseSim
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

log = logging.getLogger(__name__)


def _find_onscreen_env_cls():
    """
    Try to find an onscreen render env class from LIBERO.
    Different versions may expose different names.
    """
    # Common names
    cand_names = ("RenderEnv", "OnScreenRenderEnv", "ViewerEnv")
    try:
        import libero.libero.envs as e
        for n in cand_names:
            if hasattr(e, n):
                return getattr(e, n)
    except Exception:
        pass
    return None


def _img_to_tensor(img_hwc_uint8: np.ndarray, device: str) -> torch.Tensor:
    # (H,W,C) uint8 -> (1,1,C,H,W) float [0,1]
    return (
        torch.from_numpy(img_hwc_uint8)
        .to(device)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        / 255.0
    )


class MultiTaskSimRender(BaseSim):
    """
    Viewer-first simulation class.
    - render=True: uses onscreen env (must exist in your LIBERO install)
    - render=False: uses OffScreenRenderEnv
    """

    def __init__(
        self,
        num_episode: int,
        max_step_per_episode: int,
        task_suite: str,
        use_eye_in_hand: bool,
        seed: int,
        device: str,
        render: bool,
        n_cores: int,
        use_multiprocessing: bool = True,
    ):
        super().__init__(seed, device, render, n_cores)

        self.task_suite = task_suite
        self.use_eye_in_hand = bool(use_eye_in_hand)
        self.render = bool(render)

        self.num_episode = int(num_episode)
        self.max_step_per_episode = int(max_step_per_episode)

        # 开窗渲染：强制单进程（更稳定）
        self.use_multiprocessing = bool(use_multiprocessing)
        if self.render and self.use_multiprocessing:
            log.warning("render=True: force use_multiprocessing=False for onscreen viewer.")
            self.use_multiprocessing = False
            self.n_cores = 1

        # task embeddings: dict[file_name] -> Tensor
        self.task_embs: Dict[str, torch.Tensor] = {}

        # pick env class
        self._OnScreenEnv = _find_onscreen_env_cls()
        if self.render and self._OnScreenEnv is None:
            try:
                import libero.libero.envs as e
                names = [n for n in dir(e) if "Env" in n or "Render" in n]
            except Exception:
                names = []
            raise ImportError(
                "render=True but cannot find onscreen env class in libero.libero.envs.\n"
                "Tried: RenderEnv / OnScreenRenderEnv / ViewerEnv\n"
                f"Available symbols sample: {names[:60]}\n"
                "Fix: check your LIBERO version and update _find_onscreen_env_cls()."
            )

    def get_task_embs(self, task_embs: Dict[str, torch.Tensor]):
        self.task_embs = task_embs

    def _make_env(self, env_args: Dict[str, Any]):
        if self.render:
            return self._OnScreenEnv(**env_args)
        return OffScreenRenderEnv(**env_args)

    @torch.no_grad()
    def _rollout_one(self, agent, context: int, epi_idx: int):
        task_suite = benchmark.get_benchmark_dict()[self.task_suite]()
        task_bddl_file = task_suite.get_task_bddl_file_path(int(context))
        file_name = os.path.basename(task_bddl_file).split(".")[0]

        if file_name not in self.task_embs:
            raise KeyError(
                f"Missing task embedding for key: {file_name}\n"
                f"Available keys sample: {list(self.task_embs.keys())[:5]}"
            )

        task_emb = self.task_embs[file_name].to(self.device).unsqueeze(0)  # (1,D)

        init_states = task_suite.get_task_init_states(int(context))
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        env = self._make_env(env_args)

        try:
            agent.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[int(epi_idx)])

            # warmup
            dummy = np.zeros(7, dtype=np.float32)
            dummy[-1] = -1.0
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)
                if self.render:
                    try:
                        env.render()
                        time.sleep(0.002)
                    except Exception:
                        pass

            success = 0.0
            ep_len = self.max_step_per_episode

            for t in range(self.max_step_per_episode):
                agentview_rgb = _img_to_tensor(obs["agentview_image"], self.device)
                eye_in_hand_rgb = _img_to_tensor(obs["robot0_eye_in_hand_image"], self.device)

                joint_state = obs["robot0_joint_pos"]
                gripper_state = obs["robot0_gripper_qpos"]
                robot_states = (
                    torch.from_numpy(np.concatenate([joint_state, gripper_state], axis=-1))
                    .to(self.device)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                obs_dict = {
                    "agentview_image": agentview_rgb,
                    "eye_in_hand_image": eye_in_hand_rgb,
                    "lang_emb": task_emb,  # 和你原始代码保持一致：shape(1,D)
                    "robot_states": robot_states,
                }

                action = agent.predict(obs_dict).cpu().numpy()
                obs, r, done, _ = env.step(action)

                if self.render:
                    try:
                        env.render()
                        time.sleep(0.002)
                    except Exception:
                        pass

                if r == 1:
                    success = 1.0
                    ep_len = t + 1
                    break

            return success, float(ep_len)

        finally:
            try:
                env.close()
            except Exception:
                pass

    def test_agent(self, agent, agent_config, cpu_set=None, epoch=None):
        logging.info("Start testing agent (viewer mode)")

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode])
        episode_lengths = torch.zeros([num_tasks, self.num_episode])

        all_runs = num_tasks * self.num_episode
        pbar = tqdm(total=all_runs, desc="Testing agent (viewer)")

        for context in range(num_tasks):
            for epi in range(self.num_episode):
                s, L = self._rollout_one(agent, context, epi)
                success[context, epi] = s
                episode_lengths[context, epi] = L
                pbar.update(1)

        pbar.close()

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()
        log.info("Average success rate: %s", average_success)

        # wandb：没 init 就别 log
        if getattr(wandb, "run", None) is not None:
            for num in range(num_tasks):
                wandb.log({f"{epoch}_task_{num}_success": success_rate[num].item()})
            wandb.log({f"epoch{epoch}_average_success": average_success})

        return average_success