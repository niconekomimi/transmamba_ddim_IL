import logging
import os
import time
import numpy as np
import torch
import hydra
import multiprocessing as mp
from tqdm import tqdm
import wandb

from .base_sim import BaseSim
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.envs import OffScreenRenderEnv

log = logging.getLogger(__name__)


class MultiTaskSimOnscreen(BaseSim):
    """
    LIBERO sim that supports onscreen rendering via ControlEnv(has_renderer=True).
    - render=True  -> ControlEnv (onscreen viewer)
    - render=False -> OffScreenRenderEnv (offscreen)
    """

    def __init__(
        self,
        num_episode,
        max_step_per_episode,
        task_suite: str,
        use_eye_in_hand: bool,
        seed,
        device,
        render,
        n_cores,
        use_multiprocessing=True,
    ):
        super().__init__(seed, device, render, n_cores)

        self.task_suite = task_suite
        self.use_eye_in_hand = use_eye_in_hand
        self.render = bool(render)

        self.num_episode = int(num_episode)
        self.max_step_per_episode = int(max_step_per_episode)

        # viewer 最稳：单进程
        self.use_multiprocessing = bool(use_multiprocessing)
        if self.render and self.use_multiprocessing:
            log.warning("render=True: force use_multiprocessing=False and n_cores=1 for onscreen viewer")
            self.use_multiprocessing = False
            self.n_cores = 1

        self.task_embs = {}

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def _make_env(self, env_args: dict):
        if self.render:
            env_args = dict(env_args)
            # 开窗
            env_args["has_renderer"] = True
            # 同时也要开 offscreen，否则拿不到 camera obs（agentview_image 等）
            env_args["has_offscreen_renderer"] = True
            env_args["use_camera_obs"] = True
            return ControlEnv(**env_args)

        return OffScreenRenderEnv(**env_args)

    def eval_agent(
        self,
        contexts,
        context_ind,
        success,
        episode_lengths,
        pid,
        cpu_set,
        counter,
        agent=None,
        agent_config=None,
        model_states=None,
    ):
        # init agent (multiprocess)
        if agent_config is not None:
            assert model_states is not None, "model_states must be provided when using agent_config"
            agent = hydra.utils.instantiate(agent_config)
            agent.recover_model_state(model_states["model"], model_states["scaler"])
        else:
            assert agent is not None, "Either agent or (agent_config + states) must be provided"

        for i, context in enumerate(contexts):
            task_suite = benchmark.get_benchmark_dict()[self.task_suite]()
            task_bddl_file = task_suite.get_task_bddl_file_path(int(context))
            file_name = os.path.basename(task_bddl_file).split(".")[0]

            if file_name not in self.task_embs:
                raise KeyError(
                    f"task_embs missing key: {file_name}. "
                    f"Available keys sample: {list(self.task_embs.keys())[:5]}"
                )
            task_emb = self.task_embs[file_name].to(self.device).unsqueeze(0)

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
                obs = env.set_init_state(init_state=init_states[int(context_ind[i])])

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

                for j in range(self.max_step_per_episode):
                    agentview_rgb = (
                        torch.from_numpy(obs["agentview_image"])
                        .to(self.device)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        / 255.0
                    )
                    eye_in_hand_rgb = (
                        torch.from_numpy(obs["robot0_eye_in_hand_image"])
                        .to(self.device)
                        .float()
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        / 255.0
                    )

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
                        "lang_emb": task_emb,
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
                        success[int(context), int(context_ind[i])] = r
                        episode_lengths[int(context), int(context_ind[i])] = j + 1
                        break

                if success[int(context), int(context_ind[i])] == 0:
                    episode_lengths[int(context), int(context_ind[i])] = self.max_step_per_episode

            finally:
                try:
                    env.close()
                except Exception:
                    pass

            # progress update
            if hasattr(counter, "get_lock"):
                with counter.get_lock():
                    counter.value += 1
            else:
                counter.value += 1
                counter.update()

    def test_agent(self, agent, agent_config, cpu_set=None, epoch=None):
        logging.info("Start testing agent")

        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)

        if self.task_suite == "libero_90":
            num_tasks = 90
        else:
            num_tasks = 10

        success = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.num_episode]).share_memory_()
        all_runs = num_tasks * self.num_episode

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.num_episode)
        context_ind = np.arange(self.num_episode)
        context_ind = np.tile(context_ind, num_tasks)

        if not self.use_multiprocessing:
            pbar = tqdm(total=all_runs, desc="Testing agent (onscreen)")
            counter = type("Counter", (), {"value": 0})()

            def _upd():
                pbar.update(1)

            counter.update = _upd

            self.eval_agent(
                contexts=contexts,
                context_ind=context_ind,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                agent=agent,
            )
            pbar.close()
        else:
            # 留着兼容，但 render=True 时上面已强制关闭
            repeat_num = all_runs // num_cpu
            repeat_res = all_runs % num_cpu
            workload_array = np.ones([num_cpu], dtype=int)
            workload_array[:repeat_res] += repeat_num
            workload_array[repeat_res:] = repeat_num
            ind_workload = np.cumsum(workload_array)
            ind_workload = np.concatenate([[0], ind_workload])

            ctx = mp.get_context("spawn")
            processes_list = []
            counter = ctx.Value("i", 0)
            pbar = tqdm(total=all_runs, desc="Testing agent")

            model_states = agent.get_model_state
            shared_states = {"model": {}, "scaler": model_states[1]}
            for k, t in model_states[0].items():
                shared_states["model"][k] = t.share_memory_()

            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "contexts": contexts[ind_workload[i] : ind_workload[i + 1]],
                        "context_ind": context_ind[ind_workload[i] : ind_workload[i + 1]],
                        "success": success,
                        "episode_lengths": episode_lengths,
                        "pid": i,
                        "cpu_set": set(cpu_set[i : i + 1]),
                        "counter": counter,
                        "agent": None,
                        "agent_config": agent_config,
                        "model_states": shared_states,
                    },
                )
                p.start()
                processes_list.append(p)

            last = 0
            while any(p.is_alive() for p in processes_list):
                if counter.value > last:
                    pbar.update(counter.value - last)
                    last = counter.value
                time.sleep(0.05)

            for p in processes_list:
                p.join()
            pbar.close()

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()
        log.info("Average success rate: %s", average_success)

        if getattr(wandb, "run", None) is not None:
            wandb.log({f"epoch{epoch}_average_success": average_success})

        return average_success