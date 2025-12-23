import logging
import os
import sys
import importlib
import importlib.util
import pkgutil

import cv2
import einops
import numpy as np
import robosuite.utils.transform_utils as T
import torch
import wandb
from omegaconf import ListConfig
from tqdm import tqdm

from agents.base_agent import BaseAgent
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)

def _prefer_local_robocasa():
    """强制使用仓库内的 robocasa 包，绕过 NamespaceLoader 命名空间包。"""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # /home/public/wyl/X_IL
    local_pkg_dir = os.path.join(repo_root, "robocasa", "robocasa")              # /home/public/wyl/X_IL/robocasa/robocasa
    init_py = os.path.join(local_pkg_dir, "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError(f"本地 robocasa 包缺少 __init__.py: {init_py}")

    # 清理已加载的 robocasa*，避免命名空间包混入
    for k in list(sys.modules.keys()):
        if k == "robocasa" or k.startswith("robocasa."):
            del sys.modules[k]

    # 以“真正包”的方式强制载入本地 robocasa
    parent_dir = os.path.dirname(local_pkg_dir)  # /home/public/wyl/X_IL/robocasa
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    spec = importlib.util.spec_from_file_location(
        "robocasa", init_py, submodule_search_locations=[local_pkg_dir]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["robocasa"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    log.info(f"robocasa loaded from: {module.__file__}")

def _patch_robosuite_register_env():
    """robosuite 某些安装版本 register_env 为空，这里补上。"""
    try:
        from robosuite.environments import base as rs_base
        def _register_env(target_class):
            name = getattr(target_class, "ENVIRONMENT_NAME", None) \
                   or getattr(target_class, "name", None) \
                   or target_class.__name__
            rs_base.REGISTERED_ENVS[name] = target_class
        rs_base.register_env = _register_env
        return True
    except Exception as e:
        log.warning(f"patch register_env 失败: {e}")
        return False

def _force_register_robocasa_envs(expected_env: str | None = None):
    """
    尝试官方注册；若失败，直接把 REGISTERED_KITCHEN_ENVS 写入 robosuite 的 REGISTERED_ENVS。
    """
    from robosuite.environments import base as rs_base

    if expected_env and expected_env in rs_base.REGISTERED_ENVS:
        return

    _patch_robosuite_register_env()
    _prefer_local_robocasa()

    # 1) 官方注册（若可用）
    try:
        reg = importlib.import_module("robocasa.environments.registration")
        if hasattr(reg, "register_environments"):
            reg.register_environments()
    except Exception as e:
        log.warning(f"register_environments 调用失败: {e}")

    # 2) 兜底：用 REGISTERED_KITCHEN_ENVS 直接写入
    if expected_env and expected_env not in rs_base.REGISTERED_ENVS:
        try:
            rc_envs = importlib.import_module("robocasa.environments")
            reg_map = getattr(rc_envs, "REGISTERED_KITCHEN_ENVS", None)
            if reg_map is None:
                kk = importlib.import_module("robocasa.environments.kitchen.kitchen")
                reg_map = getattr(kk, "REGISTERED_KITCHEN_ENVS", None)
            if not reg_map:
                raise RuntimeError("未找到 REGISTERED_KITCHEN_ENVS")

            for name, cls in reg_map.items():
                rs_base.REGISTERED_ENVS[name] = cls

            log.info(f"RoboCasa 手动注册完成，共 {len(reg_map)} 个环境")
        except Exception as e:
            log.warning(f"手动注册 RoboCasa 环境失败: {e}")

    # 3) 最终校验
    if expected_env and expected_env not in rs_base.REGISTERED_ENVS:
        avail = ", ".join(sorted(rs_base.REGISTERED_ENVS.keys()))
        raise RuntimeError(f"RoboCasa 环境未注册: 期望 {expected_env}，已注册: {avail}")

class RoboCasaSim(BaseSim):
    def __init__(self, *args, **kwargs):
        # 吞掉 Hydra 传入但 BaseSim 不认识的字段
        self.if_vision = kwargs.pop("if_vision", True)

        self.num_episode = int(kwargs.pop("num_episode", 1))
        self.max_step_per_episode = int(kwargs.pop("max_step_per_episode", 200))

        cam_names = kwargs.pop("camera_names", [])
        if isinstance(cam_names, ListConfig):
            cam_names = list(cam_names)
        self.camera_names = list(cam_names)

        self.global_action = bool(kwargs.pop("global_action", False))

        # 仿真/渲染配置
        self.env_name = kwargs.pop("env_name", None)
        self.img_height = int(kwargs.pop("img_height", 256))
        self.img_width = int(kwargs.pop("img_width", 256))
        self.render = bool(kwargs.pop("render", False))

        self.task_embs = None  # 兼容未设置时的访问

        # 其余参数交给 BaseSim 处理
        super().__init__(*args, **kwargs)

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    # 提供与调用处一致的别名，内部复用已实现的方法
    def get_local_action(self, global_action: np.ndarray) -> np.ndarray:
        return self._get_local_action(global_action)

    # 对齐 libero 接口，兼容可选参数
    def test_agent(self, agent: BaseAgent, agent_config=None, cpu_set=None, epoch=None, **kwargs):
        name_map = {
            "PnPCabToCounter": "PnPCounterToCab",
            "PnPStoveToCounter": "PnPCounterToStove",
            "PnPSinkToCounter": "PnPCounterToSink",
        }
        # 规范化 env 列表，兼容 str / 逗号分隔字符串 / ListConfig / list/tuple/set
        raw_envs = self.env_name
        if isinstance(raw_envs, str):
            env_list = [e.strip() for e in raw_envs.split(",")] if "," in raw_envs else [raw_envs]
        elif isinstance(raw_envs, (ListConfig, list, tuple, set)):
            env_list = list(raw_envs)
        else:
            env_list = [str(raw_envs)]

        for env in env_list:
            env = name_map.get(str(env), str(env))  # 旧名映射到新名
            print(f"Initializing environment {env}")
            self._init_env(
                env,
                self.img_width,
                self.img_height,
                self.render,
            )

            success_count = 0
            task_completion_hold_count = -1

            for i in range(self.num_episode):
                obs = self.env.reset()
                if self.render:
                    self.env.render()

                agent.reset()

                lang = self.env.get_ep_meta()["lang"]

                print(f"episode {i}: ", lang)

                for j in tqdm(range(self.max_step_per_episode)):
                    obs_dict = {}
                    obs_dict["lang"] = lang
                    gripper_state = torch.from_numpy(obs["robot0_gripper_qpos"]).float()
                    gripper_state = einops.rearrange(gripper_state, "d -> 1 1 d").to(
                        self.device
                    )

                    joint_pos = torch.from_numpy(obs["robot0_joint_pos_cos"]).float()
                    joint_pos = einops.rearrange(joint_pos, "d -> 1 1 d").to(self.device)
                    robot_state = torch.cat([gripper_state, joint_pos], dim=-1)
                    obs_dict["robot_states"] = robot_state
                    for cam_name in self.camera_names:

                        # cv2.imshow(f"{cam_name}_image", obs[f"{cam_name}_image"])
                        # cv2.waitKey(1)
                        rgb = (
                            torch.from_numpy(obs[f"{cam_name}_image"].copy())
                            .float()
                            .permute(2, 0, 1)
                            / 255.0
                        )
                        rgb = einops.rearrange(rgb, "c h w -> 1 1 c h w").to(self.device)
                        obs_dict[f"{cam_name}_image"] = rgb
                    action = agent.predict(obs_dict).cpu().numpy()
                    action = np.concatenate(
                        [action, np.array([0, 0, 0, 0, -1])]
                    )

                    # action = np.concatenate(
                    #     [action[1:], action[:1], np.array([0, 0, 0, 0, -1])]
                    # )
                    if self.global_action:
                        action = self.get_local_action(action)
                    obs, _, done, _ = self.env.step(action)

                    if self.render:
                        self.env.render()

                    if self.env._check_success():
                        if task_completion_hold_count > 0:
                            task_completion_hold_count -= (
                                1  # latched state, decrement count
                            )
                        else:
                            task_completion_hold_count = (
                                10  # reset count on first success timestep
                            )
                    else:
                        task_completion_hold_count = (
                            -1
                        )  # null the counter if there's no success

                    if task_completion_hold_count == 0:
                        success_count += 1
                        done = True

                    if done:
                        break
            success_rate = success_count / self.num_episode
            print(f"Success rate: {success_rate}")

            wandb.log({f"{env}_average_success": success_rate})
            self.env.close()

    def _get_local_action(self, global_action: np.ndarray) -> np.ndarray:
        base_mat = self.env.sim.data.get_site_xmat(
            f"mobilebase{self.env.robots[0].idn}_center"
        )

        global_action_pos = global_action[:3]
        global_action_axis_angle = global_action[3:6]
        global_action_mat = T.quat2mat(T.axisangle2quat(global_action_axis_angle))

        local_action_pos = base_mat.T @ global_action_pos
        local_action_mat = base_mat.T @ global_action_mat @ base_mat
        local_action_axis_angle = T.quat2axisangle(T.mat2quat(local_action_mat))
        local_action = np.concatenate(
            [local_action_pos, local_action_axis_angle, global_action[6:]]
        )
        return local_action

    def _init_env(
        self,
        env_name,
        img_width,
        img_height,
        render,
    ):
        # 先确保注册，再创建
        _force_register_robocasa_envs(expected_env=env_name)

        # 延迟导入，确保走本地 robocasa
        _prefer_local_robocasa()
        from robocasa.utils.env_utils import create_env

        base_env = create_env(
            env_name=env_name,
            camera_names=self.camera_names,
            camera_widths=img_width,
            camera_heights=img_height,
            seed=self.seed,
            render_onscreen=bool(render),
        )
        self.env = RobosuiteWrapper(base_env)
