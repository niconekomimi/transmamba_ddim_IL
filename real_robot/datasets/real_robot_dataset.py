import os
import pickle
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeIndex:
    episode_id: int
    start: int


class RealRobotDataset(Dataset):
    """Sliding-window dataset for real robot episodes stored as npz or LIBERO-style HDF5."""

    def __init__(
        self,
        data_directory: str,
        split: str,
        window_size: int,
        action_dim: int,
        state_dim: int,
        camera_names: list[str],
        max_episodes: int | None = None,
        val_ratio: float = 0.1,
        task_description: str | None = None,
        task_name: str | None = None,
        task_embedding_path: str | None = None,
    ):
        self.data_directory = data_directory
        self.split = split
        self.window_size = int(window_size)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.camera_names = list(camera_names)
        self.val_ratio = float(val_ratio)
        self.task_description = task_description
        self.task_name = task_name
        self.task_embedding_path = task_embedding_path
        self.task_embedding_map = self._load_task_embedding_map()

        if os.path.isfile(self.data_directory) and self.data_directory.endswith((".hdf5", ".h5")):
            self.episodes = self._load_hdf5_episodes(self.data_directory, max_episodes)
        elif self._has_hdf5_files(self.data_directory):
            self.episodes = self._load_hdf5_directory(self.data_directory, max_episodes)
        else:
            self.episodes = self._load_npz_episodes(max_episodes)

        self.index = self._build_index()
        self.tasks = [episode.get("task_name", f"episode_{i}") for i, episode in enumerate(self.episodes)]

    def _load_task_embedding_map(self) -> dict | None:
        if not self.task_embedding_path:
            return None

        with open(self.task_embedding_path, "rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _has_hdf5_files(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(name.endswith((".hdf5", ".h5")) for name in os.listdir(path))

    @staticmethod
    def _infer_task_name(source_name: str) -> str:
        filename = os.path.splitext(os.path.basename(source_name))[0]
        if filename.endswith("_demo"):
            filename = filename[:-5]
        return filename

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray, key: str) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            return embedding[None, :]
        if embedding.ndim == 2:
            return embedding
        raise ValueError(f"Task embedding for `{key}` must have shape [D] or [1, D]")

    def _get_task_embedding(self, task_name: str | None) -> np.ndarray | None:
        if self.task_embedding_map is None:
            return None

        lookup_keys = [key for key in (task_name, self.task_name, self.task_description) if key]
        for key in lookup_keys:
            if key in self.task_embedding_map:
                return self._normalize_embedding(self.task_embedding_map[key], key)

        raise KeyError(
            f"None of {lookup_keys} were found in task embedding file {self.task_embedding_path}"
        )

    def _load_npz_episodes(self, max_episodes: int | None) -> list[dict]:
        if not os.path.isdir(self.data_directory):
            raise FileNotFoundError(
                f"Real robot data directory not found: {self.data_directory}. "
                "Expected either a HDF5 file or a directory containing .hdf5/.h5/.npz files."
            )

        episode_files = sorted(
            os.path.join(self.data_directory, name)
            for name in os.listdir(self.data_directory)
            if name.endswith(".npz")
        )
        if not episode_files:
            raise FileNotFoundError(f"No .npz episodes found in {self.data_directory}")

        split_index = max(1, int(round(len(episode_files) * (1.0 - self.val_ratio))))
        if self.split == "train":
            episode_files = episode_files[:split_index]
        elif self.split == "val":
            episode_files = episode_files[split_index:]
        else:
            raise ValueError(f"Unsupported split `{self.split}` for npz dataset")

        if max_episodes is not None:
            episode_files = episode_files[:max_episodes]
        if not episode_files:
            raise ValueError(f"Split `{self.split}` has no npz episodes in {self.data_directory}")
        return [self._load_npz_episode(path) for path in episode_files]

    def _load_hdf5_episodes(self, path: str, max_episodes: int | None) -> list[dict]:
        with h5py.File(path, "r") as dataset_file:
            if "data" not in dataset_file:
                raise KeyError(f"HDF5 file {path} is missing `/data` group")

            demo_keys = sorted(dataset_file["data"].keys())
            if max_episodes is not None:
                demo_keys = demo_keys[:max_episodes]
            if not demo_keys:
                raise ValueError(f"No demos found under /data in {path}")

            split_index = max(1, int(round(len(demo_keys) * (1.0 - self.val_ratio))))
            if self.split == "train":
                demo_keys = demo_keys[:split_index]
            elif self.split == "val":
                demo_keys = demo_keys[split_index:]
            else:
                raise ValueError(f"Unsupported split `{self.split}` for HDF5 dataset")

            if not demo_keys:
                raise ValueError(f"Split `{self.split}` has no demos after applying val_ratio={self.val_ratio}")

            task_name = self.task_name or self._infer_task_name(path)
            return [self._load_hdf5_demo(dataset_file["data"][demo_key], demo_key, task_name) for demo_key in demo_keys]

    def _load_hdf5_directory(self, directory: str, max_episodes: int | None) -> list[dict]:
        hdf5_files = sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.endswith((".hdf5", ".h5"))
        )
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {directory}")

        episodes: list[dict] = []
        for path in hdf5_files:
            task_name = self.task_name or self._infer_task_name(path)
            with h5py.File(path, "r") as dataset_file:
                if "data" not in dataset_file:
                    raise KeyError(f"HDF5 file {path} is missing `/data` group")

                demo_keys = sorted(dataset_file["data"].keys())
                split_index = max(1, int(round(len(demo_keys) * (1.0 - self.val_ratio))))
                if self.split == "train":
                    demo_keys = demo_keys[:split_index]
                elif self.split == "val":
                    demo_keys = demo_keys[split_index:]
                else:
                    raise ValueError(f"Unsupported split `{self.split}` for HDF5 dataset")

                for demo_key in demo_keys:
                    episodes.append(self._load_hdf5_demo(dataset_file["data"][demo_key], f"{task_name}/{demo_key}", task_name))
                    if max_episodes is not None and len(episodes) >= max_episodes:
                        return episodes

        if not episodes:
            raise ValueError(f"Split `{self.split}` has no demos in HDF5 directory {directory}")
        return episodes

    def _load_hdf5_demo(self, demo_group: h5py.Group, demo_key: str, task_name: str) -> dict:
        obs_group = demo_group["obs"]
        episode = {
            "task_name": task_name,
            "actions": np.asarray(demo_group["actions"], dtype=np.float32),
            "agentview_image": np.asarray(obs_group["agentview_rgb"], dtype=np.uint8),
            "eye_in_hand_image": np.asarray(obs_group["eye_in_hand_rgb"], dtype=np.uint8),
        }

        if "robot_states" in demo_group:
            episode["robot_states"] = np.asarray(demo_group["robot_states"], dtype=np.float32)
        elif "states" in demo_group:
            episode["robot_states"] = np.asarray(demo_group["states"], dtype=np.float32)

        if "states" in demo_group:
            episode["states"] = np.asarray(demo_group["states"], dtype=np.float32)

        if "joint_states" in obs_group and "gripper_states" in obs_group:
            joint_states = np.asarray(obs_group["joint_states"], dtype=np.float32)
            gripper_states = np.asarray(obs_group["gripper_states"], dtype=np.float32)
            episode["robot_states"] = np.concatenate([joint_states, gripper_states], axis=-1)

        return self._validate_episode(episode, source=demo_key)

    def _load_npz_episode(self, path: str) -> dict:
        raw = np.load(path, allow_pickle=True)
        episode = {key: raw[key] for key in raw.files}

        return self._validate_episode(episode, source=path)

    def _validate_episode(self, episode: dict, source: str) -> dict:
        if "actions" not in episode:
            raise KeyError(f"Episode {source} is missing required key `actions`")
        if episode["actions"].shape[-1] != self.action_dim:
            raise ValueError(
                f"Episode {source} action_dim={episode['actions'].shape[-1]} does not match config action_dim={self.action_dim}"
            )

        for camera in self.camera_names:
            key = f"{camera}_image"
            if key not in episode:
                raise KeyError(f"Episode {source} is missing required key `{key}`")

        if "lang_emb" in episode:
            lang_emb = np.asarray(episode["lang_emb"], dtype=np.float32)
            if lang_emb.ndim == 1:
                episode["lang_emb"] = lang_emb[None, :]
            elif lang_emb.ndim != 2:
                raise ValueError(f"Episode {source} lang_emb must have shape [D] or [1, D]")
        else:
            task_embedding = self._get_task_embedding(episode.get("task_name"))
            if task_embedding is not None:
                episode["lang_emb"] = task_embedding
            elif self.task_description is not None:
                episode["lang"] = self.task_description
            elif episode.get("task_name"):
                episode["lang"] = str(episode["task_name"])
            else:
                raise ValueError(
                    "RealRobotDataset requires one of: `lang_emb` in data, `task_embedding_path`, `task_description`, or `task_name`. "
                    f"Missing task goal for episode {source}."
                )

        return episode

    def _build_index(self) -> list[EpisodeIndex]:
        index: list[EpisodeIndex] = []
        for episode_id, episode in enumerate(self.episodes):
            episode_len = int(episode["actions"].shape[0])
            if episode_len < self.window_size:
                continue
            for start in range(0, episode_len - self.window_size + 1):
                index.append(EpisodeIndex(episode_id=episode_id, start=start))
        if not index:
            raise ValueError("No valid training windows could be built from the real robot dataset")
        return index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        item = self.index[idx]
        episode = self.episodes[item.episode_id]
        start = item.start
        end = start + self.window_size

        obs_dict: dict[str, torch.Tensor] = {}

        for camera in self.camera_names:
            key = f"{camera}_image"
            obs_dict[key] = self._to_image_tensor(episode[key][start:end])

        if "robot_states" in episode:
            robot_states = np.asarray(episode["robot_states"][start:end], dtype=np.float32)
            obs_dict["robot_states"] = torch.from_numpy(robot_states)

        if "point_cloud" in episode:
            point_cloud = np.asarray(episode["point_cloud"][start:end], dtype=np.float32)
            obs_dict["point_cloud"] = torch.from_numpy(point_cloud)

        if "lang_emb" in episode:
            obs_dict["lang_emb"] = torch.from_numpy(np.asarray(episode["lang_emb"], dtype=np.float32))
        elif "lang" in episode:
            obs_dict["lang"] = episode["lang"]

        actions = torch.from_numpy(np.asarray(episode["actions"][start:end], dtype=np.float32))
        mask = torch.ones(self.window_size, dtype=torch.float32)
        return obs_dict, actions, mask

    def get_all_actions(self) -> torch.Tensor:
        actions = np.concatenate([np.asarray(episode["actions"], dtype=np.float32) for episode in self.episodes], axis=0)
        return torch.from_numpy(actions)

    @staticmethod
    def _to_image_tensor(images: np.ndarray) -> torch.Tensor:
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"Image tensor must have 4 dims [T, H, W, C] or [T, C, H, W], got {images.shape}")
        if images.shape[-1] in (1, 3):
            images = np.transpose(images, (0, 3, 1, 2))
        images = images.astype(np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        return torch.from_numpy(images)
