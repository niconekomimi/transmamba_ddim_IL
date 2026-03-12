import argparse
import pickle
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.models.beso.models.networks.clip import build_model, tokenize  # noqa: E402


def derive_task_name(path: Path) -> str:
    name = path.stem
    if name.endswith("_demo"):
        name = name[:-5]
    return name


def collect_task_names(data_dir: Path) -> list[str]:
    task_names = []
    for path in sorted(data_dir.glob("*.hdf5")):
        task_names.append(derive_task_name(path))
    for path in sorted(data_dir.glob("*.h5")):
        task_name = derive_task_name(path)
        if task_name not in task_names:
            task_names.append(task_name)
    if not task_names:
        raise FileNotFoundError(f"No .hdf5 or .h5 files found under {data_dir}")
    return task_names


def load_text_encoder_from_checkpoint(ckpt_dir: Path) -> torch.nn.Module:
    state_dict = torch.load(ckpt_dir / "last_model.pth", map_location="cpu", weights_only=False)
    prefix = "language_encoder.clip_rn50."
    clip_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not clip_state:
        raise KeyError(f"No language encoder weights found in {(ckpt_dir / 'last_model.pth')}")
    return build_model(clip_state).to("cpu").eval()


def generate_embeddings(task_names: list[str], text_encoder: torch.nn.Module) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        tokens = tokenize(task_names).to("cpu")
        encoded = text_encoder.encode_text(tokens).float().cpu()
    for task_name, embedding in zip(task_names, encoded, strict=True):
        embeddings[task_name] = embedding.unsqueeze(0)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate task embeddings from a checkpoint text encoder.")
    parser.add_argument("--ckpt-dir", required=True, help="Directory containing last_model.pth")
    parser.add_argument("--data-dir", required=True, help="Directory containing task demo .hdf5 files")
    parser.add_argument("--output-path", required=True, help="Output .pkl path")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output_path).resolve()

    task_names = collect_task_names(data_dir)
    text_encoder = load_text_encoder_from_checkpoint(ckpt_dir)
    embeddings = generate_embeddings(task_names, text_encoder)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(embeddings, handle)

    print(f"Saved {len(embeddings)} task embeddings to {output_path}")
    for task_name in task_names:
        print(task_name)


if __name__ == "__main__":
    main()
