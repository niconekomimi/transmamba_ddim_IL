import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

LOG_ROOT = Path("/home/public/wyl/X_IL/logs")
OUTPUT_JSON = LOG_ROOT / "summary_success_rates.json"

# 匹配 "Average success rate: x.x"
AVG_PATTERN = re.compile(r"Average success rate:\s*([0-9]*\.?[0-9]+)")

def safe_number(v: str):
    if v in ("True", "False"):
        return v == "True"
    try:
        if re.fullmatch(r"[+-]?\d+", v):
            return int(v)
        return float(v)
    except Exception:
        return v

def parse_param_dir(param_str: str) -> Dict[str, Any]:
    """解析形如 'k=v,k=v,...' 的参数串为字典"""
    params: Dict[str, Any] = {}
    for seg in param_str.split(","):
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        params[k] = safe_number(v)
    return params

def extract_metadata(run_log_path: Path) -> Optional[Dict[str, Any]]:
    """
    结构: logs/<libero_task>/sweeps/<model>/<date>/<time>/<param_dirs...>/run.log
    日期仅记录，不分组
    """
    parts = run_log_path.parts
    try:
        idx = parts.index("logs")
    except ValueError:
        return None
    try:
        if parts[idx + 2] != "sweeps":
            return None
        libero_task = parts[idx + 1]
        model = parts[idx + 3]
        date = parts[idx + 4] if len(parts) > idx + 4 else ""
        time = parts[idx + 5] if len(parts) > idx + 5 else ""

        # 仅解析最后一层(倒数第二个路径片段)的参数目录，前面的不重要
        param_str = parts[-2] if len(parts) >= 2 else ""
        raw_params = parse_param_dir(param_str) if param_str else {}

        # 去掉 agent_name
        params = {k: v for k, v in raw_params.items() if k != "agent_name"}

        # 关键参数：traj_per_task / encoder / decoder
        traj_per_task = int(params.get("traj_per_task", 10))

        encoder_keys_priority = [
            "encoder_n_layer",
            "mamba_n_layer_encoder",
            "xlstm_encoder_blocks",
            "encoder_blocks",
        ]
        decoder_keys_priority = [
            "mamba_n_layer_decoder",
            "decoder_n_layer",
            "decoder_blocks",
        ]

        def pick_first_number(keys, mapping):
            for k in keys:
                if k in mapping and isinstance(mapping[k], (int, float)):
                    return mapping[k]
            return None

        encoder_val = pick_first_number(encoder_keys_priority, params)
        decoder_val = pick_first_number(decoder_keys_priority, params)

        key_params = {
            "traj_per_task": traj_per_task,
            "encoder": encoder_val,
            "decoder": decoder_val,
        }

        return {
            "libero_task": libero_task,
            "model": model,
            "date": date,
            "time": time,
            "params": params,
            "key_params": key_params
        }
    except IndexError:
        return None

def parse_average(run_log_path: Path) -> Optional[float]:
    """返回最后一次出现的平均成功率"""
    avg = None
    try:
        with run_log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = AVG_PATTERN.search(line)
                if m:
                    try:
                        avg = float(m.group(1))
                    except Exception:
                        pass
    except OSError:
        return None
    return avg

def collect():
    # 结果: { libero_task: { model: [ runs... ] } }
    result: Dict[str, Dict[str, list]] = {}
    for root, _, files in os.walk(LOG_ROOT):
        if "run.log" not in files:
            continue
        run_log = Path(root) / "run.log"

        meta = extract_metadata(run_log)
        if not meta:
            continue

        avg = parse_average(run_log)
        if avg is None:
            continue

        lt = meta["libero_task"]
        model = meta["model"]

        run_entry = {
            "date": meta["date"],
            "time": meta["time"],
            "average_success_rate": avg,
            "params": meta["params"],        # 已去掉 agent_name
            "key_params": meta["key_params"],# traj_per_task 数值；encoder/decoder 为数值或 null
            "run_log_path": str(run_log)
        }

        result.setdefault(lt, {}).setdefault(model, []).append(run_entry)

    # 按 date, time 排序
    for lt_models in result.values():
        for runs in lt_models.values():
            runs.sort(key=lambda r: (r.get("date", ""), r.get("time", ""), r.get("run_log_path", "")))

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"写入完成: {OUTPUT_JSON}")

if __name__ == "__main__":
    collect()