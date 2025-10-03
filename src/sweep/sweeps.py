import json
import random
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, Tuple, List

import numpy as np
from omegaconf import OmegaConf


def _assert_override_keys_exist(cfg, param_grid: Dict[str, Iterable[Any]]):
    for key in param_grid.keys():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            assert p in node, f"Invalid override {key}: section '{p}' not in config"
            node = node[p]
        last = parts[-1]
        assert last in node, (
            f"Invalid override {key}: '{last}' not found in section '{'.'.join(parts[:-1])}'"
        )


def _param_grid_iter(grid: Dict[str, Iterable[Any]]) -> Iterable[Tuple[Dict[str, Any], str]]:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        tag = "_".join([f"{k.replace('.', '-')}-{v}" for k, v in params.items()])
        yield params, tag


def _param_grid_random_samples(
    grid: Dict[str, Iterable[Any]],
    n_samples: int,
    seed: int,
) -> List[Tuple[Dict[str, Any], str]]:
    all_combos = list(_param_grid_iter(grid))
    rng = random.Random(seed)
    if n_samples >= len(all_combos):
        return all_combos
    return rng.sample(all_combos, n_samples)


def _apply_overrides_to_cfg(base_cfg, params: Dict[str, Any]):
    cfg = deepcopy(base_cfg)
    for k, v in params.items():
        parts = k.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return cfg


def _to_jsonable(obj):
    """Recursively convert tensors/np types/Path to JSON-serializable Python types."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu()
            if obj.numel() == 1:
                return _to_jsonable(obj.item())
            return _to_jsonable(obj.tolist())
    except Exception:
        pass

    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def sweep_experiments(
    base_cfg,
    param_grid: Dict[str, Iterable[Any]],
    run_single_trial,
    num_class,
    seeds: Iterable[int],
    out_dir: Optional[str] = None,
    sweep_name: str = "sweep",
    maximize: bool = True,
    result_file: str = "results",
    rank_metric: Optional[str] = None,
    random_sample: Optional[int] = None,
    random_seed: int = 42,
):
    _assert_override_keys_exist(base_cfg, param_grid)

    if out_dir is None:
        out_dir = f"runs/{base_cfg.dataset}/sweep/{sweep_name}"

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = base_dir / (result_file + ".jsonl")
    (base_dir / "_base_config.yaml").write_text(OmegaConf.to_yaml(base_cfg))
    (base_dir / "_param_grid.json").write_text(
        json.dumps({k: list(v) for k, v in param_grid.items()}, indent=2, ensure_ascii=False)
    )

    trials = (
        _param_grid_random_samples(param_grid, random_sample, random_seed)
        if random_sample is not None else
        list(_param_grid_iter(param_grid))
    )

    all_results: List[Dict[str, Any]] = []

    for idx, (params, tag) in enumerate(trials, start=1):
        trial_dir = base_dir / f"trial_{idx:03d}_{tag}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Trial {idx}/{len(trials)} ===")
        print(f"Params: {params}")
        print(f"Dir: {trial_dir}")

        trial_cfg = _apply_overrides_to_cfg(base_cfg, params)

        out = run_single_trial(
            cfg=trial_cfg,
            num_class=num_class,
            seeds=seeds,
        )

        if not isinstance(out, dict):
            out = {"result": out}

        record = {
            "tag": tag,
            "params": params,             
            "trial_dir": str(trial_dir),
            **out,
        }

        record = _to_jsonable(record)

        all_results.append(record)
        with result_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "sweep_name": sweep_name,
        "seeds": list(seeds),
        "n_trials": len(all_results),
        "results": all_results,           # already augmented with params/tag
        "rank_metric": rank_metric,
        "maximize": maximize,
    }

    best = None
    if rank_metric is not None:
        pairs = []
        for r in all_results:
            val = None
            if isinstance(r, dict):
                val = (
                    r.get("linear_agg", {}).get(rank_metric, None)
                    if isinstance(r.get("linear_agg", {}), dict)
                    else r.get(rank_metric, None)
                )
            if isinstance(val, (int, float)):
                pairs.append((val, r))
        if pairs:
            best = (max if maximize else min)(pairs, key=lambda x: x[0])[1]

    summary["best"] = best
    (base_dir / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False))

    if best is not None:
        print("\n=== BEST TRIAL ===")
        print("tag:", best.get("tag"))
        print("params:", best.get("params"))
        metric_val = (
            best.get("linear_agg", {}).get(rank_metric)
            if isinstance(best.get("linear_agg", {}), dict)
            else best.get(rank_metric)
        )
        print(f"{rank_metric}: {metric_val}")
    else:
        print("\n(no rank_metric provided or not found in results — summary saved)")

    return summary
