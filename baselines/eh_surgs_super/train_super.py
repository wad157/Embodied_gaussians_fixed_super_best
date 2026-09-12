#!/usr/bin/env python3
"""Train pinned EH-SurGS on legal SUPER prefix observations."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

import train as upstream_train
from arguments import FDMHiddenParams, ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams

from common import SuperScene
from protocol import (
    ADAPTER_VERSION,
    DATASETS,
    TRAIN_DOWNSAMPLE,
    UPSTREAM_COMMIT,
    depth_cache,
    resolve_native,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "super_unified.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hidden = FDMHiddenParams(parser)
    parser.add_argument("--dataset-key", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--super-downsample", type=int, default=TRAIN_DOWNSAMPLE)
    parser.add_argument("--super-init-points", type=int, default=30000)
    parser.add_argument("--super-init-views", type=int, default=64)
    parser.add_argument("--configs", default=str(DEFAULT_CONFIG))
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", default=None)
    parser.add_argument("--expname", default="")
    args = parser.parse_args()
    import mmcv

    args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    if not args.model_path:
        parser.error("--model_path must be explicit")
    native = resolve_native(REPO_ROOT, args.dataset_key, Path(args.source_path))
    cache = depth_cache(REPO_ROOT, args.dataset_key, args.super_downsample)
    if not (cache / "COMPLETE").is_file():
        parser.error("Depth cache is incomplete: {}".format(cache))
    model_path = Path(args.model_path).expanduser().resolve()
    if model_path.exists() and any(model_path.iterdir()):
        parser.error("Refusing a non-empty checkpoint directory: {}".format(model_path))
    args.source_path = str(native)
    args.model_path = str(model_path)
    args.super_repo_root = str(REPO_ROOT)
    args.super_dataset_key = args.dataset_key
    args.super_seed = int(args.seed)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    return args, model, optimization, pipeline, hidden


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args, model, optimization, pipeline, hidden = parse_args()
    safe_state(args.quiet)
    reset_seed(args.seed)
    upstream_train.Scene = SuperScene
    upstream_train.args = args
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_args = model.extract(args)
    for name in (
        "super_repo_root",
        "super_dataset_key",
        "super_downsample",
        "super_init_points",
        "super_init_views",
        "super_seed",
    ):
        setattr(model_args, name, getattr(args, name))
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    os.chdir(args.model_path)
    print("EH-SurGS upstream commit: {}".format(UPSTREAM_COMMIT))
    print("SUPER adapter: {}; dataset: {}; seed: {}".format(
        ADAPTER_VERSION, args.dataset_key, args.seed
    ))
    upstream_train.training(
        model_args,
        hidden.extract(args),
        optimization.extract(args),
        pipeline.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.expname,
        args.extra_mark,
        True,
    )
    if int(args.iterations) == 3000:
        mask = Path(args.model_path) / "deformation_mask.npy"
        checkpoint = Path(args.model_path) / "point_cloud" / "iteration_3000" / "point_cloud.ply"
        if not mask.is_file() or not checkpoint.is_file():
            raise RuntimeError("Formal training ended without checkpoint/deformation mask")
    print("Training complete.")


if __name__ == "__main__":
    main()
