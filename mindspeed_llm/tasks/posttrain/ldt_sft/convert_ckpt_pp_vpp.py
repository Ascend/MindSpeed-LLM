# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Bidirectional conversion between PP and VPP checkpoint format.

Supports VTP (Virtual Tensor Parallelism): edge and cloud stages can have
different TP sizes. TP ranks are discovered per stage group, not globally.

Subcommands:
    merge - PP -> VPP: Merge standard PP checkpoints into VPP format
    split - VPP -> PP: Split VPP checkpoints to standard PP format

Examples:
    # PP -> VPP (single source, uniform TP)
    python convert_ckpt_pp_vpp.py merge \
        --load-dir ./pp4_ckpt/ \
        --save-dir-edge ./vpp_edge/ \
        --save-dir-cloud ./vpp_cloud/ \
        --merge-stages 0,3 \
        --merge-cloud-stages 1,2

    # PP -> VPP (dual source, different TP per side)
    python convert_ckpt_pp_vpp.py merge \
        --load-dir-edge ./pp4_tp1/ \
        --load-dir-cloud ./pp4_tp8/ \
        --save-dir-edge ./vpp_edge/ \
        --save-dir-cloud ./vpp_cloud/ \
        --merge-stages 0,3 \
        --merge-cloud-stages 1,2

    # PP -> VPP (uniform TP, cloud has multiple PP ranks)
    python convert_ckpt_pp_vpp.py merge \
        --load-dir ./pp5_ckpt/ \
        --save-dir-edge ./vpp_edge/ \
        --save-dir-cloud ./vpp_cloud/ \
        --merge-stages 0,4 \
        --middle-stages 1,2,3

    # VPP -> PP
    python convert_ckpt_pp_vpp.py split \
        --load-dir-edge ./vpp_edge/ \
        --load-dir-cloud ./vpp_cloud/ \
        --save-dir ./pp4_ckpt/ \
        --split-rank 0 \
        --split-cloud-rank 0 \
        --num-cloud-vpp-chunks 2

"""

import argparse
import copy
import os
import logging as logger

import torch

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def get_checkpoint_name(checkpoints_path, iteration, tensor_rank, pipeline_rank):
    """Get the checkpoint file path for a specific TP/PP rank."""
    directory = f"iter_{iteration:07d}"
    return os.path.join(
        checkpoints_path,
        directory,
        f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}",
        "model_optim_rng.pt",
    )


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, "latest_checkpointed_iteration.txt")


def read_iteration(checkpoints_path):
    tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
    if not os.path.isfile(tracker_filename):
        raise FileNotFoundError(f"Tracker file not found: {tracker_filename}")

    with open(tracker_filename, "r") as f:
        return int(f.read().strip())


def find_tp_ranks_for_stage(iter_dir, pp_rank):
    """Find TP ranks available for a specific PP stage.

    Needed for VTP where different PP stages have different TP sizes.
    E.g., edge (PP=0) has TP=1, cloud (PP=1) has TP=8.
    """
    tp_ranks = set()
    for dirname in os.listdir(iter_dir):
        if dirname.startswith('mp_rank_'):
            parts = dirname.split("_")
            # mp_rank_XX_YYY -> parts = ['mp', 'rank', 'XX', 'YYY']
            if len(parts) >= 4 and int(parts[3]) == pp_rank:
                tp_ranks.add(int(parts[2]))
    return sorted(tp_ranks)


def save_checkpoint(save_iter_dir, tp_rank, pp_rank, state_dict):
    save_subdir = os.path.join(save_iter_dir, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}")
    os.makedirs(save_subdir, exist_ok=True)
    save_path = os.path.join(save_subdir, "model_optim_rng.pt")
    logger.info(f"  Saving to: {save_path}")
    torch.save(state_dict, save_path)


def save_tracker(save_dir, iteration):
    tracker_path = get_checkpoint_tracker_filename(save_dir)
    os.makedirs(os.path.dirname(tracker_path) or ".", exist_ok=True)

    with open(tracker_path, "w") as f:
        f.write(str(iteration))
    logger.info(f"Saved iteration tracker: {tracker_path}")


def copy_metadata(state_dict):
    meta = {}
    for key in ["optimizer", "opt_param_scheduler", "rng_state"]:
        if key in state_dict:
            meta[key] = state_dict[key]
    return meta


def load_ckpt(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def prepare_iter_dir(load_dir, iteration):
    iter_dir = os.path.join(load_dir, f"iter_{iteration:07d}")
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")
    return iter_dir


def _merge_stages_to_vpp(iter_dir, load_dir, iteration, stages, tp_ranks,
                          save_iter_dir, save_pp_rank, label):
    """Merge multiple PP stages into a single VPP checkpoint.

    Each stage becomes a VPP chunk (model0, model1, ...).
    Iterates only over the given tp_ranks (VTP-safe).
    """
    num_vpp = len(stages)

    for tp_rank in tp_ranks:
        logger.info(f"--- {label} TP rank {tp_rank} ---")
        merged = None

        for vpp_idx, old_pp_rank in enumerate(stages):
            ckpt_path = get_checkpoint_name(load_dir, iteration, tp_rank, old_pp_rank)
            logger.info(f"  Loading PP={old_pp_rank} from: {ckpt_path}")
            state_dict = load_ckpt(ckpt_path)

            if merged is None:
                merged = {
                    "args": state_dict.get("args"),
                    "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                    "iteration": state_dict.get("iteration", iteration),
                }
                merged.update(copy_metadata(state_dict))

            model_key = f"model{vpp_idx}"
            if "model" in state_dict:
                merged[model_key] = state_dict["model"]
                logger.info(f"  -> {model_key}: {len(state_dict['model'])} keys")
            else:
                logger.warning(f"  No 'model' key in checkpoint PP={old_pp_rank}")

        if merged is not None and merged.get("args") is not None:
            merged["args"].virtual_pipeline_model_parallel_size = num_vpp
            merged["args"].pipeline_model_parallel_size = 1

        save_checkpoint(save_iter_dir, tp_rank, save_pp_rank, merged)


def merge_checkpoints(args):
    """Merge standard PP checkpoints into VPP format (edge/cloud split).

    Supports two source modes:
    1. Single source (--load-dir): all stages from one directory
    2. Dual source (--load-dir-edge + --load-dir-cloud): edge and cloud from
       separately converted directories (e.g., different TP sizes)

    Supports two cloud modes:
    --merge-cloud-stages: merge cloud stages into a single VPP checkpoint
    --middle-stages: keep cloud stages as separate PP ranks
    """
    merge_stages = [int(x.strip()) for x in args.merge_stages.split(",")]

    merge_cloud_stages = []
    if args.merge_cloud_stages:
        merge_cloud_stages = [int(x.strip()) for x in args.merge_cloud_stages.split(",")]

    middle_stages = []
    if args.middle_stages:
        middle_stages = [int(x.strip()) for x in args.middle_stages.split(",")]

    if merge_cloud_stages and middle_stages:
        raise ValueError("--merge-cloud-stages and --middle-stages are mutually exclusive")

    # Determine source mode
    dual_source = (hasattr(args, 'load_dir_edge') and args.load_dir_edge
                   and hasattr(args, 'load_dir_cloud') and args.load_dir_cloud)

    num_edge_vpp = len(merge_stages)
    edge_save_dir = args.save_dir_edge
    cloud_save_dir = args.save_dir_cloud

    logger.info(f"=== Checkpoint Merge (PP -> VPP) ===")
    logger.info(f"Mode: {'Dual source' if dual_source else 'Single source'}")
    logger.info(f"Edge merge stages: {merge_stages} -> VPP={num_edge_vpp}")
    if merge_cloud_stages:
        logger.info(f"Cloud merge stages: {merge_cloud_stages} -> VPP={len(merge_cloud_stages)}")
    if middle_stages:
        logger.info(f"Cloud middle stages (separate PP): {middle_stages}")
    logger.info(f"Edge output: {edge_save_dir}")
    logger.info(f"Cloud output: {cloud_save_dir}")

    if dual_source:
        edge_load_dir = args.load_dir_edge
        cloud_load_dir = args.load_dir_cloud
        iteration = (args.iteration if args.iteration is not None
                     else read_iteration(edge_load_dir))
        edge_iter_dir = prepare_iter_dir(edge_load_dir, iteration)
        cloud_iter_dir = prepare_iter_dir(cloud_load_dir, iteration)
    else:
        iteration = (args.iteration if args.iteration is not None
                     else read_iteration(args.load_dir))
        iter_dir = prepare_iter_dir(args.load_dir, iteration)
        edge_load_dir = cloud_load_dir = args.load_dir
        edge_iter_dir = cloud_iter_dir = iter_dir

    edge_save_iter_dir = os.path.join(edge_save_dir, f'iter_{iteration:07d}')
    cloud_save_iter_dir = os.path.join(cloud_save_dir, f'iter_{iteration:07d}')
    os.makedirs(edge_save_iter_dir, exist_ok=True)
    if merge_cloud_stages or middle_stages:
        os.makedirs(cloud_save_iter_dir, exist_ok=True)

    # 1. Merge edge stages into VPP
    edge_tp_ranks = find_tp_ranks_for_stage(edge_iter_dir, merge_stages[0])
    logger.info(f"Edge TP ranks (from PP={merge_stages[0]}): {edge_tp_ranks}")
    _merge_stages_to_vpp(
        edge_iter_dir, edge_load_dir, iteration, merge_stages,
        edge_tp_ranks, edge_save_iter_dir, save_pp_rank=0, label="Edge"
    )

    # 2. Merge cloud stages into single VPP checkpoint
    if merge_cloud_stages:
        cloud_tp_ranks = find_tp_ranks_for_stage(cloud_iter_dir, merge_cloud_stages[0])
        logger.info(f"Cloud TP ranks (from PP={merge_cloud_stages[0]}): {cloud_tp_ranks}")
        logger.info(f"Merging cloud stages {merge_cloud_stages} into PP rank 0 with VPP={len(merge_cloud_stages)}")
        _merge_stages_to_vpp(
            cloud_iter_dir, cloud_load_dir, iteration, merge_cloud_stages,
            cloud_tp_ranks, cloud_save_iter_dir, save_pp_rank=0, label="Cloud"
        )
        logger.info(f"Cloud merge complete. Should have {len(cloud_tp_ranks)} checkpoints at PP rank 0")

    # 3. Keep middle stages as separate PP ranks
    if middle_stages:
        logger.info(f"\n=== Processing middle stages: {middle_stages} ===")
        cloud_tp_ranks = find_tp_ranks_for_stage(cloud_iter_dir, middle_stages[0])
        logger.info(f"Middle stages TP ranks (from PP={middle_stages[0]}): {cloud_tp_ranks}")

        for tp_rank in cloud_tp_ranks:
            for idx, old_pp_rank in enumerate(middle_stages):
                cloud_pp_rank = idx + 1
                ckpt_path = get_checkpoint_name(cloud_load_dir, iteration, tp_rank, old_pp_rank)
                logger.info(f"  Converting PP={old_pp_rank} -> cloud PP={cloud_pp_rank}")
                state_dict = load_ckpt(ckpt_path)

                new_state_dict = {
                    "args": state_dict.get("args"),
                    "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                    "iteration": state_dict.get("iteration", iteration),
                }
                new_state_dict.update(copy_metadata(state_dict))

                if "model" in state_dict:
                    new_state_dict["model0"] = state_dict["model"]
                    new_state_dict["model1"] = {}
                else:
                    new_state_dict["model0"] = {}
                    new_state_dict["model1"] = {}

                if new_state_dict.get("args") is not None:
                    new_state_dict["args"].pipeline_model_parallel_size = 1 + len(middle_stages)
                    new_state_dict["args"].virtual_pipeline_model_parallel_size = 1

                save_checkpoint(cloud_save_iter_dir, tp_rank, cloud_pp_rank, new_state_dict)

    save_tracker(edge_save_dir, iteration)
    if merge_cloud_stages or middle_stages:
        save_tracker(cloud_save_dir, iteration)

    logger.info(f"\n=== Merge complete ===")
    logger.info(f"Edge: {edge_save_dir} (PP=1, VPP={num_edge_vpp})")
    if merge_cloud_stages:
        logger.info(f"Cloud: {cloud_save_dir} (PP=1, VPP={len(merge_cloud_stages)})")
    if middle_stages:
        logger.info(f"Cloud: {cloud_save_dir} (PP={len(middle_stages)})")


def split_checkpoints(args):
    """Split VPP checkpoints (edge/cloud) to standard PP format.

    Supports two cloud modes:
    --split-cloud-rank + --num-cloud-vpp-chunks: split merged cloud VPP
        back into separate PP stages
    --middle-ranks: convert separate cloud PP ranks (existing behavior)
    """
    split_rank = args.split_rank

    split_cloud_rank = getattr(args, 'split_cloud_rank', None)
    num_cloud_vpp = getattr(args, 'num_cloud_vpp_chunks', None)

    middle_ranks = []
    if args.middle_ranks:
        middle_ranks = [int(x.strip()) for x in args.middle_ranks.split(",")]

    edge_load_dir = args.load_dir_edge
    cloud_load_dir = args.load_dir_cloud

    # Determine total output PP stages
    if split_cloud_rank is not None and num_cloud_vpp:
        total_new_pp_stages = 2 + num_cloud_vpp  # first + last + cloud chunks
    else:
        total_new_pp_stages = 2 + len(middle_ranks)

    logger.info(f"=== Checkpoint Split (VPP -> PP) ===")
    logger.info(f"Edge input: {edge_load_dir}")
    logger.info(f"Cloud input: {cloud_load_dir}")
    logger.info(f"Split VPP rank (edge): {split_rank}")
    if split_cloud_rank is not None:
        logger.info(f"Split cloud VPP rank: {split_cloud_rank}, chunks: {num_cloud_vpp}")
    if middle_ranks:
        logger.info(f"Middle ranks (cloud): {middle_ranks}")
    logger.info(f"Output PP size: {total_new_pp_stages}")

    iteration = (args.iteration if args.iteration is not None
                 else read_iteration(edge_load_dir))

    edge_iter_dir = os.path.join(edge_load_dir, f'iter_{iteration:07d}')
    if not os.path.isdir(edge_iter_dir):
        raise FileNotFoundError(f"Edge iteration directory not found: {edge_iter_dir}")

    save_iter_dir = os.path.join(args.save_dir, f"iter_{iteration:07d}")
    os.makedirs(save_iter_dir, exist_ok=True)

    # 1. Split edge VPP into first and last PP stages
    edge_tp_ranks = find_tp_ranks_for_stage(edge_iter_dir, split_rank)
    logger.info(f"Edge TP ranks: {edge_tp_ranks}")

    for tp_rank in edge_tp_ranks:
        logger.info(f"--- Edge TP rank {tp_rank} ---")
        vpp_ckpt_path = get_checkpoint_name(
            edge_load_dir, iteration, tp_rank, split_rank
        )
        logger.info(f"  Splitting edge VPP rank {split_rank} -> PP=0 and PP={total_new_pp_stages - 1}")
        vpp_state_dict = load_ckpt(vpp_ckpt_path)

        base_metadata = {
            "checkpoint_version": vpp_state_dict.get("checkpoint_version", 3.0),
            "iteration": vpp_state_dict.get("iteration", iteration),
        }

        # PP=0: model0 -> model (first layers)
        first_state_dict = dict(base_metadata)
        first_state_dict["args"] = copy.deepcopy(vpp_state_dict.get("args"))
        first_state_dict.update(copy_metadata(vpp_state_dict))
        model0 = vpp_state_dict.get("model0", {})
        first_state_dict["model"] = model0
        logger.info(f"  -> PP=0: model from model0 ({len(model0)} keys)")
        if first_state_dict.get("args") is not None:
            first_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
            first_state_dict["args"].virtual_pipeline_model_parallel_size = None
        save_checkpoint(save_iter_dir, tp_rank, 0, first_state_dict)

        # PP=last: model1 -> model (last layers)
        last_pp_rank = total_new_pp_stages - 1
        last_state_dict = dict(base_metadata)
        last_state_dict["args"] = copy.deepcopy(vpp_state_dict.get("args"))
        model1 = vpp_state_dict.get("model1", {})
        last_state_dict["model"] = model1
        logger.info(f"  -> PP={last_pp_rank}: model from model1 ({len(model1)} keys)")
        if last_state_dict.get("args") is not None:
            last_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
            last_state_dict["args"].virtual_pipeline_model_parallel_size = None
        save_checkpoint(save_iter_dir, tp_rank, last_pp_rank, last_state_dict)

    # 2. Split merged cloud VPP into separate PP stages
    if split_cloud_rank is not None and num_cloud_vpp:
        cloud_iter_dir = os.path.join(cloud_load_dir, f'iter_{iteration:07d}')
        cloud_tp_ranks = find_tp_ranks_for_stage(cloud_iter_dir, split_cloud_rank)
        logger.info(f"Cloud TP ranks: {cloud_tp_ranks}")

        for tp_rank in cloud_tp_ranks:
            logger.info(f"--- Cloud TP rank {tp_rank} ---")
            cloud_ckpt_path = get_checkpoint_name(
                cloud_load_dir, iteration, tp_rank, split_cloud_rank
            )
            cloud_state_dict = load_ckpt(cloud_ckpt_path)

            base_metadata = {
                "checkpoint_version": cloud_state_dict.get("checkpoint_version", 3.0),
                "iteration": cloud_state_dict.get("iteration", iteration),
            }

            for vpp_idx in range(num_cloud_vpp):
                new_pp_rank = vpp_idx + 1  # cloud PP ranks start from 1
                model_key = f"model{vpp_idx}"
                model_data = cloud_state_dict.get(model_key, {})

                new_state_dict = dict(base_metadata)
                new_state_dict["args"] = copy.deepcopy(cloud_state_dict.get("args"))
                new_state_dict["model"] = model_data
                logger.info(f"  -> PP={new_pp_rank}: model from {model_key} ({len(model_data)} keys)")
                if new_state_dict.get("args") is not None:
                    new_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
                    new_state_dict["args"].virtual_pipeline_model_parallel_size = None
                save_checkpoint(save_iter_dir, tp_rank, new_pp_rank, new_state_dict)

    # 3. Convert separate cloud VPP ranks to PP
    if middle_ranks:
        cloud_iter_dir = os.path.join(cloud_load_dir, f'iter_{iteration:07d}')
        cloud_tp_ranks = find_tp_ranks_for_stage(cloud_iter_dir, middle_ranks[0])
        logger.info(f"Middle ranks TP ranks: {cloud_tp_ranks}")

        for tp_rank in cloud_tp_ranks:
            for idx, cloud_pp_rank in enumerate(middle_ranks):
                new_pp_rank = idx + 1
                old_ckpt_path = get_checkpoint_name(
                    cloud_load_dir, iteration, tp_rank, cloud_pp_rank
                )
                logger.info(f"  Converting cloud PP={cloud_pp_rank} -> PP={new_pp_rank}")
                state_dict = load_ckpt(old_ckpt_path)

                new_state_dict = {
                    "args": state_dict.get("args"),
                    "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                    "iteration": state_dict.get("iteration", iteration),
                }
                new_state_dict.update(copy_metadata(state_dict))
                model_data = state_dict.get('model0', state_dict.get('model', {}))
                new_state_dict['model'] = model_data
                logger.info(f"  -> model from model0 ({len(model_data)} keys)")
                if new_state_dict.get('args') is not None:
                    new_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
                    new_state_dict["args"].virtual_pipeline_model_parallel_size = None
                save_checkpoint(save_iter_dir, tp_rank, new_pp_rank, new_state_dict)

    save_tracker(args.save_dir, iteration)
    logger.info(f"\n=== Split complete ===")
    logger.info(f"Output: {args.save_dir}, PP size: {total_new_pp_stages}")


def main():
    parser = argparse.ArgumentParser(
        description='Bidirectional conversion between PP and VPP checkpoint formats'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # merge subcommand
    merge_parser = subparsers.add_parser('merge', help='convert PP -> VPP')
    merge_parser.add_argument('--load-dir', type=str, default=None,
                              help='Source checkpoint dir (single source, uniform TP)')
    merge_parser.add_argument('--load-dir-edge', type=str, default=None,
                              help='Source checkpoint dir for edge stages')
    merge_parser.add_argument('--load-dir-cloud', type=str, default=None,
                              help='Source checkpoint dir for cloud stages')
    merge_parser.add_argument('--save-dir-edge', type=str, required=True,
                              help='Output dir for edge (first+last layers VPP)')
    merge_parser.add_argument('--save-dir-cloud', type=str, required=True,
                              help='Output dir for cloud (middle layers VPP)')
    merge_parser.add_argument('--merge-stages', type=str, required=True,
                              help='PP stage indices to merge into edge VPP. e.g. "0,3"')
    merge_parser.add_argument('--merge-cloud-stages', type=str, default=None,
                              help='PP stage indices to merge into cloud VPP. e.g. "1,2". '
                                   'Mutually exclusive with --middle-stages.')
    merge_parser.add_argument('--middle-stages', type=str, default=None,
                              help='PP stage indices kept as separate cloud PP ranks. e.g. "1,2,3". '
                                   'Mutually exclusive with --merge-cloud-stages.')
    merge_parser.add_argument('--iteration', type=int, default=None)

    # split subcommand
    split_parser = subparsers.add_parser('split', help='convert VPP -> PP')
    split_parser.add_argument('--load-dir-edge', type=str, required=True,
                              help='Edge VPP checkpoint dir')
    split_parser.add_argument('--load-dir-cloud', type=str, required=True,
                              help='Cloud VPP checkpoint dir')
    split_parser.add_argument('--save-dir', type=str, required=True)
    split_parser.add_argument('--split-rank', type=int, default=0,
                              help='PP rank in edge dir containing VPP to split (default: 0)')
    split_parser.add_argument('--split-cloud-rank', type=int, default=None,
                              help='PP rank in cloud dir containing merged VPP to split. '
                                   'Mutually exclusive with --middle-ranks.')
    split_parser.add_argument('--num-cloud-vpp-chunks', type=int, default=None,
                              help='Number of VPP chunks in merged cloud checkpoint. '
                                   'Required with --split-cloud-rank.')
    split_parser.add_argument('--middle-ranks', type=str, default=None,
                              help='PP ranks in cloud dir to convert from VPP to PP. e.g. "1,2,3"')
    split_parser.add_argument('--iteration', type=int, default=None)

    args = parser.parse_args()
    if args.command == 'merge':
        merge_checkpoints(args)
    elif args.command == 'split':
        split_checkpoints(args)
    else:
        raise ValueError('only support merge and split')


if __name__ == '__main__':
    main()
