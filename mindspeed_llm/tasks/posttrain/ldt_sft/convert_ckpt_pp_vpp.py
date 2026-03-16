# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""
Bidirectional conversion between Pp and VPP checkpoint format.

Subcommands:
    merge - PP -> VPP: Merge standard PP checkpoints into VPP format
    split - VPP -> PP: Split VPP checkpoints to standard PP format

Examples:
    # PP -> VPP
    python mindspeed_llm\tasks\layerwise_disaggregated_training\convert_ckpt_pp_vpp.py merge \
        --load-dir ./model_weights/qwen2.5_7b_mcore_tp1pp5/ \
        --save-dir-edge ./model_weights/qwen2.5_7b_vpp_edge/ \
        --save-dir-cloud ./model_weights/qwen2.5_7b_vpp_cloud/ \
        --merge-stages 0,4 \
        --middle-stages 1,2,3

    # VPP -> PP
    python mindspeed_llm\tasks\layerwise_disaggregated_training\convert_ckpt_pp_vpp.py split \
        --load-dir-edge ./save_dir/edge/ \
        --load-dir-cloud ./save_dir/cloud/ \
        --save-dir ./save_dir/qwen2.5_7b_tp1pp5/ \
        --split-rank 0 \
        --middle-ranks 1,2,3

"""

import argparse
import copy
import os
import logging as logger

import torch

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def get_checkpoint_name(checkpoints_path, iteration, tensor_rank, pipeline_rank):
    """Get the checkpoint file path for a specific TP/PP rank"""
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
    if not os.path.join(tracker_filename):
        raise FileNotFoundError(f"Tracker file not found: {tracker_filename}")

    with open(tracker_filename, "r") as f:
        return int(f.read().strip())


def find_tp_ranks(iter_dir):
    tp_ranks = set()
    for dirname in os.listdir(iter_dir):
        if dirname.startswith('mp_rank_'):
            parts = dirname.split("_")
            if len(parts) >= 3:
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


def get_iteration(args):
    return (
        args.iteration if args.iteration is not None else read_iteration(args.load_dir)
    )


def get_iteration_from_dir(load_dir, iteration_override=None):
    return iteration_override if iteration_override is not None else read_iteration(load_dir)


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


def prepare_iter_dir(args, iteration):
    iter_dir = os.path.join(args.load_dir, f"iter_{iteration:07d}")
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")
    return iter_dir


def merge_checkpoints(args):
    """Merge standard PP checkpoints into VPP format(edge/cloud split)."""
    merge_stages = [int(x.strip()) for x in args.merge_stages.split(",")]
    middle_stages = []
    if args.middle_stages:
        middle_stages = [int(x.strip()) for x in args.middle_stages.split(",")]

    num_vpp_stages = len(merge_stages)
    
    edge_save_dir = args.save_dir_edge
    cloud_save_dir = args.save_dir_cloud

    logger.info(f"=== Checkpoint Merge (PP -> VPP) ===")
    logger.info(f"Merge stages (to VPP, edge): {merge_stages}")
    logger.info(f"Middle stages (convert to VPP, cloud): {middle_stages}")
    logger.info(f"Edge output: {edge_save_dir}")
    logger.info(f"Cloud output: {cloud_save_dir}")

    iteration = get_iteration(args)
    logger.debug(f"Using iteration: {iteration}")

    iter_dir = prepare_iter_dir(args, iteration)
    tp_ranks = find_tp_ranks(iter_dir)
    logger.debug(f"Found TP ranks: {tp_ranks}")

    edge_iter_dir = os.path.join(edge_save_dir, f'iter_{iteration:07d}')
    cloud_iter_dir = os.path.join(cloud_save_dir, f'iter_{iteration:07d}')
    os.makedirs(edge_iter_dir, exist_ok=True)
    if middle_stages:
        os.makedirs(cloud_iter_dir, exist_ok=True)

    for tp_rank in tp_ranks:
        logger.info(f"--- Processing TP rank {tp_rank} ---")

        # 1. Merge specified stages into VPP format -> edge dir (PP rank 0)
        logger.info(f" [1] Merging stages {merge_stages} into VPP format (edge) ...")
        merged_state_dict = None

        for vpp_idx, old_pp_rank in enumerate(merge_stages):
            ckpt_path = get_checkpoint_name(
                args.load_dir, iteration, tp_rank, old_pp_rank
            )
            logger.info(f"  Loading old PP={old_pp_rank} from: {ckpt_path}")
            state_dict = load_ckpt(ckpt_path)

            if merged_state_dict is None:
                merged_state_dict = {
                    "args": state_dict.get("args"),
                    "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                    "iteration": state_dict.get("iteration", iteration),
                }
                merged_state_dict.update(copy_metadata(state_dict))

            model_key = f"model{vpp_idx}"
            if "model" in state_dict:
                merged_state_dict[model_key] = state_dict["model"]
                logger.info(
                    f"  -> Saved as '{model_key} ({len(state_dict['model'])} keys)"
                )
            else:
                logger.warning(f"   No 'model' key in checkpoint")

        if merged_state_dict.get("args") is not None:
            merged_state_dict["args"].virtual_pipeline_model_parallel_size = (
                num_vpp_stages
            )
            merged_state_dict["args"].pipeline_model_parallel_size = 1

        save_checkpoint(edge_iter_dir, tp_rank, 0, merged_state_dict)

        # 2. Convert midele stages to VPP format -> cloud dir (PP rank from 1)
        for idx, old_pp_rank in enumerate(middle_stages):
            cloud_pp_rank = idx + 1 # cloud dir PP ranks start from 1
            old_ckpt_path = get_checkpoint_name(
                args.load_dir, iteration, tp_rank, old_pp_rank
            )
            logger.info(
                f"  [2] Converting old PP={old_pp_rank} to VPP format as cloud PP={cloud_pp_rank} ..."
            )
            state_dict = load_ckpt(old_ckpt_path)

            new_state_dict = {
                "args": state_dict.get("args"),
                "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                "iteration": state_dict.get("iteration", iteration),
            }
            new_state_dict.update(copy_metadata(state_dict))

            if "model" in state_dict:
                new_state_dict["model0"] = state_dict["model"]
                new_state_dict["model1"] = {}
                logger.info(
                    f"  -> model0: original data ({len(state_dict['model'])} keys), model1: empty"
                )
            else:
                new_state_dict["model0"] = {}
                new_state_dict["model1"] = {}
                logger.warning(f"  No 'model' key, both model0/model1 empty")

            if new_state_dict.get("args") is not None:
                new_state_dict["args"].pipeline_model_parallel_size = len(middle_stages)
                new_state_dict["args"].virtual_pipeline_model_parallel_size = num_vpp_stages

            save_checkpoint(cloud_iter_dir, tp_rank, cloud_pp_rank, new_state_dict)

    save_tracker(edge_save_dir, iteration)
    if middle_stages:
        save_tracker(cloud_save_dir, iteration)

    logger.info(f"=== Merge complete ===")
    logger.info(f"Edge:{edge_save_dir} (PP=1, VPP={num_vpp_stages})")
    if middle_stages:
        logger.info(f"Cloud:{cloud_save_dir}(PP={len(middle_stages)})")


def split_checkpoints(args):
    """split VPP checkpoints(edge/cloud) to standard PP format."""

    split_rank = args.split_rank
    middle_ranks = []
    if args.middle_ranks:
        middle_ranks = [int(x.strip()) for x in args.middle_ranks.split(",")]

    total_new_pp_stages = 2 + len(middle_ranks)

    edge_load_dir = args.load_dir_edge
    cloud_load_dir = args.load_dir_cloud

    logger.info(f"=== Checkpoint split (VPP -> PP) ===")
    logger.info(f"Edge input:{edge_load_dir}")
    logger.info(f"Cloud input:{cloud_load_dir}")
    logger.info(f"Split VPP rank(edge): {split_rank}")
    logger.info(f"Midele ranks(cloud): {middle_ranks}")
    logger.info(f"Output PP size: {total_new_pp_stages}")

    iteration = get_iteration_from_dir(edge_load_dir, args.iteration)
    logger.debug(f"Using iteration: {iteration}")

    edge_iter_dir = os.path.join(edge_load_dir, f'iter_{iteration:07d}')
    if not os.path.isdir(edge_iter_dir):
        raise FileNotFoundError(f"Edge iteration directory not found:{edge_iter_dir}")
    
    tp_ranks = find_tp_ranks(edge_iter_dir)
    logger.debug(f"Found TP ranks: {tp_ranks}")

    save_iter_dir = os.path.join(args.save_dir, f"iter_{iteration:07d}")
    os.makedirs(save_iter_dir, exist_ok=True)

    for tp_rank in tp_ranks:
        logger.info(f"--- Processing TP rank {tp_rank} ---")

        # 1. Split edge VPP rank into first and last PP stages
        vpp_ckpt_path = get_checkpoint_name(
            edge_load_dir, iteration, tp_rank, split_rank
        )
        logger.info(
            f" [1] Splitting edge VPP rank {split_rank} into PP=0 and PP={total_new_pp_stages - 1} ..."
        )
        vpp_state_dict = load_ckpt(vpp_ckpt_path)

        base_metadata = {
            "checkpoint_version": vpp_state_dict.get("checkpoint_version", 3.0),
            "iteration": vpp_state_dict.get("iteration", iteration),
        }

        # pp=0: model0 -> model
        first_state_dict = dict(base_metadata)
        first_state_dict["args"] = copy.deepcopy(vpp_state_dict.get("args"))
        first_state_dict.update(copy_metadata(vpp_state_dict))
        model0 = vpp_state_dict.get("model0", {})
        first_state_dict["model"] = model0
        logger.info(f"  -> PP=0: model from model0 ({len(model0)}) keys")
        if first_state_dict.get("args") is not None:
            first_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
            first_state_dict["args"].virtual_pipeline_model_parallel_size = None
        save_checkpoint(save_iter_dir, tp_rank, 0, first_state_dict)

        # pp=last: model1 -> model
        last_pp_rank = total_new_pp_stages - 1
        last_state_dict = dict(base_metadata)
        last_state_dict["args"] = copy.deepcopy(vpp_state_dict.get("args"))
        model1 = vpp_state_dict.get("model1", {})
        last_state_dict["model"] = model1
        logger.info(f"  -> PP={last_pp_rank}: model from model1 ({len(model1)}) keys")
        if last_state_dict.get("args") is not None:
            last_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
            last_state_dict["args"].virtual_pipeline_model_parallel_size = None
        save_checkpoint(save_iter_dir, tp_rank, last_pp_rank, last_state_dict)

        # 2. Convert cloud VPP ranks to standard PP (middle stages)
        for idx, cloud_pp_rank in enumerate(middle_ranks):
            new_pp_rank = idx + 1 # middle PP ranks start from 1
            old_ckpt_path = get_checkpoint_name(
                cloud_load_dir, iteration, tp_rank, cloud_pp_rank
            )
            logger.info(
                f" [2] Converting cloud PP={cloud_pp_rank} to PP={new_pp_rank} ..."
            )
            state_dict = load_ckpt(old_ckpt_path)

            new_state_dict = {
                "args": state_dict.get("args"),
                "checkpoint_version": state_dict.get("checkpoint_version", 3.0),
                "iteration": state_dict.get("iteration", iteration),
            }
            new_state_dict.update(copy_metadata(state_dict))
            model_data = state_dict.get('model0', state_dict.get('model', {}))
            new_state_dict['model'] = model_data
            logger.info(f"  -> model from mode0 ({len(model_data)}) keys, discarded model1")
            if new_state_dict.get('args') is not None:
                new_state_dict["args"].pipeline_model_parallel_size = total_new_pp_stages
                new_state_dict["args"].virtual_pipeline_model_parallel_size = None
            save_checkpoint(save_iter_dir, tp_rank, new_pp_rank, new_state_dict)
    
    save_tracker(args.save_dir, iteration)
    logger.info(f"=== Split complete ===")
    logger.info(f"Output: {args.save_dir}, PP size: {total_new_pp_stages}")


def main():
    parser = argparse.ArgumentParser(
        description='Bidirectional conversion between PP and VPP checkpoint formats'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # merge subcommand
    merge_parser = subparsers.add_parser('merge', help='convert PP -> VPP')
    merge_parser.add_argument('--load-dir', type=str, required=True)
    merge_parser.add_argument('--save-dir-edge', type=str, required=True,
                              help='Output dir for edge (first+last layers VPP)')
    merge_parser.add_argument('--save-dir-cloud', type=str, required=True,
                              help='Output dir for cloud (middle layers VPP)')
    merge_parser.add_argument('--merge-stages', type=str, required=True,
                              help='PP stage indices to merge into VPP. e.g. "0,4"')
    merge_parser.add_argument('--middle-stages', type=str, default=None,
                              help='PP stage indices for middle stages. e.g. "1,2,3"')
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