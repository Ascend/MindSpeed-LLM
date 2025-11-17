# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""
Note that we don't combine the main with trainer as trainer is used by other main.
"""
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict
import sys

import hydra
import ray
import torch
import yaml
from ray.util import placement_group

from mindspeed_rl.utils import seed_all
from mindspeed_rl.utils import get_tokenizer
from mindspeed_rl.utils.utils import MsProbe, get_node_nums
from mindspeed_rl.utils.loggers import Loggers
from mindspeed_rl.utils.utils import parse_args_from_config, init_torch_compile
from mindspeed_rl.config_cls.validate_config import validate_rl_args
from mindspeed_rl.config_cls.megatron_config import MegatronConfig
from mindspeed_rl.config_cls.rl_config import RLConfig
from mindspeed_rl.config_cls.generate_config import GenerateConfig
from mindspeed_rl.config_cls.mindstudio_config import ProfilerConfig, MsprobeConfig
from mindspeed_rl.datasets.prompt_dataset import PromptDataset
from mindspeed_rl.datasets.dataloader import PromptDataLoader
from mindspeed_rl.datasets.build_dataset import build_train_valid_test_datasets
from mindspeed_rl.workers.actor_hybrid_worker import ActorHybridWorker
from mindspeed_rl.workers.reference_worker import ReferenceWorker
from mindspeed_rl.workers.reward_worker import RewardWorker
from mindspeed_rl.workers.integrated_worker import IntegratedWorker
from mindspeed_rl.workers.rule_reward import RuleReward
from mindspeed_rl.workers.scheduler.launcher import construct_colocate_placement_groups

cur_file_dir = Path(__file__).absolute().parent
logger = Loggers("rlhf_train")


# Select different components based on the algorithm.
def get_algorithm_components(algorithm):
    logger.info(f'get_algorithm_components for {algorithm.upper()}')
    if algorithm == 'grpo':
        from mindspeed_rl.trainer.grpo_trainer_hybrid import RayGRPOTrainer as Trainer
        return Trainer, None, None
    elif algorithm == 'ppo':
        from mindspeed_rl.trainer.ppo_trainer_hybrid import RayPPOTrainer as Trainer
        from mindspeed_rl.workers.critic_worker import CriticWorker
        return Trainer, CriticWorker, None
    elif algorithm == 'dapo':
        from mindspeed_rl.trainer.dapo_trainer_hybrid import RayDAPOTrainer as Trainer
        from mindspeed_rl.workers.dynamic_sampling import DynamicSampling
        return Trainer, None, DynamicSampling
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Only 'grpo', 'ppo', and 'dapo' are supported.")


@ray.remote
def train(config):
    algorithm = config.get("megatron_training")["stage"][4:].lower()
    
    Trainer, CriticWorker, DynamicSampling = get_algorithm_components(algorithm)
    
    # Parse Configuration
    config_dict = parse_training_config(config, algorithm)
    
    if algorithm == 'ppo':
        actor_config, ref_config, reward_config, rl_config, generate_config, critic_config, profiler_config, msprobe_config = config_dict.values()
    else:
        actor_config, ref_config, reward_config, rl_config, generate_config, profiler_config, msprobe_config = config_dict.values()

    MsProbe.config_init(msprobe_config)
    configs_to_save = {
        'actor': actor_config.dict(),
        'rl': rl_config.dict(),
        'generate': generate_config.dict()
    }
    if algorithm != 'dapo':
        configs_to_save['ref'] = ref_config.dict()
        configs_to_save['reward'] = reward_config.dict()
    if algorithm == 'ppo':
        configs_to_save['critic'] = critic_config.dict()
    if algorithm == 'dapo':
        configs_to_save['reward'] = reward_config.dict()
    MsProbe.save_configs(configs_to_save)

    tokenizer = get_tokenizer(tokenizer_model=actor_config.tokenizer_name_or_path,
                              prompt_type=actor_config.prompt_type, prompt_type_path=actor_config.prompt_type_path)

    logger.info(f'start async initializing ray actor groups for {algorithm.upper()}')

    reward_list = []
    dynamic_sampling_list = []
    
    if algorithm == 'grpo' and hasattr(config.get('megatron_training', {}), "ai_framework") and config['megatron_training']['ai_framework'] == "mindspore":
        from mindspeed_rl.workers.scheduler.launcher_ms import RayActorGroupMs as RayActorGroup
    else:
        from mindspeed_rl.workers.scheduler.launcher import RayActorGroup

    if algorithm == 'ppo':
        pgs = construct_colocate_placement_groups(rl_config)
    else:
        pgs = None

    if rl_config.use_integrated_worker:
        integrated_worker = RayActorGroup(
            worker=IntegratedWorker,
            placement_group=pgs,
            megatron_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=gpt_model_provider,
            initialize_func=initialize_megatron,
            profiler_config=profiler_config["integrated"],
            msprobe_config=msprobe_config,
            tokenizer=tokenizer,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()

        actor_worker = integrated_worker

        if algorithm != 'dapo':
            reference_worker = integrated_worker

        if algorithm == 'ppo':
            critic_worker = RayActorGroup(
                worker=CriticWorker,
                placement_group=pgs,
                megatron_config=critic_config,
                rl_config=rl_config,
                model_provider=rm_model_provider,
                tokenizer=tokenizer,
                initialize_func=initialize_megatron,
                get_megatron_module=get_megatron_module,
                profiler_config=profiler_config["integrated"],
                msprobe_config=msprobe_config,
                global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
            ).initialize()

    else:
        actor_worker = RayActorGroup(
            worker=ActorHybridWorker,
            placement_group=None,
            megatron_config=actor_config,
            rl_config=rl_config,
            generate_config=generate_config,
            model_provider=gpt_model_provider,
            tokenizer=tokenizer,
            initialize_func=initialize_megatron,
            get_megatron_module=get_megatron_module,
            global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
        ).initialize()

        if algorithm != 'dapo':
            reference_worker = RayActorGroup(
                worker=ReferenceWorker,
                placement_group=None,
                megatron_config=ref_config,
                rl_config=rl_config,
                generate_config=generate_config if algorithm == 'grpo' else None,
                model_provider=gpt_model_provider,
                tokenizer=tokenizer,
                initialize_func=initialize_megatron,
                get_megatron_module=get_megatron_module,
                global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
            ).initialize()

        if rl_config.reward_resource:
            reward_worker = RayActorGroup(
                worker=RewardWorker,
                placement_group=None,
                megatron_config=reward_config,
                rl_config=rl_config,
                generate_config=generate_config if algorithm != 'ppo' else None,
                model_provider=rm_model_provider,
                tokenizer=tokenizer,
                initialize_func=initialize_megatron,
                get_megatron_module=get_megatron_module,
                global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
            ).initialize()

            reward_list.append(reward_worker)

        if algorithm == 'ppo':
            critic_worker = RayActorGroup(
                worker=CriticWorker,
                placement_group=None,
                megatron_config=critic_config,
                rl_config=rl_config,
                model_provider=rm_model_provider,
                tokenizer=tokenizer,
                initialize_func=initialize_megatron,
                get_megatron_module=get_megatron_module,
                profiler_config=profiler_config["integrated"],
                msprobe_config=msprobe_config,
                global_batch_size=actor_config.global_batch_size * rl_config.n_samples_per_prompt
            ).initialize()
    if algorithm != 'ppo':
        actor_config.max_prompt_length = rl_config.max_prompt_length
    num_process = get_node_nums()
    
    if rl_config.rule_reward:
        pg = placement_group(
            [{"CPU": rl_config.num_cpus_for_local_task} for _ in range(num_process)],
            strategy='SPREAD'
        )
        ray.get(pg.ready())
        for i in range(num_process):
            rule_reward = RuleReward.options(placement_group=pg, placement_group_bundle_index=i).remote()
            if algorithm == 'grpo':
                rule_reward.initialize.remote(reward_config, rl_config, tokenizer, generate_config.trust_remote_code, dp_rank=i)
            else:
                rule_reward.initialize.remote(reward_config, rl_config, tokenizer, dp_rank=i)
            reward_list.append(rule_reward)

    if algorithm == 'dapo' and rl_config.filter_groups_enable:
        pg = placement_group(
            [{"CPU": rl_config.num_cpus_for_local_task} for _ in range(num_process)],
            strategy='SPREAD'
        )
        ray.get(pg.ready())
        for i in range(num_process):
            dynamic_sampling = DynamicSampling.options(placement_group=pg, placement_group_bundle_index=i).remote()
            dynamic_sampling.initialize.remote(reward_config, rl_config)
            dynamic_sampling_list.append(dynamic_sampling)

    train_ds, _, _ = build_train_valid_test_datasets(
        data_prefix=[actor_config.data_path, ],
        splits_string=actor_config.split,
        seq_length=actor_config.seq_length,
        train_valid_test_num_samples=[
            actor_config.train_iters * actor_config.global_batch_size, 0, 0
        ],
        seed=actor_config.seed,
        dataset_cls=PromptDataset,
        extra_param=actor_config
    )
    logger.info('after dataset is built')

    actor_worker.wait_all_ref_objs_run_over()

    if algorithm != 'dapo':
        consumed_train_samples = actor_worker.get_consumed_train_samples()
        data_loader = PromptDataLoader(
            train_ds, actor_config.global_batch_size,
            actor_config.num_workers, actor_config.seed, actor_config.dataset_additional_keys,
            actor_config.no_shuffle
        )
        data_iters = iter(data_loader)
        [next(data_iters) for _ in range(consumed_train_samples // actor_config.global_batch_size)]
        logger.info('after dataloader is built')
    else:
        data_loader = PromptDataLoader(
            train_ds, actor_config.global_batch_size,
            actor_config.num_workers, actor_config.seed, actor_config.dataset_additional_keys,
            actor_config.no_shuffle
        )
        logger.info('after dataloader is built')

    if algorithm != 'dapo':
        reference_worker.wait_all_ref_objs_run_over()
    
    for reward in reward_list:
        if hasattr(reward, 'wait_all_ref_objs_run_over'):
            reward.wait_all_ref_objs_run_over()

    if algorithm == 'grpo':
        trainer = Trainer(
            actor_worker,
            reference_worker,
            reward_list,
            tokenizer=tokenizer,
            global_batch_size=actor_config.global_batch_size,
            micro_batch_size=rl_config.adv_dispatch_size,
            train_iters=actor_config.train_iters,
            save_interval=actor_config.save_interval,
            dataset_additional_keys=actor_config.dataset_additional_keys,
            **rl_config.dict()
        )
        trainer.fit(data_iters)
    elif algorithm == 'ppo':
        trainer = Trainer(
            actor_worker,
            reference_worker,
            reward_list,
            critic_worker,
            tokenizer=tokenizer,
            global_batch_size=actor_config.global_batch_size,
            train_iters=actor_config.train_iters,
            save_interval=actor_config.save_interval,
            dataset_additional_keys=actor_config.dataset_additional_keys,
            **rl_config.dict()
        )
        trainer.fit(data_iters)
    else: 
        trainer = Trainer(
            actor_worker,
            reward_list,
            dynamic_sampling_list,
            tokenizer=tokenizer,
            global_batch_size=actor_config.global_batch_size,
            micro_batch_size=rl_config.adv_dispatch_size,
            train_iters=actor_config.train_iters,
            save_interval=actor_config.save_interval,
            dataset_additional_keys=actor_config.dataset_additional_keys,
            **rl_config.dict()
        )
        trainer.fit(data_loader)

    logger.info(f"{algorithm.upper()} training process successfully!")


def parse_training_config(config: Dict, algorithm: str):
    """
    Parse the training configuration and extract different configuration items based on the algorithm type. 

    :param config: The input global configuration dictionary.
    :param algorithm: The type of algorithm, 'grpo', 'ppo' or 'dapo'
    :return: A dictionary containing the configuration. 
    """
    actor_config = MegatronConfig({**config.get("megatron_training"), **config.get("actor_config")},
                                  config.get('model'))
    rl_config = RLConfig(config.get("rl_config"))

    if rl_config.use_integrated_worker:
        if "ref_config" in config:
            raise ValueError(
                f"ref_config should not be set when use_integrated_worker mode is on.")
        
        if algorithm == 'dapo':
            ref_config = None
        else:
            ref_config = actor_config

        if "reward_config" in config:
            raise ValueError(
                f"reward_config should not be set when use_integrated_worker mode is on.")
        reward_config = actor_config

    else:
        if algorithm == 'dapo':
            ref_config = None
        else:
            ref_config = MegatronConfig({**config.get("megatron_training"), **config.get("ref_config")},
                                        config.get('model'))

        reward_config = MegatronConfig({**config.get("megatron_training"), **config.get("reward_config")},
                                       config.get('model'))

    generate_config = GenerateConfig(config.get("generate_config"))

    if algorithm == 'ppo':
        critic_config = MegatronConfig({**config.get("megatron_training"), **config.get("critic_config")},
                                      config.get('model'))
        validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config, critic_config)
    else:
        validate_rl_args(actor_config, ref_config, reward_config, rl_config, generate_config)

    profiler_config = {}
    profiler_config.update({
        "integrated": ProfilerConfig(
            config.get("profiler_config", {}).get("integrated", {}),
            role="integrated"
        ),
    })

    msprobe_config = MsprobeConfig(
            config.get("msprobe_config", {}),
            role="integrated"
        )


    if algorithm == 'ppo':
        return {
            "actor_config": actor_config,
            "ref_config": ref_config,
            "reward_config": reward_config,
            "rl_config": rl_config,
            "generate_config": generate_config,
            "critic_config": critic_config,
            "profiler_config": profiler_config,
            "msprobe_config": msprobe_config
        }
    else:
        return {
            "actor_config": actor_config,
            "ref_config": ref_config,
            "reward_config": reward_config,
            "rl_config": rl_config,
            "generate_config": generate_config,
            "profiler_config": profiler_config,
            "msprobe_config": msprobe_config
        }


def get_megatron_module():
    from megatron.core import parallel_state
    from megatron.core import DistributedDataParallel
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.training import get_optimizer_param_scheduler
    from megatron.training import get_args
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core import DistributedDataParallel as LocalDDP
    from megatron.core.transformer.module import Float16Module
    from megatron.training.training import get_model, unwrap_model
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
    from megatron.training.training import setup_model_and_optimizer
    from megatron.core.enums import ModelType
    from megatron.core.distributed import finalize_model_grads
    from mindspeed.utils import set_actual_seq_len, set_position_ids, get_actual_seq_len
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

    return {
        'parallel_state': parallel_state,
        'get_model': get_model,
        'get_megatron_optimizer': get_megatron_optimizer,
        'get_optimizer_param_scheduler': get_optimizer_param_scheduler,
        'load_checkpoint': load_checkpoint,
        'save_checkpoint': save_checkpoint,
        'get_args': get_args,
        'get_forward_backward_func': get_forward_backward_func,
        'float16_module': Float16Module,
        'unwrap_model': unwrap_model,
        'local_ddp': LocalDDP,
        'distributed_data_parallel_config': DistributedDataParallelConfig,
        'vocab_parallel_cross_entropy': vocab_parallel_cross_entropy,
        'setup_model_and_optimizer': setup_model_and_optimizer,
        'model_type': ModelType,
        'distributed_data_parallel': DistributedDataParallel,
        'finalize_model_grads': finalize_model_grads,
        'set_actual_seq_len': set_actual_seq_len,
        'get_actual_seq_len': get_actual_seq_len,
        'set_position_ids': set_position_ids,
        'distributed_optimizer': DistributedOptimizer,
        'float16_optimizer_with_float16_params': Float16OptimizerWithFloat16Params
    }


def gpt_model_provider(pre_process, post_process):
    """
    Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    from megatron.training import get_args
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args
    args = get_args()

    qk_layernorm = getattr(args, 'qk_layernorm', False)

    logger.info('building GPT model ...')
    
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, qk_layernorm=qk_layernorm)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
    )

    return model


def rm_model_provider(pre_process, post_process):
    """
    Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.

    Returns:
        GPTRewardModel: The returned model
    """
    from megatron.training import get_args
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.spec_utils import import_module
    from megatron.training.arguments import core_transformer_config_from_args
    from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel
    args = get_args()
    
    qk_layernorm = getattr(args, 'qk_layernorm', False)
    logger.info('building RM GPT model ...')
    # Experimental loading arguments from configs
    config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, qk_layernorm=qk_layernorm)

    if (not args.untie_embeddings_and_output_weights) and (args.pipeline_model_parallel_size > 1):
        args.untie_embeddings_and_output_weights = True
        logger.warning(
            "untie_embeddings_and_output_weights is set to True, "
            "since output_layer is not used in Outcome Reward model training."
        )
    model = GPTRewardModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        post_layer_norm=not args.no_post_layer_norm,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    return model


def initialize_megatron(
        extra_args_provider=None,
        args_defaults={},
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
        config=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    origin_sys_argv = sys.argv
    sys.argv = [sys.argv[0]]
    parse_args_from_config(config)

    # Initialize torch.compile global variables to avoid training-related patches affecting vLLM graph mode enabling.
    init_torch_compile(torch.compile)
    # Note: Importing this line activates the megatron_adapter.
    from mindspeed_llm.training.arguments import parse_args_decorator
    import megatron

    args = megatron.training.arguments.parse_args()
    sys.argv = origin_sys_argv

    if not allow_no_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Megatron requires CUDA.")

    from megatron.core import parallel_state
    from megatron.training import get_args
    from megatron.training.arguments import validate_args
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import _set_random_seed, \
        _init_autoresume, _compile_dependencies, \
        _initialize_tp_communicators

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        if args.load is None:
            raise ValueError("--use-checkpoints-args requires --load argument.")
        load_args_from_checkpoint(args)

    validate_args(args, args_defaults)

    set_global_variables(args)

    from mindspeed.core.tensor_parallel.lcal_coc.user_config import initialize_coc_from_cfg
    initialize_coc_from_cfg(args)

    if args.npu_deterministic:
        seed_all(args.seed)
        logger.info("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            logger.info("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)
        if args.use_ascend_mc2:
            from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
            initialize_cfg_from_args(args)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        parallel_state.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        parallel_state.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks):
    """Initialize torch.distributed and core model parallel."""
    from megatron.core import parallel_state
    from megatron.training import get_args
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            logger.info("torch distributed is already initialized, skipping initialization...")
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            logger.info("> initializing torch distributed...")
        # Manually set the device ids.
        if device_count > 0:
            if args.stage in ["ray_ppo", "ray_online_dpo", "ray_grpo", "ray_dapo"]:
                allocated_device = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
                torch.cuda.set_device(allocated_device)
            else:
                device = args.rank % device_count
                if args.local_rank is not None:
                    if args.local_rank != device:
                        raise ValueError("expected local-rank to be the same as rank % device-count.")
                else:
                    args.local_rank = device
                torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            logger.info("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                encoder_tensor_model_parallel_size=args.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=args.encoder_pipeline_model_parallel_size,
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=args.enable_gloo_process_groups,
            )
            if args.rank == 0:
                logger.info(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                logger.info(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )


@hydra.main(config_path='configs/rlhf', config_name='test_ppo_qwen25_7b_A3', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        stage = config.get("megatron_training")["stage"]
        algorithm = stage[4:].lower()    
        logger.info(f'start initializing local ray cluster for {algorithm.upper()}')
        rl_config = RLConfig(config.get("rl_config"))
        with open(os.path.join(cur_file_dir, "configs/rlhf/envs/runtime_env.yaml")) as file:
            runtime_env = yaml.safe_load(file)
        if algorithm == 'grpo' or algorithm == 'dapo':
            runtime_env["env_vars"]["IS_MULTIMODAL"] = str(rl_config.is_multimodal)
        logger.info(f"ray init with runtime_env: {runtime_env}")
        ray.init(runtime_env=runtime_env)

    ray.get(train.remote(config))


if __name__ == '__main__':
    main()