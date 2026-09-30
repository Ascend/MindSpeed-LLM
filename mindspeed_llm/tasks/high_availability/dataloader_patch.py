"""DataLoader patches used only by high-availability training."""

from functools import wraps

import torch
from megatron.legacy.data.data_samplers import RandomSeedDataset
from megatron.training import get_args

from mindspeed_llm.legacy.data.data_samplers import build_pretraining_data_loader


class HARestartableDataLoader(torch.utils.data.DataLoader):
    """Keep the loader reachable from its iterator for TFT sampler rewind."""

    def __iter__(self):
        iterator = super().__iter__()
        iterator._ha_dataloader = self
        return iterator


def ha_build_pretraining_data_loader(dataset, consumed_samples):
    """Build a persistent loader when TFT can safely rewind its sampler."""
    args = get_args()
    loader = build_pretraining_data_loader(dataset, consumed_samples)
    if loader is None:
        return None
    initial_consumed_samples = loader.batch_sampler.consumed_samples
    if args.num_workers == 0 or isinstance(dataset, RandomSeedDataset):
        loader._ha_initial_consumed_samples = initial_consumed_samples
        return loader
    # The generic MindSpeed-LLM builder already selected the sampler and collator.
    # Copy that configuration into an HA loader before any workers are started.
    ha_loader = HARestartableDataLoader(
        loader.dataset,
        batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        generator=loader.generator,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        pin_memory_device=loader.pin_memory_device,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        prefetch_factor=loader.prefetch_factor,
        in_order=loader.in_order,
        persistent_workers=True,
    )
    ha_loader._ha_initial_consumed_samples = initial_consumed_samples
    return ha_loader


def ha_build_train_valid_test_data_loaders_wrapper(fn):
    """Restore cyclic sampler position after the existing worker prewarm wrapper."""

    @wraps(fn)
    def wrapper(build_train_valid_test_datasets_provider):
        dataloaders = fn(build_train_valid_test_datasets_provider)
        if get_args().dataloader_type == 'cyclic':
            for dataloader in dataloaders:
                if dataloader is not None and hasattr(dataloader, '_ha_initial_consumed_samples'):
                    dataloader.batch_sampler.consumed_samples = dataloader._ha_initial_consumed_samples
        return dataloaders

    return wrapper
