# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from typing import Tuple

import numpy
import torch

from megatron.core.datasets.utils import Split, log_single_rank
from megatron.core.datasets.gpt_dataset import (_build_document_index,
                                                _build_shuffle_index
                                                )
from mindspeed_llm.tasks.utils.error_utils import GPTDatasetSampleIndexError
from .blended_megatron_dataset_builder import need_to_build_dataset

logger = logging.getLogger(__name__)


def _build_document_sample_shuffle_indices(
    self,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Build the document index, the sample index, and the shuffle index

    The document index:
        -- 1-D
        -- An ordered array of document ids

    The sample index:
        -- 2-D
        -- The document indices and offsets which mark the start of every sample

    The shuffle index:
        -- 1-D
        -- A random permutation of index range of the sample index

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the shuffle index
    """
    path_to_cache = self.config.path_to_cache
    if path_to_cache is None and not self.config.mock:
        path_to_cache = os.path.join(
            self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
        )

    # start of megatron_adaptation,
    # here we change from (class)GPTDataset._build_document_sample_shuffle_indices
    # end of megatron_adaptation

    if path_to_cache:
        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )
    else:
        cache_hit = False

    if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or need_to_build_dataset())
    ):

        log_single_rank(
            logger,
            logging.INFO,
            f"Build and save the {type(self).__name__} {self.index_split.name} indices",
        )
        self.built_anew_on_cache_miss = True
        t_beg = time.time()


        sequence_length = self.config.sequence_length
        num_tokens_per_epoch = self._get_num_tokens_per_epoch()
        num_epochs = self._get_num_epochs(num_tokens_per_epoch)

        if num_epochs == 1:
            separate_final_epoch = False
        else:
            # Get the number of samples for the last epoch
            num_samples_sans_final_epoch = (
                                                   (num_epochs - 1) * num_tokens_per_epoch
                                                   - self.config.add_extra_token_to_sequence
                                           ) // sequence_length
            num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
            num_samples_per_epoch = (
                                            num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                                    ) // sequence_length

            # num_samples_from_final_epoch should be non-negative
            assert num_samples_from_final_epoch >= 0

            # num_samples_from_final_epoch should not exceed max value
            assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

            # Separate the final epoch if it falls below the threshold
            threshold = 0.80
            separate_final_epoch = num_samples_from_final_epoch < int(
                threshold * num_samples_per_epoch
            )

            log_single_rank(
                logger,
                logging.DEBUG,
                f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
            )
            log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
            log_single_rank(
                logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
            )

        log_single_rank(
            logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
        )

        numpy_random_state = numpy.random.RandomState(self.config.random_seed)

        # Build the document index
        document_index = _build_document_index(
            self.indices, num_epochs, numpy_random_state, separate_final_epoch
        )
        drop_last_partial_sequence = True
        if self.index_split == Split.valid:
            drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

        # Build the sample index
        from megatron.core.datasets import helpers

        if self.index_split == Split.valid:
            drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
        else:
            drop_last_partial_sequence = True

        assert document_index.dtype == numpy.int32
        assert self.dataset.sequence_lengths.dtype == numpy.int32
        if len(document_index) * 2 > len(self.dataset.sequence_lengths):
            # Heuristic: if "access density" of sequence_lengths is relatively high,
            # force loading the mmap-ed array into memory by taking a copy.
            # System performance benefits come from two aspects:
            # 1. **sequentially** pre-loading the whole file if we're gonna read a large fraction anyways.
            # 2. GIL is held when calling into c++ code; making the c++ func faster improves parallelism.
            sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
        else:
            sequence_lengths_for_cpp = self.dataset.sequence_lengths
        sample_index = helpers.build_sample_idx(
            sequence_lengths_for_cpp,
            document_index,
            sequence_length,
            num_epochs,
            num_tokens_per_epoch,
            drop_last_partial_sequence,
            self.config.add_extra_token_to_sequence,
        )

        if any(sample_index[:, 0] < 0):
            _url = "https://gitee.com/ascend/MindSpeed-LLM/wikis/megatron%20data%20helpers%E5%8F%AF%E8%83%BD%E5%BC%95%E5%85%A5%E7%9A%84%E9%97%AE%E9%A2%98"
            raise GPTDatasetSampleIndexError(f"Bad sample index. Visit {_url} for more information")

        # Build the shuffle index
        if separate_final_epoch:
            shuffle_index = _build_shuffle_index(
                num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
            )
        else:
            shuffle_index = _build_shuffle_index(
                sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
            )

        if path_to_cache:
            os.makedirs(path_to_cache, exist_ok=True)
            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
        else:
            log_single_rank(
                logger,
                logging.WARNING,
                f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
            )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )
        log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

        return document_index, sample_index, shuffle_index

    log_single_rank(
        logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
    )

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
    )
    t_beg = time.time()
    document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
    )
    t_beg = time.time()
    sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')

    if any(sample_index[:, 0] < 0):
        _url = "https://gitee.com/ascend/MindSpeed-LLM/wikis/megatron%20data%20helpers%E5%8F%AF%E8%83%BD%E5%BC%95%E5%85%A5%E7%9A%84%E9%97%AE%E9%A2%98"
        raise GPTDatasetSampleIndexError(f"Bad sample index. Visit {_url} for more information")

    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
    )
    t_beg = time.time()
    shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
    )

    return document_index, sample_index, shuffle_index

