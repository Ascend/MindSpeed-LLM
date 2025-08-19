from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class DatasetFeature(MindSpeedFeature):

    def __init__(self):
        super(DatasetFeature, self).__init__(feature_name="dataset", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument("--no-shuffle", action='store_true',
                            help="Disable data shuffling, mainly for loss comparison.")
        group.add_argument('--neat-pack', action='store_true',
                            help='Use a zigzag attention mask.')
        group.add_argument('--padded-samples', action='store_true',
                            help='fill in the missing samples within an epoch, starting at index 0, aligned with the LlamaFatory.')
        group.add_argument("--enable-thinking", type=lambda x: {"true": True, "false": False, "none": None}[x.lower()], default=None,
                            help="Whether or not to enable thinking mode for reasoning models.")
        group.add_argument('--pad-to-multiple-of', type=int, default=8,
                            help='Used for Padding multiple in finetune. The default is 8.')

    def register_patches(self, patch_manager, args):
        from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
        from megatron.core.datasets.gpt_dataset import GPTDataset
        from mindspeed_llm.core import (build_generic_dataset, _build_document_sample_shuffle_indices,
                                        indexed_dataset_builder_init_wrapper, add_item_wrapper, finalize_wrapper)
        from mindspeed_llm.training.training import build_train_valid_test_data_loaders_wrapper
        from mindspeed_llm.core.datasets.gpt_dataset import (gpt_dataset_getitem_wrapper, 
                                                            _get_ltor_masks_and_position_ids)

        patch_manager.register_patch('megatron.core.datasets.gpt_dataset.GPTDataset._build_document_sample_shuffle_indices',
                                    _build_document_sample_shuffle_indices)
        patch_manager.register_patch('megatron.core.datasets.blended_megatron_dataset_builder.BlendedMegatronDatasetBuilder.build_generic_dataset',
                                    build_generic_dataset)
        patch_manager.register_patch('megatron.training.training.build_train_valid_test_data_loaders',
                                    build_train_valid_test_data_loaders_wrapper)
        patch_manager.register_patch('megatron.core.datasets.gpt_dataset.GPTDataset.__getitem__',
                                    gpt_dataset_getitem_wrapper)
        # data preprocess and finetune dataloader
        patch_manager.register_patch('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.__init__',
                                    indexed_dataset_builder_init_wrapper)
        patch_manager.register_patch('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.add_item',
                                    add_item_wrapper)
        patch_manager.register_patch('megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.finalize',
                                    finalize_wrapper)