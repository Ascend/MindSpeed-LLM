import os
from pathlib import Path
import pytest
import pandas as pd
from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import build_args, create_testconfig, compare_file_md5_same
from preprocess_data import main


class TestProcessInstructionDataMerge:
    """
        The instruction dataset is divided into two parts, 
        individual processing results as well as results from the merge instruction dataset.
        The three designed test cases are as follows: 
        1. processing of the first segment of the split instruction dataset
        2. processing of the second segment of the split instruction dataset
        3. merging the two segments and processing them together.
    """

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("full_params, params, merge_params, slice_range", 
        [
            (test_config["instruction_dataset"][0], test_config["test_instruction_datasets_part1"][0], test_config["test_merge_instrction_datasets"][0], slice(0, 25000)),
            (test_config["instruction_dataset"][0], test_config["test_instruction_datasets_part2"][0], test_config["test_merge_instrction_datasets"][0], slice(25000, None))
        ])
    def test_instruction_datasets(self, build_args, full_params, params, merge_params, slice_range):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-part"]):
            os.makedirs(full_params["test-out-part"])

        # read and split dataset
        df = pd.read_parquet(full_params["input-dataset"])
        df.iloc[slice_range, :].to_parquet(params["input"])

        # process instruction datasets
        if slice_range == slice(0, 25000):
            print("\n=============== preprocess instruction datasets part1 =============")
        elif slice_range == slice(25000, None):
            print("\n=============== preprocess instruction datasets part2 =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [merge_params["merge-group-keys"][0], merge_params["merge-group-keys"][1], merge_params["merge-group-keys"][2]]
        end_suffixes = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixes:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-part"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)

    
    @pytest.mark.parametrize("full_params, params", 
        [(test_config["instruction_dataset"][0], test_config["test_merge_instrction_datasets"][0])])
    def test_merge_instruction_datasets(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-merge"]):
            os.makedirs(full_params["test-out-merge"])

        # merge instruction dataset
        print("\n=============== merge instruction datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = [params["merge-group-keys"][0], params["merge-group-keys"][1], params["merge-group-keys"][2]]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-merge"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)


class TestProcessInstructionDataMultiHandler:
    # test config
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    
    @pytest.mark.parametrize("full_params, params", 
        [(test_config["handler_dir"][0], test_config["alpaca_style_instruction_handler"][0])])
    def test_alpaca_style_instruction_handler(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-handler"]):
            os.makedirs(full_params["test-out-handler"])
        
        # process instruction dataset
        print("\n=============== alpaca_style instruction datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["packed_attention_mask_document", "packed_input_ids_document", "packed_labels_document"]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-handler"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)

    @pytest.mark.parametrize("full_params, params", 
        [(test_config["handler_dir"][0], test_config["alpaca_style_pack_instruction_handler"][0])])
    def test_alpaca_style_pack_instruction_handler(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-handler"]):
            os.makedirs(full_params["test-out-handler"])
        
        # process instruction dataset
        print("\n=============== alpaca_style_pack instruction datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["packed_attention_mask_document", "packed_input_ids_document", "packed_labels_document"]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-handler"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)


    @pytest.mark.parametrize("full_params, params", 
        [(test_config["handler_dir"][0], test_config["sharegpt_style_instruction_handler"][0])])
    def test_sharegpt_style_instruction_handler(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-handler"]):
            os.makedirs(full_params["test-out-handler"])
        
        # process instruction dataset
        print("\n=============== sharegpt_style instruction datasets =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["packed_attention_mask_document", "packed_input_ids_document", "packed_labels_document"]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-handler"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)


class TestProcessInstructionDataTemplate:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("full_params, params", 
        [(test_config["template_dir"][0], test_config["reasoning_template"][0])])
    def test_reasoning_template(self, build_args, full_params, params):
        # create output dir if it doesn't exist
        if not os.path.isdir(full_params["test-out-template"]):
            os.makedirs(full_params["test-out-template"])
        
        # process instruction dataset
        print("\n=============== reasoning template test =============")
        main()

        # compare file MD5 hashes
        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["packed_attention_mask_document", "packed_input_ids_document", "packed_labels_document"]
        end_suffixs = [".bin", ".idx"]
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = "_" + mid_str + end_suffix
                base_file = full_params["base-out-template"] + prefix_str + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)