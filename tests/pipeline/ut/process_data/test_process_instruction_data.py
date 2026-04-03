import io
import os
import contextlib
from pathlib import Path
import pytest
import logging
import math
import pandas as pd

from mindspeed_llm import megatron_adaptor
from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.preprocess.data_handler import build_dataset, get_dataset_handler
from tests.test_tools.utils import build_args, create_testconfig, compare_file_md5_same, judge_expression
from preprocess_data import main, get_args, build_splitter


class TestProcessInstructionData:

    
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))


    @pytest.mark.parametrize("params, base_path", 
        [
            (test_config["test_alpaca_dataset"][0], "/data/ci/datasets/tune_dataset/Llamafactoryhandler/alpaca/alpaca"),
            (test_config["test_alpaca_history_dataset"][0], "/data/ci/datasets/tune_dataset/Llamafactoryhandler/alpaca_his/alpaca_his"),
            (test_config["test_sharegpt_dataset"][0], "/data/ci/datasets/tune_dataset/Llamafactoryhandler/sharegpt/sharegpt"),
            (test_config["test_openai_dataset"][0], "/data/ci/datasets/tune_dataset/Llamafactoryhandler/openai/sss")
        ])
    def test_datasets(self, build_args, params, base_path):
        """
        Tests dataset preprocessing and validates output files by comparing MD5 checksums.

        Parameters:
        - params: dict
            A dictionary containing dataset-specific configurations, such as input files,
            output prefix, and tokenizer information. Extracted from `test_config`.
        - base_path: str
            The base path of the reference dataset files (e.g., Alpaca, Alpaca History, ShareGPT, OpenAI).
            Used to locate the ground truth files for comparison with the generated output.
        """
        # create output dir if it doesn't exist
        out_dir = os.path.dirname(params["output-prefix"])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # run the main preprocessing function
        main()

        # print dataset name for clarity
        dataset_name = base_path.split('/')[-1]
        print(f"=============== test_{dataset_name}_dataset =============")

        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["_packed_attention_mask_document", "_packed_input_ids_document", "_packed_labels_document"]
        end_suffixs = [".bin", ".idx"]

        # loop through mid_strs and end_suffixs, checking file MD5 hashes
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = mid_str + end_suffix
                base_file = base_path + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)


    @pytest.mark.parametrize("params, base_path", 
        [
            (test_config["test_alpaca_history_dataset"][1], "/data/ci/datasets/tune_dataset/Llamafactoryhandler/alpaca_his/alpaca_his_seq1024"),
        ])
    def test_skip_num(self, build_args, params, base_path):
        """
        Tests skip_num in preprocessing and validates output files by comparing MD5 checksums.

        Parameters:
        - params: dict
            A dictionary containing dataset-specific configurations, such as input files,
            output prefix, and tokenizer information. Extracted from `test_config`.
        - base_path: str
            The base path of the reference dataset files (e.g., Alpaca, Alpaca History, ShareGPT, OpenAI).
            Used to locate the ground truth files for comparison with the generated output.
        """
        # create output dir if it doesn't exist
        out_dir = os.path.dirname(params["output-prefix"])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        # run the main preprocessing function
        log_capture_string  = io.StringIO()
        # run the main preprocessing function
        log_handler = logging.StreamHandler(log_capture_string)
        log_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(log_handler)
        main()
        output = log_capture_string.getvalue()
        assert("Skip " in output and " sample exceeded seq-length" in output)

        index1 = output.find("Skip ")
        index2 = output.find(" sample exceeded seq-length")
        skip_num = output[index1 + 5: index2]
        assert(skip_num == "796.0")
        logger.removeHandler(log_handler)
        log_capture_string.close()

        # print dataset name for clarity
        dataset_name = base_path.split('/')[-1]
        print(f"=============== test_{dataset_name}_dataset =============")

        prefix_str = params["output-prefix"].split('/')[-1]
        mid_strs = ["_packed_attention_mask_document", "_packed_input_ids_document", "_packed_labels_document"]
        end_suffixs = [".bin", ".idx"]

        # loop through mid_strs and end_suffixs, checking file MD5 hashes
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = mid_str + end_suffix
                base_file = base_path + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)


class TestProcessInstructionPackData:

    def setup_class(self):
        sys.argv = [
            sys.argv[0],
            "--input", "/data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
            "--tokenizer-type", "PretrainedFromHF",
            "--handler-name", "GeneralInstructionHandler",
            "--output-prefix", "/data/ci/datasets/tune_dataset/tune_pack_dataset/alpaca",
            "--tokenizer-name-or-path", "/data/ci/models/llama2/hf/llama-2-7b-hf",
            "--workers", "4",
            "--log-interval", "1000",
            "--append-eod",
            "--pack",
            "--seq-length", "4096"
        ]
        self.args = get_args()
        self.tokenizer = build_tokenizer(self.args)
        self.splitter = build_splitter(self.args)
        self.raw_dataset = build_dataset(self.args)
        self.handler = get_dataset_handler(self.args, self.raw_dataset, self.tokenizer, self.splitter)

    def test_serialize_to_disk(self):
        """
        Test generate pretrain object files and files are not None(MB).
        """
        self.handler.serialize_to_disk()
        folder_path = "/data/ci/datasets/tune_dataset/tune_pack_dataset"
        bin_file = 0
        idx_file = 0
        total_size = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                if file_path.endswith(".bin") and file_name.startswith('alpaca_pack'):
                    bin_file += 1
                    total_size += os.path.getsize(file_path)
                if file_path.endswith(".idx") and file_name.startswith('alpaca_pack'):
                    idx_file += 1
                    total_size += os.path.getsize(file_path)
        judge_expression(bin_file == 3)
        judge_expression(idx_file == 3)
        judge_expression(math.isclose(total_size / (1024 * 1024), 90.67, abs_tol=3))

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