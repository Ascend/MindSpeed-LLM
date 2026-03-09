import os
from pathlib import Path
import pytest

from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import build_args, create_testconfig, compare_file_md5_same
from preprocess_data import main


class TestProcessPairwiseData:

    
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))


    @pytest.mark.parametrize("params, base_path", 
        [
            (test_config["test_alpaca_dataset"][0][0], test_config["test_alpaca_dataset"][0][1]),
            (test_config["test_sharegpt_dataset"][0][0], test_config["test_sharegpt_dataset"][0][1]),
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

        mid_strs = ["_packed_chosen_labels_document", "_packed_chosen_input_ids_document", 
                    "_packed_rejected_labels_document", "_packed_rejected_input_ids_document"]
        end_suffixs = [".bin", ".idx"]

        # loop through mid_strs and end_suffixs, checking file MD5 hashes
        for mid_str in mid_strs:
            for end_suffix in end_suffixs:
                end_str = mid_str + end_suffix
                base_file = base_path + end_str
                test_file = params["output-prefix"] + end_str
                assert compare_file_md5_same(base_file, test_file)
                
