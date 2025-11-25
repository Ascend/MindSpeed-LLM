import sys
import os
import math
from mindspeed_llm import megatron_adaptor

from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.preprocess.data_handler import build_dataset, get_dataset_handler
from preprocess_data import get_args, build_splitter
from tests.test_tools.utils import judge_expression


class TestProcessInstructionData:

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
