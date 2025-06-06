# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import os
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("examples/") or file.startswith("tests/poc/")


def is_poc(file):
    return file.startswith("tests/poc")


def is_0day(file):
    return file.startswith("tests/0day") or file.startswith("tests/mindspore/0day")


def is_pipecase(file):
    return file.startswith("tests/pipeline")


def is_markdown(file):
    return file.endswith(".md")


def is_image(file):
    return file.endswith(".jpg") or file.endswith(".png")


def is_txt(file):
    return file.endswith(".txt")


def is_owners(file):
    return file.startswith("OWNERS")


def is_license(file):
    return file.startswith("LICENSE")


def is_ut(file):
    return file.startswith("tests/ut")


def is_no_suffix(file):
    return os.path.splitext(file)[1] == ''


def skip_ci(files, skip_conds):
    for file in files:
        if not any(condition(file) for condition in skip_conds):
            return False
    return True


def choose_skip_ci(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return False
    
    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_examples,
        is_pipecase,
        is_markdown,
        is_image,
        is_txt,
        is_owners,
        is_license,
        is_no_suffix,
        is_poc,
        is_0day
    ]

    return skip_ci(file_list, skip_conds)


def filter_exec_ut(raw_txt_file):
    file_list = read_files_from_txt(raw_txt_file)
    filter_conds = [
        is_ut,
        is_markdown
    ]
    for file in file_list:
        if not any(condition(file) for condition in filter_conds):
            return False, None
    return True, file_list


def acquire_exitcode(command):
    exitcode = os.system(command)
    real_code = os.WEXITSTATUS(exitcode)
    return real_code


# =============================
# UT test, run with pytest
# =============================

class UTTest:
    def __init__(self):
        self.base_dir = Path(__file__).absolute().parents[1]
        self.test_dir = os.path.join(self.base_dir, 'tests')
        self.mstest_dir = os.path.join(self.test_dir, 'mindspore')
        self.ut_files = os.path.join(
            self.mstest_dir, 'ut'
        )
    
    def run_ut(self, raw_txt_file=None):
        if raw_txt_file is not None and os.path.exists(raw_txt_file):
            filtered_results = filter_exec_ut(raw_txt_file)

            if filtered_results[0]:
                filtered_files = filtered_results[1]
                full_path = [os.path.join(self.base_dir, file) for file in filtered_files]
                exsit_ut_files = [file for file in full_path if os.path.exists(file) and file.endswith(".py")]
                self.ut_files = " ".join(exsit_ut_files)

        command = f"pytest -x --log-cli-level=INFO {self.ut_files}"
        code = acquire_exitcode(command)
        if code == 0:
            print("UT test success")
        else:
            print("UT failed")
            exit(1)


# ===============================================
# ST test, run with sh.
# ===============================================

class STTest:
    def __init__(self):
        self.base_dir = Path(__file__).absolute().parents[1]
        self.test_dir = os.path.join(self.base_dir, 'tests')
        self.mstest_dir = os.path.join(self.test_dir, 'mindspore')
        self.st_dir = "st"
        self.st_shell = os.path.join(
            self.mstest_dir, self.st_dir, "st_run.sh"
        )

    def run_st(self):
        rectify_case = f"bash {self.st_shell}"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            exit(1)


def run_tests(raw_txt_file):
    st = STTest()
    if filter_exec_ut(raw_txt_file)[0]:
        pass
    else:
        st.run_st()

        
def main():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    skip_signal = choose_skip_ci(raw_txt_file)
    if skip_signal:
        print("Skipping CI")
    else:
        run_tests(raw_txt_file)


if __name__ == "__main__":
    main()
    