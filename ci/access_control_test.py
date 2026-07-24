import os
import sys
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("examples/") or file.startswith("tests/poc/")


def is_poc(file):
    return file.startswith("tests/poc")


def is_0day(file):
    return file.startswith("tests/0day")


def is_pipecase(file):
    return is_pipeline_case(file)


def is_markdown(file):
    return file.endswith(".md")


def is_image(file):
    return file.endswith(".jpg") or file.endswith(".png")


def is_txt(file):
    return file.endswith(".txt")


def is_json(file):
    return file.endswith(".json")


def is_pipeline_case(file):
    if file == "tests/pipeline/st/test_pipeline_st.py":
        return False

    return file.startswith("tests/pipeline") and (
        file.endswith(".py")
        or (file.startswith("tests/pipeline/ut/") and file.endswith(".json"))
        or (file.startswith("tests/pipeline/st/") and (file.endswith(".sh") or file.endswith(".yaml")))
        or (file.startswith("tests/pipeline/st/baseline/") and file.endswith(".json"))
    )


def is_owners(file):
    return file.startswith("OWNERS")


def is_license(file):
    return file.startswith("LICENSE")


def is_ut(file):
    return file.startswith("tests/ut")


def is_no_suffix(file):
    return os.path.splitext(file)[1] == ''


def skip_ci(files, skip_conds):
    if not files:
        return False

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
        is_0day,
        is_json,
    ]

    return skip_ci(file_list, skip_conds)


def filter_exec_ut(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return False, None

    file_list = read_files_from_txt(raw_txt_file)
    if not file_list:
        return False, None

    filter_conds = [is_ut, is_markdown]
    for file in file_list:
        if not any(condition(file) for condition in filter_conds):
            return False, None
    return True, file_list


def filter_exec_pipeline(raw_txt_file):
    if not os.path.exists(raw_txt_file):
        return []

    file_list = read_files_from_txt(raw_txt_file)
    return [file for file in file_list if is_pipeline_case(file)]


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
        self.ut_files = os.path.join(self.base_dir, self.test_dir, "ut")

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
            sys.exit(1)


# ===============================================
# ST test, run with sh.
# ===============================================


class STTest:
    def __init__(self):
        self.base_dir = Path(__file__).absolute().parents[1]
        self.test_dir = os.path.join(self.base_dir, 'tests')

        self.st_dir = "st"
        self.pytest_suit = os.path.join(self.test_dir, self.st_dir, "test_st.py")

    def run_st(self):
        rectify_case = f"python -m pytest {self.pytest_suit} -v -x"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            sys.exit(1)


class PipelineTest:
    def __init__(self):
        self.base_dir = Path(__file__).absolute().parents[1]
        self.pipeline_st_case = os.path.join(self.base_dir, "tests", "pipeline", "st", "test_pipeline_st.py")

    def get_pipeline_command(self, file, full_path):
        if file.startswith("tests/pipeline/st/") and file.endswith(".sh"):
            case_name = os.path.splitext(os.path.basename(file))[0]
            return f"python -m pytest {self.pipeline_st_case}::test_st_script[{case_name}] -v -x"

        if file.startswith("tests/pipeline/st/") and file.endswith(".yaml"):
            case_name = os.path.splitext(os.path.basename(file))[0]
            return f"python -m pytest {self.pipeline_st_case}::test_st_script[{case_name}] -v -x"

        if file.startswith("tests/pipeline/st/baseline/") and file.endswith(".json"):
            case_name = os.path.splitext(os.path.basename(file))[0]
            return f"python -m pytest {self.pipeline_st_case}::test_st_script[{case_name}] -v -x"

        if file.startswith("tests/pipeline/ut/") and file.endswith(".json"):
            ut_case = os.path.splitext(full_path)[0] + ".py"
            return f"pytest --log-level=INFO {ut_case}"

        if file.endswith(".py"):
            return f"pytest --log-level=INFO {full_path}"

        return None

    def run_pipeline(self, pipeline_files):
        commands = []
        for file in pipeline_files:
            full_path = os.path.join(self.base_dir, file)
            if not os.path.exists(full_path):
                continue

            command = self.get_pipeline_command(file, full_path)
            if command is None or command in commands:
                continue
            commands.append(command)

        for command in commands:
            code = acquire_exitcode(command)
            if code != 0:
                print(f"pipeline case failed: {command}")
                sys.exit(1)
        return len(commands) > 0


def run_tests(raw_txt_file):
    ut = UTTest()
    st = STTest()
    if filter_exec_ut(raw_txt_file)[0]:
        ut.run_ut(raw_txt_file)
    else:
        st.run_st()
        ut.run_ut()


def main():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    pipeline_files = filter_exec_pipeline(raw_txt_file)
    pipeline_executed = False
    if pipeline_files:
        pipeline_test = PipelineTest()
        pipeline_executed = pipeline_test.run_pipeline(pipeline_files)

    skip_signal = choose_skip_ci(raw_txt_file)
    if skip_signal:
        if pipeline_executed:
            print("Skipping UT/ST")
        else:
            print("Skipping CI")
    else:
        run_tests(raw_txt_file)


if __name__ == "__main__":
    main()
