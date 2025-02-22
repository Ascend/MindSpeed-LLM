import os
import stat
import sys
from pathlib import Path


# =============================
# ST test, run with shell
# =============================

class UT_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')
        self.ut_file = os.path.join(test_dir, "ut")
    
    def run_ut(self):
        command = f"python3.8 -m pytest -k 'not allocator' {self.ut_file}"
        ut_exitcode = os.system(command)
        if ut_exitcode == 0:
            print("UT test success")
        else:
            print("UT failed")
            exit(1)


class ST_Test:
    
    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')

        st_dir = "st"
        llama_pretrain_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_pretrain_ptd.sh")
        llama_inference_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_inference_ptd.sh")
        gemma_pretrain_shell_file = os.path.join(
            test_dir, st_dir, "test_gemma_pretrain_ptd.sh")
        gemma_inference_shell_file = os.path.join(
            test_dir, st_dir, "test_gemma_inference_ptd.sh")
        llama_vpp_pretrain_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_vpp_pretrain_ptd.sh")
        llama_instruction_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_instruction_ptd.sh")

        self.st_file_list = [
            llama_pretrain_shell_file,
            llama_inference_shell_file,
            gemma_pretrain_shell_file,
            gemma_inference_shell_file,
            llama_vpp_pretrain_shell_file,
            llama_instruction_shell_file
        ]

    def run_st(self):
        all_success = True
        for shell_file in self.st_file_list:
            command = f"sh {shell_file}"
            st_exitcode = os.system(command)
            if st_exitcode != 0:
                all_success = False
                print(f"ST run {shell_file} failed")
                break

        if all_success:
            print("ST test success")
        else:
            print("ST failed")
            exit(1)


# ===============================================
# UT test, run with pytest, waiting for more ...
# ===============================================


if __name__ == "__main__":
    pass
    # ut = UT_Test()
    # ut.run_ut()
    # st = ST_Test()
    # st.run_st()
