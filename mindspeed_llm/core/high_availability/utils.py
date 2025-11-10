# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
from dataclasses import dataclass


@dataclass
class HighAvailabilityConstant:
    RET_OK = 0
    RET_ERROR = 1
    RET_NO_REBUILD = 2

    TRAIN_PARAM = 0
    TEST_DATA_ITER = 1
    MODEL_INDEX = 1
    OPTIM_INDEX = 2
    SCHEDULER_INDEX = 3
    TRAIN_DATA_INDEX = 4
    VALID_DATA_INDEX = 5
    CONFIG_INDEX = -1

    UCE_LOW_LEVEL = 2
    UCE_HIGH_LEVEL = 3

    DEFAULT_MIN_FILE_SIZE = 1
    DEFAULT_MAX_FILE_SIZE = 1024 * 1024 * 1024


ha_constant = HighAvailabilityConstant()


class FileUtils:
    """
    This is a class with some class methods
    to handle some file path check
    """

    @classmethod
    def check_file_exists(cls, file_path):
        return os.path.exists(file_path)

    @classmethod
    def check_directory_exists(cls, dir_path):
        return os.path.isdir(dir_path)

    @classmethod
    def is_symlink(cls, file_path):
        clean_path = file_path.rstrip('/')
        return os.path.islink(clean_path)

    @classmethod
    def abs_dir(cls, dir_path):
        if not os.path.isabs(dir_path):
            return os.path.abspath(dir_path)
        else:
            return dir_path

    @classmethod
    def regular_file_path(cls, file_path, base_dir="/", allow_symlink=False):
        """
        regular file path;
            1. check is path empty?
            2. file_path length more than 1024?
            3. if not allow symlink; check is symlink?
            4. linux shell: realpath file_path?
            5. path is in base dir[realpath]?

        :param base_dir: base_dir must a realpath; file path must in base dir
        :param file_path: path
        :param allow_symlink: default is False
        :return: check_status[True or False], err_msg[if False], real_file_path[if True]
        """
        if not file_path or not isinstance(file_path, str):
            err_msg = f"The file path: {os.path.basename(file_path)} is empty or not a string type."
            return False, err_msg, None

        if not base_dir or not isinstance(base_dir, str):
            err_msg = f"The base dir path is empty or not a string type."
            return False, err_msg, None

        if len(file_path) > 1024:
            err_msg = f"The file path {os.path.basename(file_path)} exceeds the maximum value."
            return False, err_msg, None

        if not allow_symlink and FileUtils.is_symlink(file_path):
            err_msg = f"The file {os.path.basename(file_path)} is a link."
            return False, err_msg, None

        try:
            real_file_path = os.path.realpath(file_path)
        except Exception as e:
            err_msg = f"Realpath parsing failed for path {os.path.basename(file_path)}: {str(e)}"
            return False, err_msg, None

        base_dir = base_dir if base_dir[-1] == "/" else base_dir + '/'
        if not cls.is_base_dir_path(base_dir, real_file_path):
            err_msg = f'check path failed, path: {os.path.basename(file_path)} invalid, or such as .. in path'
            return False, err_msg, None

        return True, None, real_file_path

    @classmethod
    def is_base_dir_path(cls, base_dir, path):
        abs_path = os.path.abspath(path)
        base_abs_path = os.path.abspath(base_dir)
        return os.path.commonpath([abs_path, base_abs_path]) == base_abs_path

    @classmethod
    def check_file_size(cls, file_path):
        """
        safe check file size

        :param file_path: path
        :return: check status, err_msg[if False]
        """
        # Check if the file exists
        if not FileUtils.check_file_exists(file_path):
            err_msg = f"File: {os.path.basename(file_path)} does not exist!"
            return False, err_msg

        # Get the real_file_path
        flag, err_msg, real_file_path = FileUtils.regular_file_path(file_path)
        if not flag:
            err_msg = f"Regular_file_path failed by: {err_msg}"
            return False, err_msg

        try:
            # Open the file in binary read mode
            with open(real_file_path, "rb") as fp:
                # Seek to the end of the file
                fp.seek(0, os.SEEK_END)
                # Get the file size
                file_size = fp.tell()
            if file_size < ha_constant.DEFAULT_MIN_FILE_SIZE or file_size > ha_constant.DEFAULT_MAX_FILE_SIZE:
                err_msg = f"Read input file {os.path.basename(file_path)} failed, file size is invalid"
                return False, err_msg
            return True, None
        except Exception as e:
            err_msg = f"Error: {str(e)}"
            return False, err_msg

    @classmethod
    def constrain_owner(cls, file_path, check_owner):
        try:
            file_stat = os.stat(file_path)
        except FileNotFoundError:
            err_msg = f"Error: File '{os.path.basename(file_path)}' not found."
            return False, err_msg

        current_user_id = os.getuid()
        file_owner_id = file_stat.st_uid

        if file_owner_id != current_user_id:
            err_msg = f"File '{os.path.basename(file_path)}' owner ID mismatch. Current process user ID: " \
                      f"{current_user_id}, file owner ID: {file_owner_id} "
            if check_owner:
                return False, err_msg
            else:
                return True, err_msg

        return True, None

    @classmethod
    def constrain_permission(cls, file_path, mode, check_permission):
        try:
            file_stat = os.stat(file_path)
        except FileNotFoundError:
            err_msg = f"Error: File '{os.path.basename(file_path)}' not found."
            return False, err_msg

        current_permissions = file_stat.st_mode & 0o777
        required_permissions = mode & 0o777

        for i in range(3):
            cur_perm = (current_permissions >> (i * 3)) & 0o7
            max_perm = (required_permissions >> (i * 3)) & 0o7
            if (cur_perm | max_perm) != max_perm:
                err_msg = f"File: {os.path.basename(file_path)} Check {['Other group', 'Owner group', 'Owner'][i]} " \
                          f"permission failed: Current permission is {cur_perm}, " \
                          f"but required no greater than {max_perm}. "
                if check_permission:
                    return False, err_msg
                else:
                    return True, err_msg
        return True, None

    @classmethod
    def is_file_valid(cls, file_path, mode, check_owner=True, check_permission=True) -> (bool, str):
        if not FileUtils.check_file_exists(file_path):
            err_msg = f"Error: File '{os.path.basename(file_path)}' not found."
            return False, err_msg

        check_flag, err_msg = FileUtils.check_file_size(file_path)
        if not check_flag:
            return False, err_msg

        check_flag, err_msg = FileUtils.constrain_owner(file_path, check_owner)
        if not check_flag:
            return False, err_msg

        check_flag, err_msg = FileUtils.constrain_permission(file_path, mode, check_permission)
        if not check_flag:
            return False, err_msg

        return True, None

    @classmethod
    def is_dir_valid(cls, dir_path, mode) -> (bool, str):
        if not FileUtils.check_directory_exists(dir_path):
            return False, f"Error: Directory '{os.path.basename(dir_path)}' not found."

        check_flag, err_msg = FileUtils.constrain_owner(dir_path, True)
        if not check_flag:
            return False, err_msg

        check_flag, err_msg = FileUtils.constrain_permission(dir_path, mode, True)
        if not check_flag:
            return False, err_msg

        return True, None
