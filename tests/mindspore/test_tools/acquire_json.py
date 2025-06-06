# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import re
import json
import enum
import os


class TypeOfTest(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2


def transfer_logs_as_json(log_file, output_json_file):
    """
    Read a log file from the input path, and return the
    summary specified as input as a list

    Args:
        log_file: str, path to the dir where the logs are located.
        output_json_file: str, path of the json file transferred from the logs.

    Returns:
        data: json, the values parsed from the log, formatted as a json file.
    """

    log_pattern = re.compile(
        r"lm loss:\s*(\d+\.\d+)"
    )


    data = {
        "lm loss": []
    }
    with open(log_file, "r") as f:
        log_content = f.read()
    log_matches = log_pattern.findall(log_content)

    if log_matches:
        data["lm loss"] = [float(match) for match in log_matches]

    with open(output_json_file, "w") as outfile:
        json.dump(data, outfile, indent=4)


def read_json(file):
    """
    Read baseline and new generate json file
    """
    if os.path.exists(file):
        with open(file) as f:
            return json.load(f)
    else:
        raise FileExistsError("The file does not exist !")
