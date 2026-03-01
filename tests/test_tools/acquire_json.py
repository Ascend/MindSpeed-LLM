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
        r"elapsed time per iteration \(ms\):\s+(\d+\.?\d*)\s*\|.*?lm loss:\s+([\d.E+-]+)\s*\|.*?grad norm:\s+(\d+\.?\d*)(?:\s*\|.*?actor/pg_loss\s*:\s*(\d+\.?\d*))?"
    )
    # Combined pattern for matching both memory formats:
    # 1. megatron_format: [Rank X] (after X iterations) memory (MB) | allocated: X | max allocated: X
    # 2. fsdp2_format: iteration X/X ... max_memory_allocated(GB): X.XX ...
    memory_pattern = re.compile(
        r"(?:\[Rank (\d+)\] \(after \d+ iterations\) memory \(MB\) \| allocated: ([0-9.]+) \| max allocated: ([0-9.]+))"
        r"|"
        r"(?:iteration\s+(\d+)/\d+\s*\|.*?max_memory_allocated\(GB\):\s+([0-9.]+))"
    )

    data = {
        "lm loss": [],
        "grad norm": [],
        "time info": [],
        "memo info": [],
        
    }
    with open(log_file, "r") as f:
        log_content = f.read()
    log_matches = log_pattern.findall(log_content)
    memory_matches = memory_pattern.findall(log_content)
    
    if log_matches:
        if log_matches[0][1] != "":
            data["lm loss"] = [float(match[1]) for match in log_matches]
            data["grad norm"] = [float(match[2]) for match in log_matches]
            data["time info"] = [float(match[0]) for match in log_matches]
        else:
            data["lm loss"] = [float(match[3]) for match in log_matches]

    if memory_matches:
        memo_info = []
        
        for match in memory_matches:
            # Check which format matched
            if match[0]:  # megatron_format matched (Rank is present)
                memo_info.append({
                    "rank": int(match[0]),
                    "allocated memory": float(match[1]),
                    "max allocated memory": float(match[2]),
                    "format": "megatron"
                })
            elif match[3]:  # fsdp2_format matched (iteration and max_memory_allocated are present)
                iteration = int(match[3])
                memo_info.append({
                    "iteration": iteration,
                    "max allocated memory": float(match[4]),
                    "format": "fsdp2"
                })
        
        # Sort by rank for megatron_format, by iteration for fsdp2_format
        data["memo info"] = sorted(memo_info, key=lambda x: x.get("rank", 0) or x.get("iteration", 0))

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
