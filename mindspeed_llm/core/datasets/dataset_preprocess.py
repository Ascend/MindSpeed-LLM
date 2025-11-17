import os
import sys
import time
import torch
import glob
import os, sys, time, subprocess
from megatron.training.utils import print_rank_0
from mindspeed_llm.tasks.preprocess.data_handler import _get_data_format


def convert_datasets(args, shared: bool):
    IDX_EXT = ".idx"
    BIN_EXT = ".bin"

    was_list = isinstance(args.data_path, (list, tuple))
    paths = [str(p).strip() for p in args.data_path] if was_list else [
        p.strip() for p in str(args.data_path).split(",") if p.strip()
    ]
    if not paths:
        return

    dist = torch.distributed
    rank = dist.get_rank() if dist.is_initialized() else 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = rank % max(1, (torch.cuda.device_count() if torch.cuda.is_available() else 1))

    # Determine which rank performs the actual conversion
    should_convert = (rank == 0) if shared else (local_rank == 0)

    # Build metadata map (output prefix + base prefix)
    out_map = {}
    user_out = getattr(args, "output_prefix", None)
    user_out_is_dir = isinstance(user_out, str) and (
        user_out.endswith("/") or user_out.endswith("\\") or user_out.endswith(os.sep)
    )

    for raw in paths:
        p = raw.strip().strip('"').strip("'")

        if os.path.isfile(p):
            auto_prefix = os.path.splitext(p)[0]
        elif os.path.isdir(p):
            auto_prefix = os.path.join(p, os.path.basename(os.path.normpath(p)))
        else:
            raise FileNotFoundError(f"[DataConvert] Expected raw file/dir but got: {p}")

        # Determine output prefix
        if not user_out:
            out_prefix = auto_prefix
        else:
            if os.path.isdir(user_out) or user_out_is_dir:
                out_prefix = os.path.join(user_out.rstrip("/\\"), os.path.basename(auto_prefix))
            else:
                if len(paths) == 1:
                    out_prefix = user_out
                else:
                    parent = os.path.dirname(user_out) or "."
                    out_prefix = os.path.join(parent, os.path.basename(auto_prefix))

        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

        out_map[p] = {
            "out_prefix": out_prefix,
            "base": out_prefix,
        }

    # Perform actual conversion only on designated rank
    if should_convert:
        for raw in paths:
            p = raw.strip().strip('"').strip("'")
            meta = out_map[p]
            out_prefix = meta["out_prefix"]

            print_rank_0(f"[DataConvert] Converting: {p} -> {out_prefix}")

            cmd = [
                sys.executable, os.path.abspath("preprocess_data.py"),
                "--input", p,
                "--tokenizer-name-or-path", args.tokenizer_name_or_path,
                "--tokenizer-type", args.tokenizer_type,
                "--handler-name", args.handler_name,
                "--output-prefix", out_prefix,
                "--workers", str(getattr(args, "workers", 1)),
                "--log-interval", "1000",
                "--n-subs", str(getattr(args, "n_subs", 1)),
            ]
            cmd += ["--json-keys"] + list(args.json_keys)

            if getattr(args, "pack", False):
                cmd.append("--pack")
            if getattr(args, "neat_pack", False):
                cmd.append("--neat-pack")
            if getattr(args, "append_eod", False):
                cmd.append("--append-eod")
            if getattr(args, "stage", False):
                if getattr(args, "enable_thinking", None) is not None:
                    cmd += ["--enable-thinking", str(args.enable_thinking)]
                if getattr(args, "prompt_type", None):
                    cmd += ["--prompt-type", args.prompt_type]
                if getattr(args, "seq_length", None):
                    cmd += ["--seq-length", str(args.seq_length)]

            subprocess.run(cmd, check=True)

    if dist.is_initialized():
        dist.barrier()

    # After conversion, find actual training prefix (.idx/.bin)
    new_paths = []
    for raw in paths:
        q = raw.strip().strip('"').strip("'")
        if q not in out_map:
            continue
        meta = out_map[q]
        base = meta["base"]

        # Case 1: direct .idx/.bin exists
        if os.path.exists(base + IDX_EXT) and os.path.exists(base + BIN_EXT):
            matched_prefix = base
        else:
            dir_name = os.path.dirname(base) or "."
            prefix_name = os.path.basename(base)
            matched_prefix = None

            # Stage = fine-tuning → search packed format
            if getattr(args, "stage", False):
                for f in os.listdir(dir_name):
                    if f.startswith(prefix_name + "_packed") and f.endswith(IDX_EXT):
                        cand = os.path.join(dir_name, f[:-len(IDX_EXT)])
                        if os.path.exists(cand + BIN_EXT):
                            matched_prefix = base
                            break
            else:
                # Stage = pretraining → search text_document format
                for f in os.listdir(dir_name):
                    if (f.startswith(prefix_name + "_text_document") or 
                        "_text_document" in f) and f.endswith(IDX_EXT):
                        cand = os.path.join(dir_name, f[:-len(IDX_EXT)])
                        if os.path.exists(cand + BIN_EXT):
                            matched_prefix = cand
                            break

        if not matched_prefix:
            raise FileNotFoundError(
                f"[DataConvert] Missing output prefix for training: {base}[*_text_document or *_packed]"
            )

        new_paths.append(matched_prefix)

    args.data_path = new_paths if was_list else ",".join(new_paths)


def _is_raw_data_path(path: str) -> bool:
    """Return True if the path is a raw file/dir recognizable by _get_data_format."""
    p = str(path).strip().strip('"').strip("'")

    if os.path.isfile(p):
        data_files = [p]
    elif os.path.isdir(p):
        data_files = [os.path.join(p, f) for f in os.listdir(p)]
    else:
        return False

    if not data_files:
        return False

    _, fmt = _get_data_format(data_files)
    return fmt is not None