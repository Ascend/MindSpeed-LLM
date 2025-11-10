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
    DATA_EXTS = (IDX_EXT, BIN_EXT)

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

    should_convert = (rank == 0) if shared else (local_rank == 0)

    if should_convert:
        # Only the designated rank performs the dataset conversion
        for p in paths:
            # Clean input path (remove quotes and spaces)
            p = p.strip().strip('"').strip("'")
            is_file, is_dir = os.path.isfile(p), os.path.isdir(p)

            # Collect candidate data files (single file or all files in a directory)
            data_files = [p] if is_file else glob.glob(os.path.join(p, "*")) if is_dir else []
            ext_detected, fmt = _get_data_format(data_files) if data_files else (None, None)
            is_raw_input = (fmt is not None) and (is_file or is_dir)

            # Determine the dataset prefix for output (.bin/.idx files) ===
            if is_file and ext_detected:
                # Example: /data/train.jsonl -> /data/train
                suffix = "." + ext_detected
                prefix = p[:-len(suffix)] if p.endswith(suffix) else os.path.splitext(p)[0]
            elif is_dir:
                # Example: /data/raw/ -> /data/raw/raw
                prefix = os.path.join(p, os.path.basename(os.path.normpath(p)))
            else:
                # If it's already a converted prefix (.idx/.bin)
                prefix = p[:-4] if any(p.endswith(ext) for ext in DATA_EXTS) else p

            # Ensure output directory exists
            os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

            # Execute dataset conversion ===
            if is_raw_input:
                print_rank_0(f"[DataConvert] Running data conversion: {p} -> {prefix}")
                cmd = [
                    sys.executable, os.path.abspath("preprocess_data.py"),
                    "--input", p,
                    "--tokenizer-name-or-path", args.tokenizer_name_or_path,
                    "--tokenizer-type", args.tokenizer_type,
                    "--handler-name", args.handler_name,
                    "--output-prefix", prefix,
                    "--workers", str(getattr(args, "workers", 1)),
                    "--log-interval", "1000",
                    "--n-subs", str(getattr(args, "n_subs", 1)),
                ]

                # Add json key arguments
                cmd += ["--json-keys"] + list(args.json_keys)

                # Add optional arguments if applicable
                if getattr(args, "pack", False):
                    cmd.append("--pack")
                if getattr(args, "append_eod", False):
                    cmd.append("--append-eod")
                if getattr(args, "stage", False):
                    if getattr(args, "enable_thinking", None) is not None:
                        cmd += ["--enable-thinking", str(args.enable_thinking)]
                    if getattr(args, "prompt_type", None):
                        cmd += ["--prompt-type", args.prompt_type]
                    if getattr(args, "seq_length", None):
                        cmd += ["--seq-length", str(args.seq_length)]

                # Run the subprocess to perform data preprocessing
                subprocess.run(cmd, check=True)
            else:
                # Skip conversion if already a prefix (not raw data)
                print_rank_0(f"[DataConvert][Skip] {p} is not a raw input, treated as prefix.")

    # Synchronize all distributed ranks to ensure consistency ===
    if dist.is_initialized():
        dist.barrier()

    # Update args.data_path with the actual dataset prefixes ===
    new_paths = []
    for raw_path in paths:
        q = raw_path.strip().strip('"').strip("'")

        if os.path.isfile(q):
            # Derive the prefix name based on detected file extension
            ext_detected, _fmt = _get_data_format([q])
            base = q[:-len("." + ext_detected)] if ext_detected and q.endswith("." + ext_detected) else os.path.splitext(q)[0]
        else:
            # For directory or prefix-based inputs
            dirn = os.path.dirname(q) or "."
            name = os.path.basename(q)
            is_prefix = (
                # Check if existing converted files (.idx/.bin or _text_document or _packed)
                any(os.path.exists(q + ext) for ext in DATA_EXTS) or
                any(os.path.exists(q + "_text_document" + ext) for ext in DATA_EXTS) or
                any(f.startswith(name + "_packed") and f.endswith(IDX_EXT) for f in os.listdir(dirn))
            )
            if is_prefix:
                base = q
            elif os.path.isdir(q):
                base = os.path.join(q, os.path.basename(os.path.normpath(q)))
            else:
                base = q[:-4] if any(q.endswith(ext) for ext in DATA_EXTS) else q

        # Locate the correct prefix for training (packed or text_document) ===
        dir_name = os.path.dirname(base) or "."
        prefix_name = os.path.basename(base)
        matched_prefix = None

        for f in os.listdir(dir_name):
            # Fine-tuning stage: match *_packed*.idx/.bin
            if args.stage and f.startswith(prefix_name + "_packed") and f.endswith(IDX_EXT):
                cand = os.path.join(dir_name, f[:-len(IDX_EXT)])
                if os.path.exists(cand + BIN_EXT):
                    matched_prefix = base
                    break
            # Pretraining stage: match *_text_document.idx/.bin
            if not args.stage and (f.startswith(prefix_name + "_text_document") or '_text_document' in f) and f.endswith(IDX_EXT):
                cand = os.path.join(dir_name, f[:-len(IDX_EXT)])
                if os.path.exists(cand + BIN_EXT):
                    matched_prefix = cand
                    break

        # Raise an error if no valid prefix was found
        if not matched_prefix:
            raise FileNotFoundError(
                f"[DataConvert] Training prefix missing: {base}[_text_document or _packed*]{IDX_EXT}/{BIN_EXT}"
            )

        new_paths.append(matched_prefix)

    # Write back the final dataset paths for training
    args.data_path = new_paths if was_list else ",".join(new_paths)
