import os
import sys
import json
import torch
import subprocess
from megatron.training.utils import print_rank_0
from mindspeed_llm.tasks.preprocess.data_handler import _get_data_format


def _load_output_prefixes(manifest_path):
    """Load and validate the exact bin/idx pairs emitted by preprocess_data.py."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"[DataConvert] Missing output manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    idx_files = manifest.get("idx_files")
    if not isinstance(idx_files, list) or not idx_files:
        raise ValueError(f"[DataConvert] Invalid output manifest: {manifest_path}")

    prefixes = []
    for idx_file in idx_files:
        if not isinstance(idx_file, str) or not idx_file.endswith(".idx"):
            raise ValueError(f"[DataConvert] Invalid index path in manifest: {idx_file!r}")
        prefix = idx_file[: -len(".idx")]
        if not os.path.isfile(idx_file) or not os.path.isfile(prefix + ".bin"):
            raise FileNotFoundError(f"[DataConvert] Missing bin/idx output pair: {prefix}")
        if prefix not in prefixes:
            prefixes.append(prefix)

    return prefixes


def convert_datasets(args, shared: bool):
    was_list = isinstance(args.data_path, (list, tuple))
    paths = (
        [str(p).strip() for p in args.data_path]
        if was_list
        else [p.strip() for p in str(args.data_path).split(",") if p.strip()]
    )
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

    for raw in paths:
        p = raw.strip().strip('"').strip("'")

        if os.path.isfile(p):
            auto_prefix = os.path.splitext(p)[0]
            raw_base = os.path.splitext(os.path.basename(p))[0]
        elif os.path.isdir(p):
            auto_prefix = os.path.join(p, os.path.basename(os.path.normpath(p)))
            raw_base = os.path.basename(os.path.normpath(p))
        else:
            raise FileNotFoundError(f"[DataConvert] Expected raw file/dir but got: {p}")

        if user_out:
            user_prefix = str(user_out).strip().strip('"').strip("'")
            if len(paths) == 1:
                out_prefix = user_prefix
            else:
                out_prefix = f"{user_prefix}_{raw_base}"
        else:
            out_prefix = auto_prefix

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

        out_map[p] = {
            "out_prefix": out_prefix,
            "base": out_prefix,
            # Keep a non-dataset extension so directory inputs do not ingest the
            # manifest as raw JSON on a later conversion.
            "manifest": out_prefix + "_preprocess.manifest",
        }

    # Perform actual conversion only on designated rank
    if should_convert:
        for raw in paths:
            p = raw.strip().strip('"').strip("'")
            meta = out_map[p]
            out_prefix = meta["out_prefix"]
            manifest_path = meta["manifest"]

            # Never consume a manifest left by an earlier or failed conversion.
            if os.path.exists(manifest_path):
                os.remove(manifest_path)

            print_rank_0(f"[DataConvert] Converting: {p} -> {out_prefix}")

            cmd = [
                sys.executable,
                os.path.abspath("preprocess_data.py"),
                "--input",
                p,
                "--tokenizer-type",
                args.tokenizer_type,
                "--handler-name",
                args.handler_name,
                "--output-prefix",
                out_prefix,
                "--output-manifest",
                manifest_path,
                "--workers",
                str(getattr(args, "workers", 1)),
                "--log-interval",
                "1000",
                "--n-subs",
                str(getattr(args, "n_subs", 1)),
            ]
            cmd += ["--json-keys"] + list(args.json_keys)

            if getattr(args, "map_keys", None):
                map_keys = json.dumps(args.map_keys)
                cmd += ["--map-keys", map_keys]

            if getattr(args, "tokenizer_model", False):
                cmd += ["--tokenizer-model", str(args.tokenizer_model)]
            if getattr(args, "tokenizer_name_or_path", False):
                cmd += ["--tokenizer-name-or-path", str(args.tokenizer_name_or_path)]
            if getattr(args, "pack", False):
                cmd.append("--pack")
            if getattr(args, "neat_pack", False):
                cmd.append("--neat-pack")
            if getattr(args, "append_eod", False):
                cmd.append("--append-eod")
            if getattr(args, "split_sentences", False):
                cmd.append("--split-sentences")
            if getattr(args, "keep_newlines", False):
                cmd.append("--keep-newlines")
            if getattr(args, "stage", False):
                if getattr(args, "enable_thinking", None) is not None:
                    cmd += ["--enable-thinking", str(args.enable_thinking)]
                if getattr(args, "prompt_type", None):
                    cmd += ["--prompt-type", args.prompt_type]
                if getattr(args, "seq_length", None):
                    cmd += ["--seq-length", str(args.seq_length)]
                if getattr(args, "reasoning_effort", None):
                    cmd += ["--reasoning-effort", str(args.reasoning_effort)]
                if getattr(args, "drop_thinking", None) is not None:
                    cmd += ["--drop-thinking", str(args.drop_thinking)]

            subprocess.run(cmd, check=True)

    if dist.is_initialized():
        dist.barrier()

    # Consume the exact output list reported by the converter. Directory scans
    # are unsafe because unrelated datasets can use the same key/level suffix.
    new_paths = []
    for raw in paths:
        q = raw.strip().strip('"').strip("'")
        if q not in out_map:
            continue
        meta = out_map[q]
        base = meta["base"]
        output_prefixes = _load_output_prefixes(meta["manifest"])

        if getattr(args, "stage", False):
            # Packed dataset readers receive the common base and discover the
            # handler-specific ``_packed_<key>_document`` pairs themselves.
            new_paths.append(base)
        else:
            new_paths.extend(output_prefixes)

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
