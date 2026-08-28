import fnmatch
import inspect
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import nn


logger = logging.getLogger(__name__)

_SUPPORTED_STATISTICS = {"abs_sum", "abs_mean", "max", "min"}
_SUPPORTED_OUTPUT_FORMATS = {"text", "jsonl", "both"}
_DEFAULT_BATCH_FIELDS = ["input_ids", "tokens", "labels", "attention_mask", "position_ids"]
_DEFAULT_CONFIG_PATH = Path(__file__).with_name("model_io_trace_config.json")


def _parse_steps(values: Any) -> Optional[Set[int]]:
    if not isinstance(values, list) or not values:
        raise ValueError("`steps` must be a non-empty list.")

    steps = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"Invalid trace step: {value}")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"Trace step must be >= 0: {value}")
            steps.add(value)
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid trace step: {value}")
        if value.lower() == "all":
            return None
        parts = value.split("-", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Step range must use `start-end`: {value}")
        try:
            start, end = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Invalid trace step range: {value}") from exc
        if start < 0 or end < start:
            raise ValueError(f"Invalid trace step range: {value}")
        steps.update(range(start, end + 1))
    return steps


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def _as_model_list(model: Any) -> List[nn.Module]:
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    if not models or not all(isinstance(item, nn.Module) for item in models):
        raise TypeError("Model I/O tracing expects a torch.nn.Module or a non-empty list of modules.")
    return models


def _walk_tensors(value: Any, path: str) -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{path}.{index}")


def _json_scalar(value: float) -> Any:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _tensor_statistics(tensor: torch.Tensor, names: List[str]) -> Dict[str, Any]:
    if tensor.numel() == 0:
        return {name: None for name in names}
    if not names:
        return {}

    absolute = tensor.detach().abs().float()
    operations = {
        "abs_sum": absolute.sum,
        "abs_mean": absolute.mean,
        "max": absolute.max,
        "min": absolute.min,
    }
    values = torch.stack([operations[name]() for name in names]).cpu().tolist()
    return {name: _json_scalar(value) for name, value in zip(names, values)}


def _tensor_preview(tensor: torch.Tensor, max_rows: int, max_tokens: int) -> Any:
    detached = tensor.detach()
    if detached.ndim == 0:
        preview = detached
    elif detached.ndim == 1:
        preview = detached[:max_tokens]
    else:
        preview = detached.reshape(-1, detached.shape[-1])[:max_rows, :max_tokens]
    return preview.cpu().tolist()


class ModuleIOTracer:
    """Register forward and full-backward hooks on selected model modules."""

    def __init__(self, manager: "ModelIOTraceManager"):
        self.manager = manager
        self.config = manager.module_config

    def register(self, models: List[nn.Module]) -> List[Any]:
        handles = []
        seen = set()
        for chunk_index, model in enumerate(models):
            prefix = f"model_chunk_{chunk_index}"
            for name, module in model.named_modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                module_name = f"{prefix}.{name}" if name else prefix
                text_name = name if len(models) == 1 else module_name
                if not self._should_trace(module_name, module):
                    continue
                if self.config["forward"]:
                    handles.append(
                        module.register_forward_hook(
                            self._forward_hook(module_name, text_name),
                            with_kwargs=True,
                        )
                    )
                if self.config["backward"]:
                    handles.append(module.register_full_backward_hook(self._backward_hook(module_name, text_name)))
        return handles

    def _should_trace(self, name: str, module: nn.Module) -> bool:
        if self.config["leaf_only"] and any(module.children()):
            return False
        if isinstance(module, nn.Identity) or module.__class__.__name__ == "IdentityOp":
            return False
        if module.__class__.__name__.startswith("Dropout") and getattr(module, "p", 1) < 1e-6:
            return False

        include = self.config["include"]
        exclude = self.config["exclude"]
        if include and not any(fnmatch.fnmatchcase(name, pattern) for pattern in include):
            return False
        return not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude)

    def _forward_hook(self, module_name: str, text_name: str):
        def hook(module, args, kwargs, output):
            self.manager.write_text_module_io(text_name, "forward", args, output)
            self._capture(module_name, module, "forward", "input", args)
            self._capture(module_name, module, "forward", "kwarg", kwargs)
            self._capture(module_name, module, "forward", "output", output)

        return hook

    def _backward_hook(self, module_name: str, text_name: str):
        def hook(module, grad_input, grad_output):
            self.manager.write_text_module_io(text_name, "backward", grad_input, grad_output)
            self._capture(module_name, module, "backward", "input", grad_input)
            self._capture(module_name, module, "backward", "output", grad_output)

        return hook

    def _capture(self, module_name: str, module: nn.Module, phase: str, slot: str, value: Any):
        if not self.manager.json_output_enabled and self.manager.tensor_mode != "tensor":
            return
        for tensor_path, tensor in _walk_tensors(value, slot):
            record = {
                "step": self.manager.current_step,
                "rank": self.manager.rank,
                "phase": phase,
                "module": module_name,
                "module_type": module.__class__.__name__,
                "slot": tensor_path,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
            }
            if self.manager.json_output_enabled:
                try:
                    record["statistics"] = _tensor_statistics(tensor, self.manager.statistics)
                except Exception as exc:
                    record["statistics_error"] = f"{type(exc).__name__}: {exc}"
            if self.manager.tensor_mode == "tensor":
                try:
                    record["tensor_file"] = self.manager.save_tensor(
                        tensor,
                        phase=phase,
                        module_name=module_name,
                        slot=tensor_path,
                    )
                except Exception as exc:
                    record["tensor_save_error"] = f"{type(exc).__name__}: {exc}"
            self.manager.write_record("module", record)


class BatchInputTracer:
    """Capture configured batch fields at each model-chunk entry."""

    def __init__(self, manager: "ModelIOTraceManager"):
        self.manager = manager
        self.config = manager.batch_config

    def register(self, models: List[nn.Module]) -> List[Any]:
        handles = []
        for chunk_index, model in enumerate(models):
            forward_arg_names = self._forward_arg_names(model)
            handles.append(
                model.register_forward_pre_hook(
                    self._pre_hook(f"model_chunk_{chunk_index}", forward_arg_names),
                    with_kwargs=True,
                )
            )
        return handles

    @staticmethod
    def _forward_arg_names(model: nn.Module) -> List[str]:
        current = model
        visited = set()
        while isinstance(current, nn.Module) and id(current) not in visited:
            visited.add(id(current))
            try:
                parameters = inspect.signature(current.forward).parameters.values()
            except (TypeError, ValueError):
                parameters = ()
            names = [
                parameter.name
                for parameter in parameters
                if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if names:
                return names

            wrapped = getattr(current, "module", None)
            if not isinstance(wrapped, nn.Module):
                wrapped = getattr(current, "model", None)
            if not isinstance(wrapped, nn.Module):
                break
            current = wrapped
        return []

    def _pre_hook(self, module_name: str, forward_arg_names: List[str]):
        def hook(module, args, kwargs):
            candidates = {}
            for index, value in enumerate(args):
                if isinstance(value, dict):
                    candidates.update(value)
                elif index < len(forward_arg_names):
                    candidates[forward_arg_names[index]] = value
            candidates.update(kwargs)

            fields = {}
            missing_fields = []
            for field in self.config["fields"]:
                tensor = candidates.get(field)
                if not isinstance(tensor, torch.Tensor):
                    missing_fields.append(field)
                    continue
                item = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                }
                try:
                    item["preview"] = _tensor_preview(
                        tensor,
                        self.config["max_rows"],
                        self.config["max_tokens"],
                    )
                except Exception as exc:
                    item["preview_error"] = f"{type(exc).__name__}: {exc}"
                fields[field] = item

            self.manager.write_text_batch(fields, missing_fields)
            self.manager.write_record(
                "batch",
                {
                    "step": self.manager.current_step,
                    "rank": self.manager.rank,
                    "module": module_name,
                    "fields": fields,
                    "missing_fields": missing_fields,
                },
            )

        return hook


class ModelIOTraceManager:
    """Collect model module I/O and batch previews for selected ranks and steps."""

    def __init__(
        self,
        enabled: bool = False,
        config_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        self.requested = enabled
        self.rank = _get_rank()
        self.current_step = None
        self._started = False
        self._active = False
        self._handles = []
        self._writers = {}
        self._record_indexes = {"module": 0, "batch": 0}
        self._batch_index = 0
        self._tensor_index = 0
        self._lock = threading.RLock()

        if not enabled:
            return
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("`model_io_trace_output_path` must be specified when model I/O tracing is enabled.")

        config_file = Path(config_path).expanduser() if config_path else _DEFAULT_CONFIG_PATH
        if not config_file.is_file():
            raise FileNotFoundError(f"Model I/O trace config file does not exist: {config_file}")
        with config_file.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Model I/O trace config must be a JSON object.")

        self._load_config(config, output_path)
        logger.info("Model I/O trace initialized with config: %s", config_file)

    @property
    def enabled(self) -> bool:
        return self.requested

    def _load_config(self, config: Dict[str, Any], output_path: str):
        self.output_path = Path(output_path).expanduser()
        self.output_format = config.get("output_format", "text")
        if not isinstance(self.output_format, str) or self.output_format not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError("`output_format` must be `text`, `jsonl`, or `both`.")
        self.json_output_enabled = self.output_format in ("jsonl", "both")

        ranks = config.get("ranks", [0])
        valid_ranks = (
            isinstance(ranks, list)
            and ranks
            and all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks)
        )
        if not valid_ranks:
            raise ValueError("`ranks` must be a non-empty list of integers.")
        if any(rank < -1 for rank in ranks):
            raise ValueError("Trace ranks must be >= 0, or -1 for all ranks.")
        self.ranks = set(ranks)
        self.steps = _parse_steps(config.get("steps", [0]))

        module_config = config.get("module", {})
        tensor_config = config.get("tensor", {})
        batch_config = config.get("batch", {})
        if not all(isinstance(item, dict) for item in (module_config, tensor_config, batch_config)):
            raise ValueError("`module`, `tensor`, and `batch` must be JSON objects.")

        self.module_config = {
            "leaf_only": self._boolean(module_config.get("leaf_only", True), "module.leaf_only"),
            "include": self._string_list(module_config.get("include", []), "module.include"),
            "exclude": self._string_list(module_config.get("exclude", []), "module.exclude"),
            "forward": self._boolean(module_config.get("forward", True), "module.forward"),
            "backward": self._boolean(module_config.get("backward", True), "module.backward"),
        }

        mode = tensor_config.get("mode", "statistics")
        if mode not in ("statistics", "tensor"):
            raise ValueError("`tensor.mode` must be `statistics` or `tensor`.")
        statistics = self._string_list(
            tensor_config.get("statistics", ["abs_sum", "abs_mean", "max", "min"]),
            "tensor.statistics",
        )
        unsupported = set(statistics) - _SUPPORTED_STATISTICS
        if unsupported:
            raise ValueError(f"Unsupported tensor statistics: {sorted(unsupported)}")
        self.statistics = statistics
        self.tensor_mode = mode

        max_rows = batch_config.get("max_rows", 2)
        max_tokens = batch_config.get("max_tokens", 64)
        rows_valid = isinstance(max_rows, int) and not isinstance(max_rows, bool) and max_rows > 0
        tokens_valid = isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0
        if not rows_valid or not tokens_valid:
            raise ValueError("`batch.max_rows` and `batch.max_tokens` must be positive integers.")
        self.batch_config = {
            "enabled": self._boolean(batch_config.get("enabled", True), "batch.enabled"),
            "fields": self._string_list(batch_config.get("fields", _DEFAULT_BATCH_FIELDS), "batch.fields"),
            "max_rows": max_rows,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _string_list(value: Any, name: str) -> List[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"`{name}` must be a list of non-empty strings.")
        return value

    @staticmethod
    def _boolean(value: Any, name: str) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"`{name}` must be a boolean.")
        return value

    def _matches_current_process(self, step: int) -> bool:
        rank_matches = -1 in self.ranks or self.rank in self.ranks
        step_matches = self.steps is None or step in self.steps
        return rank_matches and step_matches

    def start_step(self, model: Any, step: int):
        if not self.requested:
            return
        if self._started:
            raise RuntimeError("Model I/O trace has already started for the current step.")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("Model I/O trace step must be a non-negative integer.")

        self._started = True
        self.current_step = step
        if not self._matches_current_process(step):
            return

        models = _as_model_list(model)
        step_path = self.output_path / f"rank{self.rank}" / f"step{step}"
        step_path.mkdir(parents=True, exist_ok=True)
        self._step_path = step_path
        self._record_indexes = {"module": 0, "batch": 0}
        self._batch_index = 0
        self._tensor_index = 0
        self._writers = {}
        if self.output_format in ("jsonl", "both"):
            self._writers.update(
                {
                    "module": (step_path / "module_io.jsonl").open("w", encoding="utf-8"),
                    "batch": (step_path / "batch_inputs.jsonl").open("w", encoding="utf-8"),
                }
            )
        if self.output_format in ("text", "both"):
            self._writers.update(
                {
                    "module_text": (step_path / "output.txt").open("w", encoding="utf-8"),
                    "batch_text": (step_path / "token_ids.log").open("w", encoding="utf-8"),
                }
            )
        self._active = True
        try:
            self._handles.extend(ModuleIOTracer(self).register(models))
            if self.batch_config["enabled"]:
                self._handles.extend(BatchInputTracer(self).register(models))
        except Exception:
            self._teardown()
            self._started = False
            self.current_step = None
            raise

    def end_step(self):
        if not self.requested or not self._started:
            return
        self._teardown()
        self._started = False
        self.current_step = None

    def _teardown(self):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception as exc:
                logger.warning("Failed to remove a model I/O trace hook: %s", exc)
        self._handles.clear()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        self._active = False

    def write_record(self, record_type: str, record: Dict[str, Any]):
        if not self._active or record_type not in self._writers:
            return
        with self._lock:
            record["sequence"] = self._record_indexes[record_type]
            self._record_indexes[record_type] += 1
            self._writers[record_type].write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_text_module_io(self, module_name: str, phase: str, inputs: Any, outputs: Any):
        if not self._active or "module_text" not in self._writers:
            return
        with self._lock:
            writer = self._writers["module_text"]
            print("--------------------------input-------------------------------------", file=writer)
            self._write_text_tensor(writer, f"[{phase}]: {module_name} inputs", inputs)
            print("---------------------------output------------------------------------", file=writer)
            self._write_text_tensor(writer, f"[{phase}]: {module_name} outputs", outputs)
            writer.flush()

    @staticmethod
    def _write_text_tensor(writer, name: str, value: Any):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            try:
                tensor_cpu = value.detach().to("cpu", dtype=torch.float32)
            except Exception:
                print("This tensor do not support sum and abs!", file=writer)
                return
            print(name, value.shape, value.dtype, file=writer)
            print(tensor_cpu, file=writer)
            absolute = tensor_cpu.abs()
            operations = (
                ("sum", absolute.sum, "This tensor do not support sum and abs!"),
                ("mean", absolute.mean, "This tensor do not support mean!"),
                ("max", absolute.max, "This tensor do not support max!"),
                ("min", absolute.min, "This tensor do not support min!"),
            )
            for statistic, operation, error_message in operations:
                try:
                    print(f">{statistic}:, {operation().item():e}", file=writer)
                except Exception:
                    print(error_message, file=writer)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (torch.Tensor, tuple, list)):
                    ModelIOTraceManager._write_text_tensor(writer, name, item)
                else:
                    print(name, item, file=writer)
            return
        print(name, type(value), file=writer)
        print(value, file=writer)

    def write_text_batch(self, fields: Dict[str, Any], missing_fields: List[str]):
        if not self._active or "batch_text" not in self._writers:
            return
        with self._lock:
            writer = self._writers["batch_text"]
            print(f"\n=== Batch {self._batch_index} ===", file=writer)
            self._batch_index += 1
            for field in self.batch_config["fields"]:
                if field in missing_fields:
                    print(f"[{field}] Not Found", file=writer)
                    continue
                item = fields[field]
                print(f"[{field}] shape={tuple(item['shape'])}", file=writer)
                if "preview" in item:
                    print(f"[{field}] preview={item['preview']}", file=writer)
                else:
                    print(f"[{field}] preview={item['preview_error']}", file=writer)
            writer.flush()

    def save_tensor(self, tensor: torch.Tensor, phase: str, module_name: str, slot: str) -> str:
        with self._lock:
            tensor_path = self._step_path / "tensors"
            tensor_path.mkdir(exist_ok=True)
            safe_name = "".join(character if character.isalnum() else "_" for character in module_name)
            safe_slot = "".join(character if character.isalnum() else "_" for character in slot)
            filename = f"{self._tensor_index:08d}_{phase}_{safe_name}_{safe_slot}.pt"
            self._tensor_index += 1
            torch.save(tensor.detach().cpu(), tensor_path / filename)
            return str(Path("tensors") / filename)
