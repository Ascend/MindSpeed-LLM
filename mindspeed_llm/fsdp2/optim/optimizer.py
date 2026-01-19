"""
Optimizer factory for unified construction of single optimizer and EP+FSDP2 multi-optimizer, support Adamw now.
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Iterable, Optional, Sequence, Tuple
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


class MultiOptimizer(Optimizer, Stateful):
    def __init__(self, root_model: nn.Module, optimizers: dict, key_names: list[str]):
        self.model = root_model
        self.optimizers_dict = optimizers
        self._is_multi_optimizer: bool = True
        self.key_names = key_names

    def step(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.step()

    def zero_grad(self) -> None:
        for opt in self.optimizers_dict.values():
            opt.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            sd = get_optimizer_state_dict(
                self.model, opt, options=StateDictOptions(flatten_optimizer_state_dict=True)
            )
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(f"Key clash detected for optimizer '{name}': {', '.join(sorted(overlap))}")
            merged.update(sd)
        return merged

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for name in self.key_names:
            opt = self.optimizers_dict.get(name)
            set_optimizer_state_dict(
                self.model,
                opt,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def __len__(self) -> int:
        return len(self.optimizers_dict)


class OptimizerFactory:
    """Optimizer factory class for unified construction of single optimizer/EP+FSDP2 multi-optimizer"""

    @staticmethod
    def create(
            model: torch.nn.Module,
            data_parallel_mode: str,
            ep_size: int,
            lr: float,
            optimizer_type: str,
            weight_decay: float,
            betas: Tuple[float, float],
            adam_epsilon: float,
            fused: bool = False,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            no_decay_modules: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> torch.optim.Optimizer:
        """
        Build optimizer instance

        Args:
            model: Model to be optimized
            data_parallel_mode: Data parallel mode
            ep_size: Expert parallel size
            lr: Base learning rate
            optimizer_type: Type of optimizer
            weight_decay: Weight decay coefficient
            betas: Tuple of AdamW beta1 and beta2 parameters
            adam_epsilon: AdamW epsilon parameter
            fused: Whether to enable fused optimizer kernel
            param_groups: Custom parameter groups
            no_decay_modules: List of module names that do not require weight decay
            no_decay_params: List of parameter names that do not require weight decay
        """
        # Multi-optimizer (EP+FSDP2) Processing
        if data_parallel_mode == "fsdp2" and ep_size > 1:
            logger.info_rank0("Building EP+FSDP2 optimizer (MultiOptimizer)")
            return OptimizerFactory._build_ep_fsdp2_optimizer(
                model=model,
                lr=lr,
                betas=betas,
                eps=adam_epsilon,
                weight_decay=weight_decay,
                fused=fused,
                optimizer_type=optimizer_type.lower(),
                param_groups=param_groups,
                no_decay_modules=no_decay_modules,
                no_decay_params=no_decay_params,
            )

        # Single Optimizer Processing
        single_optimizer = OptimizerFactory._create_single_optimizer(
            model=model,
            lr=lr,
            optimizer_type=optimizer_type,
            weight_decay=weight_decay,
            betas=betas,
            adam_epsilon=adam_epsilon,
            fused=fused,
            param_groups=param_groups,
            no_decay_modules=no_decay_modules,
            no_decay_params=no_decay_params,
        )
        return single_optimizer

    @staticmethod
    def _create_single_optimizer(model, lr, optimizer_type, weight_decay, betas, adam_epsilon, fused, param_groups, no_decay_modules, no_decay_params):
        """
        Create single optimizer instance.
        """
        # Build parameter groups for single optimizer, automatically split parameters that need/don't need decay
        # when no custom parameter groups are provided
        if param_groups is None:
            decay_param_names = OptimizerFactory._get_parameter_names(
                model, no_decay_modules, no_decay_params
            )
            # Build parameter group for parameters that need weight decay
            param_groups = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if n in decay_param_names and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                }
            ]
            # Collect parameters that do not need weight decay
            no_decay_parameters, no_decay_param_names = [], []
            for n, p in model.named_parameters():
                if n not in decay_param_names and p.requires_grad:
                    no_decay_param_names.append(n)
                    no_decay_parameters.append(p)
            # Build parameter group for parameters that do not need weight decay
            if no_decay_parameters:
                logger.debug_rank0(f"Parameters without weight decay: {no_decay_param_names}")
                param_groups.append({
                    "params": no_decay_parameters,
                    "weight_decay": 0.0
                })

        optimizer_type = optimizer_type.lower()
        # The foreach mode organizes all parameters into a list and performs the update logic via batch traversal.
        # The fused mode fuses the multi-step computations of parameter updates into a single kernel for one-time execution.
        # They are mutually exclusive.
        foreach = not fused
        # Call unified method to instantiate single optimizer
        optimizer = OptimizerFactory._create_optimizer_instance(
            optimizer_type=optimizer_type,
            param_groups=param_groups,
            lr=lr,
            betas=betas,
            eps=adam_epsilon,
            weight_decay=weight_decay,
            fused=fused,
            foreach=foreach
        )

        logger.info_rank0(f"Created single optimizer {optimizer_type} | lr={lr}, weight_decay={weight_decay}")
        return optimizer


    @staticmethod
    def _create_optimizer_instance(
            optimizer_type: str,
            param_groups: Sequence[Dict[str, Any]],
            lr: float,
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            fused: bool = False,
            foreach: bool = False
    ) -> Optimizer:
        """
        Unified optimizer instantiation entry point, add branches here for newly extended optimizers
        """
        if optimizer_type == "adamw":
            return AdamW(
                param_groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=fused,
                foreach=foreach
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}, supported types: [adamw]")


    @staticmethod
    def _get_parameter_names(model, forbidden_layer_types, forbidden_param_names):
        forbidden_layer_types = forbidden_layer_types or []
        forbidden_param_names = forbidden_param_names or []
        result = []
        # Traverse submodules
        for name, child in model.named_children():
            child_params = OptimizerFactory._get_parameter_names(child, forbidden_layer_types, forbidden_param_names)
            result += [
                f"{name}.{n}" for n in child_params
                if child.__class__.__name__ not in forbidden_layer_types
                and not any(f in f"{name}.{n}".lower() for f in forbidden_param_names)
            ]
        # Traverse parameters directly owned by current module
        result += [
            k for k in model._parameters.keys()
            if not any(f in k.lower() for f in forbidden_param_names)
        ]
        return result


    @staticmethod
    def _make_param_groups_for_subset(
        model: nn.Module,
        params: Iterable[torch.nn.Parameter],
        weight_decay: float,
        no_decay_modules: Optional[List[str]],
        no_decay_params: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        decay_param_names = set(OptimizerFactory._get_parameter_names(model, no_decay_modules, no_decay_params))
        name_by_param = {p: n for n, p in model.named_parameters()}
        # Filter trainable parameters
        params = [p for p in params if p.requires_grad]
        # Split parameters that need/don't need decay
        decayed = [p for p in params if name_by_param.get(p) in decay_param_names]
        undecayed = [p for p in params if name_by_param.get(p) not in decay_param_names]
        # Build parameter groups
        groups = []
        if decayed:
            groups.append({"params": decayed, "weight_decay": weight_decay})
        if undecayed:
            groups.append({"params": undecayed, "weight_decay": 0.0})
        return groups


    @staticmethod
    def _build_ep_fsdp2_optimizer(
        model: nn.Module,
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        fused: bool,
        optimizer_type: str,
        param_groups: Optional[List[Dict[str, Any]]],
        no_decay_modules: Optional[List[str]],
        no_decay_params: Optional[List[str]],
    ) -> MultiOptimizer:
        ep_groups: List[Dict[str, Any]] = []
        non_ep_groups: List[Dict[str, Any]] = []

        # Process custom parameter groups
        if param_groups is not None:
            if not isinstance(param_groups, list):
                raise ValueError("param_groups must be a list")
            for group_config in param_groups:
                if "params" not in group_config:
                    raise ValueError(f"Group missing 'params' key: {group_config}")
                # Extract group configuration
                group_lr = group_config.get("lr", lr)
                group_params = group_config["params"]
                # Split EP/non-EP parameters
                group_ep_params, group_non_ep_params = [], []
                for p in group_params:
                    if not p.requires_grad:
                        continue
                    # Determine whether it is an EP parameter: it is a DTensor and its device_mesh contains the "efsdp" dimension.
                    if DTensor is not None and isinstance(p, DTensor):
                        mesh = getattr(p, "device_mesh", None)
                        names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                        if "efsdp" in names:
                            group_ep_params.append(p)
                            continue
                    group_non_ep_params.append(p)
                # Build weight decay subgroups for EP parameters
                if group_ep_params:
                    ep_subgroups = OptimizerFactory._make_param_groups_for_subset(
                        model, group_ep_params, weight_decay, no_decay_modules, no_decay_params
                    )
                    for subgroup in ep_subgroups:
                        subgroup["lr"] = group_lr
                        # Preserve custom hyperparameters
                        for k, v in group_config.items():
                            if k not in ["params", "lr", "weight_decay"]:
                                subgroup[k] = v
                    ep_groups.extend(ep_subgroups)
                # Build weight decay subgroups for non-EP parameters
                if group_non_ep_params:
                    non_ep_subgroups = OptimizerFactory._make_param_groups_for_subset(
                        model, group_non_ep_params, weight_decay, no_decay_modules, no_decay_params
                    )
                    for subgroup in non_ep_subgroups:
                        subgroup["lr"] = group_lr
                        for k, v in group_config.items():
                            if k not in ["params", "lr", "weight_decay"]:
                                subgroup[k] = v
                    non_ep_groups.extend(non_ep_subgroups)
        # No custom parameter groups: traverse all model parameters and split
        else:
            ep_params, non_ep_params = [], []
            for _, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if DTensor is not None and isinstance(p, DTensor):
                    mesh = getattr(p, "device_mesh", None)
                    names = getattr(mesh, "mesh_dim_names", []) if mesh is not None else []
                    if "efsdp" in names:
                        ep_params.append(p)
                        continue
                non_ep_params.append(p)
            # Build weight decay groups
            ep_groups = OptimizerFactory._make_param_groups_for_subset(
                model, ep_params, weight_decay, no_decay_modules, no_decay_params
            )
            non_ep_groups = OptimizerFactory._make_param_groups_for_subset(
                model, non_ep_params, weight_decay, no_decay_modules, no_decay_params
            )

        # Internal function to build optimizer
        def _build_optimizer(groups: Sequence[Dict[str, Any]]) -> Optimizer:
            # Multiple optimizers do not support the foreach/fused modes in NPU.
            foreach = False if torch.npu.is_available() else (not fused)
            fused_ = False if torch.npu.is_available() else fused
            return OptimizerFactory._create_optimizer_instance(
                optimizer_type=optimizer_type,
                param_groups=groups,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=fused_,
                foreach=foreach
            )

        # Build EP/non-EP optimizer dictionary
        optimizer_dict = {}
        if ep_groups:
            optimizer_dict["ep"] = _build_optimizer(ep_groups)
        if non_ep_groups:
            optimizer_dict["non_ep"] = _build_optimizer(non_ep_groups)

        # Cache EP/non-EP parameter groups to model
        model._ep_param_groups = {
            "ep": [p for g in ep_groups for p in g.get("params", [])] if ep_groups else [],
            "non_ep": [p for g in non_ep_groups for p in g.get("params", [])] if non_ep_groups else [],
        }

        # Wrap as MultiOptimizer
        multi_opt = MultiOptimizer(
            root_model=model,
            optimizers=optimizer_dict,
            key_names=list(optimizer_dict.keys())
        )

        logger.info_rank0(f"Created EP+FSDP2 MultiOptimizer {optimizer_type} | lr={lr}, weight_decay={weight_decay}")
        return multi_opt
