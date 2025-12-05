from functools import wraps
import torch


def communication_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        arguments = get_args()
        if arguments.enable_high_availability:
            from mindspeed_llm.core.high_availability import tft_is_arf_reboot_node
            if tft_is_arf_reboot_node():
                return None
            if arguments.enable_elastic_training:
                group_index = 2
                return torch_wrapper(fn, group_index, *args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def barrier_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        arguments = get_args()
        if arguments.enable_high_availability:
            from mindspeed_llm.core.high_availability import tft_is_arf_reboot_node, tft_get_node_group
            if tft_is_arf_reboot_node():
                node_group = tft_get_node_group()
                return fn(node_group) if node_group is not None else None
            if arguments.enable_elastic_training:
                return torch_wrapper(fn, 0, *args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def new_group_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        backend = kwargs.get('backend', None)
        from mindspeed_llm.core.high_availability import tft_is_arf_reboot_node
        if tft_is_arf_reboot_node() and isinstance(backend, str) and 'gloo' in backend:
            return None

        if (backend is None) or torch.distributed.distributed_c10d._is_barrier_after_init():
            kwargs['use_local_synchronization'] = True
        res = fn(*args, **kwargs)
        return res
    return wrapper


def is_need_change_group(group_index, *args, **kwargs):
    """
    Check whether the 'group' parameter passed in is 'None' to determine if the value of 'group'
    parameter needs to be changed in the scenario of scale-in training, and whether to modify 'args'
    or 'kwargs'.
    """
    if group_index < 0:
        return False, ""
    if len(args) <= group_index and kwargs.get('group', None) is None:
        return True, 'kwargs'
    if len(args) > group_index and args[group_index] is None:
        return True, 'args'
    if len(args) > group_index and args[group_index] == torch.distributed.group.WORLD:
        return True, 'args'
    if kwargs.get('group', None) == torch.distributed.group.WORLD:
        return True, 'kwargs'
    return False, ""


def group_index_two_torch_wrapper(fn):
    """
    In the context of scale-in training scenarios, if the 'group' parameter passed in is 'None',
    change it to the scale-in world group.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        if not get_args().enable_elastic_training:
            return fn(*args, **kwargs)
        group_index = 2
        return torch_wrapper(fn, group_index, *args, **kwargs)
    return wrapper


def group_index_three_torch_wrapper(fn):
    """
    In the context of scale-in training scenarios, if the 'group' parameter passed in is 'None',
    change it to the scale-in world group.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        if not get_args().enable_elastic_training:
            return fn(*args, **kwargs)
        group_index = 3
        return torch_wrapper(fn, group_index, *args, **kwargs)
    return wrapper


def all_to_all_single_wrapper(fn):
    """
    In the context of scale-in training scenarios, if the 'group' parameter passed in is 'None',
    change it to the scale-in world group.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        if not get_args().enable_elastic_training:
            return fn(*args, **kwargs)
        group_index = 4
        return torch_wrapper(fn, group_index, *args, **kwargs)
    return wrapper


def torch_wrapper(fn, group_index, *args, **kwargs):
    """
    In the context of scale-in training scenarios, if the 'group' parameter passed in is 'None',
    change it to the scale-in world group.
    """
    from mindspeed_llm.core.high_availability.tft_arf_group_repair import tft_is_arf_reboot_node
    from mindspeed_llm.core.high_availability import elastic_training_common
    if tft_is_arf_reboot_node():
        return None
    if elastic_training_common.zit_scale_in_running_state():
        need_change_group, change_str = is_need_change_group(group_index, *args, **kwargs)
        if need_change_group and change_str == 'args':
            args_list = list(args)
            args_list[group_index] = elastic_training_common.zit_get_scale_in_world_group()
            new_args = tuple(args_list)
            return fn(*new_args, **kwargs)
        if need_change_group and change_str == 'kwargs':
            kwargs['group'] = elastic_training_common.zit_get_scale_in_world_group()
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)
