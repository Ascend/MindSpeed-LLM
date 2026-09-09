from megatron.core.optimizer.optimizer import (
    MixedPrecisionOptimizer,
    ChainedOptimizer,
)


def _fixed_reload_model_params(self, state_dict=None):
    if self.param_groups:
        self._copy_model_params_to_main_params()


def _fixed_chained_reload_model_params(self, state_dict=None):
    state_dicts = self._split_state_dict(state_dict)
    for idx, optimizer in enumerate(self.chained_optimizers):
        optimizer.reload_model_params()


MixedPrecisionOptimizer.reload_model_params = _fixed_reload_model_params
ChainedOptimizer.reload_model_params = _fixed_chained_reload_model_params