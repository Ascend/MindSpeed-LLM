import sys
import os
import torch
import torch_npu
from utils import ParamConfig, assert_judge
from ascendspeed import megatron_adaptor
import modellink
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.training import setup_model_and_optimizer, build_train_valid_test_data_iterators
from tests.pipeline.common import DistributedTest


class TestTraining(DistributedTest):
    world_size = 8

    def init(self, config=ParamConfig):
        sys.argv = [sys.argv[0]] + config.network_size + config.auxiliary_param \
                   + config.regularization + config.learning_rate_param \
                   + config.training_aux + config.distributed_param + config.training_param
        from megatron.initialize import initialize_megatron
        os.environ.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
        initialize_megatron(extra_args_provider=None,
                            args_defaults={'no_load_rng': True,
                                           'no_load_optim': True})
        from megatron import get_args
        self.args = get_args()

    def test_training(self):
        self.init(config=ParamConfig)
        torch.npu.set_compile_mode(jit_compile=True)
        from pretrain_gpt import model_provider, forward_step
        from pretrain_gpt import train_valid_test_datasets_provider
        from megatron.global_vars import update_num_microbatches, get_num_microbatches, get_timers
        from megatron.training import train_step, training_log, save_checkpoint_and_time
        from megatron.core import mpu
        model, optimizer, lr_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder)

        config = get_model_config(model[0])
        train_valid_test_datasets_provider.is_distributed = True
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
            train_valid_test_datasets_provider
        )
        if self.args.eval_iters == 0:
            assert_judge(valid_data_iterator is None)
            assert_judge(test_data_iterator is None)

        for model_module in model:
            model_module.train()

        timers = get_timers()
        total_loss_dict = {}
        iteration = self.args.iteration
        config.grad_scale_func = optimizer.scale_loss
        config.timers = timers
        report_memory_flag = True
        timers('interval-time', log_level=0).start(barrier=True)
        saved_checkpoint = False
        while iteration < self.args.train_iters:
            update_num_microbatches(self.args.consumed_train_samples)
            self.args.curr_iteration = iteration
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                train_step(forward_step,
                           train_data_iterator,
                           model,
                           optimizer,
                           lr_scheduler,
                           config)
            iteration += 1
            self.args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                                self.args.micro_batch_size * \
                                                get_num_microbatches()
            loss_scale = optimizer.get_loss_scale().item()
            params_norm = None
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                              optimizer.param_groups[0]['lr'],
                                              iteration, loss_scale,
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad)

            if self.args.save and self.args.save_interval and \
                    iteration % self.args.save_interval == 0:
                save_checkpoint_and_time(iteration, model, optimizer, lr_scheduler)
                saved_checkpoint = True
                break

        if saved_checkpoint:
            for file_name in os.listdir(self.args.save):
                file_path = os.path.join(self.args.save, file_name)
                if os.path.isfile(file_path):
                    assert_judge(file_path.endswith(".txt"))
                else:
                    assert_judge(len(os.listdir(file_path)) == self.args.tensor_model_parallel_size)

    def test_breakpoint_renewal_training(self):
        self.init(config=ParamConfig)
        self.args.load = self.args.save
        torch.npu.set_compile_mode(jit_compile=True)
        from pretrain_gpt import model_provider, forward_step
        from pretrain_gpt import train_valid_test_datasets_provider
        from megatron.global_vars import update_num_microbatches, get_timers
        from megatron.training import train_step
        if self.args.load == self.args.save:  # We can regard it as Breakpoint Renewal Training situation
            model, optimizer, lr_scheduler = setup_model_and_optimizer(
                model_provider, ModelType.encoder_or_decoder)
            config = get_model_config(model[0])
            train_valid_test_datasets_provider.is_distributed = True
            train_data_iterator, valid_data_iterator, test_data_iterator \
                = build_train_valid_test_data_iterators(
                train_valid_test_datasets_provider
            )
            if self.args.eval_iters == 0:
                assert_judge(valid_data_iterator is None)
                assert_judge(test_data_iterator is None)

            for model_module in model:
                model_module.train()

            timers = get_timers()
            iteration = self.args.iteration
            if torch.distributed.get_rank() == 0:
                print(f"iteration:{iteration}")
            assert_judge(iteration == 10)
            config.grad_scale_func = optimizer.scale_loss
            config.timers = timers
            timers('interval-time', log_level=0).start(barrier=True)

            if iteration < self.args.train_iters:
                update_num_microbatches(self.args.consumed_train_samples)
                self.args.curr_iteration = iteration
                loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                    train_step(forward_step,
                               train_data_iterator,
                               model,
                               optimizer,
                               lr_scheduler,
                               config)
                if 'lm loss' in loss_dict.keys():
                    if torch.distributed.get_rank() == 0:
                        print(f"loss:{loss_dict['lm loss']}")
                    assert_judge(abs(loss_dict['lm loss'] - 0.97) < 0.1)
