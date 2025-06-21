# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import time
import gc
import torch
import torch.nn.functional as F

from megatron.core import mpu, tensor_parallel
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    calc_params_l2_norm,
    check_adlr_autoresume_termination,
    print_rank_0)
from megatron.training.global_vars import (
    get_args,
    get_signal_handler,
    get_timers)
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training.training import (
    print_datetime,
    get_one_logger,
    append_to_progress_log,
    num_floating_point_operations,
    training_log
)
from megatron.inference.text_generation.communication import broadcast_from_last_pipeline_stage

from mindspeed_llm.tasks.posttrain.trl_ppo.TrlPPOEngine import TrlPPOEngine
from mindspeed_llm.training.training import get_profiler, is_profile_enabled
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank, generate_actual_seq_len
from mindspeed_llm.tasks.posttrain.trl_ppo.utils import MiniBatchIterator
from mindspeed_llm.tasks.posttrain.trl_ppo.utils import pad_to_tensor_dict, save_checkpoint_and_time, save_checkpoint

_TRAIN_START_TIME = time.time()


class TrlPPOTrainer():
    def __init__(self, process_non_loss_data_func=None):
        self.trl_ppo_engine = TrlPPOEngine()

    def initialize(self):
        args = get_args()
        self.timers = get_timers()

        if args.log_progress:
            append_to_progress_log("Starting job")

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.float, device='cuda')
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')

        one_logger = get_one_logger()
        if one_logger:
            one_logger.log_metrics({
                'train_iterations_warmup': 5
            })

        self.trl_ppo_engine.initialize()

    def train_step(self, forward_step_func, exp_data,
                   model, optimizer, opt_param_scheduler):
        """Single training step."""
        args = get_args()

        seq_len = exp_data["padded_query_responses"].shape[1]

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=MiniBatchIterator(exp_data, get_num_microbatches(), args.micro_batch_size),
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=seq_len,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Update parameters.
        self.timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        self.timers('optimizer').stop()


        # Update learning rate.
        if update_successful:
            increment = get_num_microbatches() * \
                        args.micro_batch_size * \
                        args.data_parallel_size
            opt_param_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        # Empty unused memory.
        if args.empty_unused_memory_level >= 2:
            torch.cuda.empty_cache()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        numerator += val
                        denominator += 1
                loss_reduced[key] = numerator / denominator

            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def update_log_metrics(self):
        args = get_args()
        batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        args.consumed_train_samples += batch_size
        self.num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

    def step_training_log(self, model, optimizer, loss_dict, skipped_iter, grad_norm, num_zeros_in_grad):

        args = get_args()

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()

        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']

        return training_log(loss_dict, loss_dict,
                            learning_rate,
                            decoupled_learning_rate,
                            self.iteration, loss_scale,
                            self.report_memory_flag, skipped_iter,
                            grad_norm, params_norm, num_zeros_in_grad)

    def rl_steps(self):
        """
        This function is the core process of the entire RL algorithm and will run the followings in the order provided:
            1) STEP 1: Actor model generates responses
            2) STEP 2: Four model Inference
            3) STEP 3: Compute advantages and returns
            4) STEP 4: Train actor model and critic
        """

        args = get_args()
        self.timers('interval-time').elapsed(barrier=True)

        with torch.no_grad():
            # STEP 1: Generate data
            self.trl_ppo_engine.set_model_eval(self.trl_ppo_engine.actor_model.model)
            self.trl_ppo_engine.set_model_eval(self.trl_ppo_engine.critic_model)
            rollout_batch = self.rollout()

            # STEP 2: Four model Inference
            rollout_batch["output_shape"] = [rollout_batch["padded_query_responses"].shape[0],
                                             rollout_batch["padded_query_responses"].shape[1] - rollout_batch["context_length"]]
            rollout_batch["ref_logprobs"] = self.model_forward(self.trl_ppo_engine.ref_model, rollout_batch, "ref_model")
            rollout_batch["actor_logprobs"] = self.model_forward(self.trl_ppo_engine.actor_model.model, rollout_batch, "actor_model")
            rollout_batch["scores"] = self.model_forward(self.trl_ppo_engine.reward_model, rollout_batch, "reward_model")
            rollout_batch["values"] = self.model_forward(self.trl_ppo_engine.critic_model, rollout_batch, "critic_model")

            # broadcast data
            for key in ["ref_logprobs", "actor_logprobs", "values", "scores"]:
                if key == "scores":
                    shape = [rollout_batch["padded_query_responses"].shape[0], 1]
                else:
                    shape = rollout_batch["output_shape"]
                rollout_batch[key] = broadcast_from_last_pipeline_stage(shape, torch.float32, rollout_batch[key])

            # STEP 3: Compute advantages and returns
            advantages, returns, padding_mask, padding_mask_p1 = self.trl_ppo_engine.compute_advantages_and_returns(rollout_batch)
            rollout_batch["advantages"] = advantages
            rollout_batch["returns"] = returns
            rollout_batch["padding_mask"] = padding_mask
            rollout_batch["padding_mask_p1"] = padding_mask_p1

            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

        # STEP 4: Train actor model and critic model
        self.trl_ppo_engine.set_model_train(self.trl_ppo_engine.actor_model.model)
        self.trl_ppo_engine.set_model_train(self.trl_ppo_engine.critic_model)
        self.update_log_metrics()

        actor_loss_dict, actor_skipped_iter, actor_grad_norm, actor_num_zeros_in_grad = \
            self.train_step(self.trl_ppo_engine.get_actor_forward_output_and_loss_func(),
                            rollout_batch, self.trl_ppo_engine.actor_model.model,
                            self.trl_ppo_engine.actor_optimizer,
                            self.trl_ppo_engine.actor_opt_param_scheduler)

        self.step_training_log(self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer, actor_loss_dict,
                          actor_skipped_iter, actor_grad_norm, actor_num_zeros_in_grad)

        critic_loss_dict, critic_skipped_iter, critic_grad_norm, critic_num_zeros_in_grad = \
            self.train_step(self.trl_ppo_engine.get_critic_forward_output_and_loss_func(),
                            rollout_batch, self.trl_ppo_engine.critic_model,
                            self.trl_ppo_engine.critic_optimizer,
                            self.trl_ppo_engine.critic_opt_param_scheduler)
        # set report_memory_flag = True only at th first iteration for both actor model and critic model
        self.report_memory_flag = self.step_training_log(self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer, critic_loss_dict,
                          critic_skipped_iter, critic_grad_norm, critic_num_zeros_in_grad)

    def train(self):
        self.initialize()
        args = get_args()

        # Iterations.
        self.iteration = args.iteration
        one_logger = get_one_logger()
        if one_logger and True:
            iteration_start = self.iteration
            train_samples_start = args.consumed_train_samples
            train_samples_target = args.train_samples
            one_logger.log_metrics({
                'train_samples_start': args.consumed_train_samples,
                'train_iterations_start': self.iteration,
                'train_samples_target': train_samples_target,
                'train_iterations_target': args.train_iters,
            })

        self.num_floating_point_operations_so_far = 0
        self.timers('interval-time', log_level=0).start(barrier=True)
        print_datetime('before the start of training step')
        self.report_memory_flag = True

        if args.manual_gc:
            # Disable the default garbage collector and perform the collection manually.
            # This is to align the timing of garbage collection across ranks.
            if args.manual_gc_interval < 0:
                raise ValueError('Manual garbage collection interval should be larger than or equal to 0.')
            gc.disable()
            gc.collect()

        eval_duration = 0.0
        eval_iterations = 0

        def track_e2e_metrics():
            # Nested function to track a bunch of E2E APP metrics
            if one_logger:
                train_duration = self.timers('interval-time').active_time()  # overall_elapsed
                train_samples = args.consumed_train_samples - train_samples_start
                train_iterations = self.iteration - iteration_start
                train_iterations_time_msecs_avg = (train_duration * 1000.0) / train_iterations if train_iterations > 0 else None
                if eval_iterations > 0:
                    validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
                else:
                    validation_iterations_time_msecs_avg = None

                one_logger.log_metrics({
                    'train_iterations_end': self.iteration,
                    'train_samples_end': args.consumed_train_samples,
                    'train_iterations': train_iterations,
                    'train_samples': train_samples,
                    'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
                    'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
                })

        if is_profile_enabled():
            prof = get_profiler()
            prof.start()

        self.trl_ppo_engine.set_model_eval(self.trl_ppo_engine.ref_model)
        self.trl_ppo_engine.set_model_eval(self.trl_ppo_engine.reward_model)

        while self.iteration < args.train_iters:
            self.iteration += 1
            self.rl_steps()

            print_datetime('after training is done')

            if self.iteration % args.log_interval == 0:
                track_e2e_metrics()

            if args.enable_high_availability:
                args.num_floating_point_operations_so_far = self.num_floating_point_operations_so_far
                args.iteration = self.iteration

            # Autoresume for actor model and critic model
            if args.adlr_autoresume and \
                    (self.iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                                                  self.trl_ppo_engine.actor_opt_param_scheduler)

                check_adlr_autoresume_termination(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                                                  self.trl_ppo_engine.critic_opt_param_scheduler)

            # Checkpointing
            saved_checkpoint = False
            if args.exit_signal_handler:
                signal_handler = get_signal_handler()
                if any(signal_handler.signals_received()):
                    save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                                             self.trl_ppo_engine.actor_opt_param_scheduler,
                                             self.num_floating_point_operations_so_far,
                                             checkpointing_context=None,
                                             save_model_type='actor')

                    save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                                             self.trl_ppo_engine.critic_opt_param_scheduler,
                                             self.num_floating_point_operations_so_far,
                                             checkpointing_context=None,
                                             save_model_type='critic')
                    print_datetime('exiting program after receiving SIGTERM.')
                    break

            if args.save and args.save_interval and \
                    self.iteration % args.save_interval == 0:
                self.timers('interval-time').stop()
                save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                                         self.trl_ppo_engine.actor_opt_param_scheduler,
                                         self.num_floating_point_operations_so_far,
                                         checkpointing_context=None,
                                         save_model_type='actor')

                save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                                         self.trl_ppo_engine.critic_opt_param_scheduler,
                                         self.num_floating_point_operations_so_far,
                                         checkpointing_context=None,
                                         save_model_type='critic')
                saved_checkpoint = True
                self.timers('interval-time', log_level=0).start(barrier=True)

            # Exiting based on duration
            if args.exit_duration_in_mins:
                train_time = (time.time() - _TRAIN_START_TIME) / 60.0
                done_cuda = torch.cuda.IntTensor(
                    [train_time > args.exit_duration_in_mins])
                torch.distributed.all_reduce(
                    done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    if not saved_checkpoint:
                        save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                                                 self.trl_ppo_engine.actor_opt_param_scheduler,
                                                 self.num_floating_point_operations_so_far,
                                                 checkpointing_context=None,
                                                 save_model_type='actor')

                        save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                                                 self.trl_ppo_engine.critic_opt_param_scheduler,
                                                 self.num_floating_point_operations_so_far,
                                                 checkpointing_context=None,
                                                 save_model_type='critic')

                    print_datetime('exiting program after {} minutes'.format(train_time))
                    break

            # Exiting based on iterations
            if args.exit_interval and self.iteration % args.exit_interval == 0:
                if args.save and not saved_checkpoint:
                    save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                                             self.trl_ppo_engine.actor_opt_param_scheduler,
                                             self.num_floating_point_operations_so_far,
                                             checkpointing_context=None,
                                             save_model_type='actor')

                    save_checkpoint_and_time(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                                             self.trl_ppo_engine.critic_opt_param_scheduler,
                                             self.num_floating_point_operations_so_far,
                                             checkpointing_context=None,
                                             save_model_type='critic')

                torch.distributed.barrier()
                print_datetime('exiting program at iteration {}'.format(self.iteration))
                break
                # save actor critic

            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            if args.manual_gc:
                if args.manual_gc_interval != 0 and self.iteration % args.manual_gc_interval == 0:
                    gc.collect()

            if is_profile_enabled():
                prof.step()

        if is_profile_enabled():
            prof.stop()

        print_datetime('after training is done')

        if args.save and self.iteration != 0 and self.iteration % args.save_interval != 0:
            save_checkpoint(self.iteration, self.trl_ppo_engine.actor_model.model, self.trl_ppo_engine.actor_optimizer,
                            self.trl_ppo_engine.actor_opt_param_scheduler,
                            self.num_floating_point_operations_so_far,
                            checkpointing_context=None,
                            save_model_type='actor')

            save_checkpoint(self.iteration, self.trl_ppo_engine.critic_model, self.trl_ppo_engine.critic_optimizer,
                            self.trl_ppo_engine.critic_opt_param_scheduler,
                            self.num_floating_point_operations_so_far,
                            checkpointing_context=None,
                            save_model_type='critic')

    def get_batch(self, data_iterator):
        """Generate a batch."""

        keys = ['input_ids', 'attention_mask']
        args = get_args()
        if args.reset_position_ids:
            keys += ['position_ids']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)

                return tokens, None, None, attention_mask, None
            else:
                if args.reset_position_ids:
                    # Broadcast data.
                    data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                    generate_actual_seq_len(data_b)

                return None, None, None, None, None

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()

        if args.reset_position_ids:
            position_ids = data_b.get('position_ids').long()
            generate_actual_seq_len(data_b)
            batch = {
                'tokens': tokens,
            }
            batch = get_batch_on_this_cp_rank(batch)
            batch['attention_mask'] = None
            batch['position_ids'] = position_ids
            return batch.values()

        attention_mask = get_tune_attention_mask(attention_mask_1d)
        return tokens, attention_mask, None

    def rollout(self):
        """
             Actor model generates responses
        """

        args = get_args()
        args.tokenizer_padding_side = "left"

        if args.global_batch_size % (args.rollout_batch_size * args.data_parallel_size):
            raise ValueError("The global_batch_size must be divisible by rollout_batch_size.")

        total_queries = []
        total_responses = []
        num_micro_batch = args.global_batch_size // args.rollout_batch_size // args.data_parallel_size

        for _ in range(num_micro_batch):
            tokens, attention_mask, position_ids = self.get_batch(self.trl_ppo_engine.train_data_iterator)

            inputs = self.trl_ppo_engine.tokenizer.batch_decode(tokens, skip_special_tokens=True)

            queries, responses, context_length = (
                self.trl_ppo_engine.actor_model.generate(
                    input_ids=inputs,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_length=args.max_length,
                    stream=False,
                    detokenize=False,
                    truncate=True,
                    return_output_log_probs=False
                )
            )

            total_queries.extend([query.tolist() for query in queries])
            total_responses.extend(
                [response.tolist() if torch.is_tensor(response) else response for response in responses])

        responses_ori_length, responses_pad_length = pad_to_tensor_dict(
            total_responses, pad_multi_of=args.pad_to_multiple_of)
        prompts_ori_length, prompts_pad_length = pad_to_tensor_dict(
            total_queries, "left", pad_multi_of=args.pad_to_multiple_of)

        padded_query_responses = torch.tensor(
            [prompt + response for prompt, response in zip(total_queries, total_responses)],
            device=torch.cuda.current_device()
        )
        sequence_lengths = torch.tensor(
            [(prompts_pad_length.item() + response_length) for response_length in responses_ori_length],
            device=torch.cuda.current_device()
        )
        attention_mask = generate_attention_mask(
            padded_query_responses.tolist(),
            prompts_ori_length,
            prompts_pad_length,
            responses_ori_length,
            responses_pad_length
        )
        position_ids = generate_position_ids_from_attention_mask(
            padded_query_responses.tolist(),
            prompts_ori_length,
            prompts_pad_length
        )

        rollout_batch = {
            "context_length": prompts_pad_length.item(),
            "sequence_lengths": sequence_lengths,
            "padded_query_responses": padded_query_responses,
            "attention_mask": torch.tensor(attention_mask, device=torch.cuda.current_device()),
            "position_ids": torch.tensor(position_ids, device=torch.cuda.current_device()),
        }

        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        return rollout_batch

    def model_forward(self, model, input_data, output_type):
        """
             Model forward_only step
        """

        args = get_args()
        setattr(args, "output_type", output_type)
        origin_micro_batch_size = args.micro_batch_size
        args.micro_batch_size = args.rollout_batch_size
        num_micro_batch = args.global_batch_size // args.rollout_batch_size // args.data_parallel_size

        seq_len = input_data["padded_query_responses"].shape[1]

        forward_backward_func = get_forward_backward_func()

        output_tensor = forward_backward_func(
            forward_step_func=self.trl_ppo_engine.forward_only_function,
            data_iterator=MiniBatchIterator(input_data, num_micro_batch, args.rollout_batch_size),
            model=model,
            num_microbatches=num_micro_batch,
            seq_length=seq_len,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True
        )

        context_length = input_data["context_length"]
        tokens = input_data["padded_query_responses"].clone()
        sequence_lengths = input_data["sequence_lengths"].clone()

        if mpu.is_pipeline_last_stage():
            output_logits = torch.cat([output[1]['logits'] for output in output_tensor], dim=0)
            if output_type == 'actor_model' or output_type == 'ref_model':
                response_logits = output_logits[:, context_length - 1:-1].to(torch.float32)
                logprob = F.log_softmax(response_logits, dim=-1)
                tokens_indices = torch.unsqueeze(tokens[:, context_length:], 2)
                output_tensor = torch.gather(logprob, 2, index=tokens_indices).squeeze(2).contiguous()
            elif output_type == 'reward_model':
                output_tensor = output_logits.squeeze(-1).gather(dim=-1, index=(sequence_lengths.unsqueeze(-1) - 1)).to(torch.float32).contiguous()
            elif output_type == 'critic_model':
                output_tensor = output_logits.squeeze(-1)[:, context_length - 1:-1].to(torch.float32).contiguous()
        else:
            output_tensor = torch.zeros(1)

        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        args.micro_batch_size = origin_micro_batch_size

        return output_tensor


def generate_position_ids_from_attention_mask(input_ids_list, prompts_ori_length, prompts_pad_length):
    """
    生成与 attention_mask 对应的 position_ids 列表。

    参数:
    input_ids_list (list of lists): 包含 input_ids 的列表，每个元素是一个列表。
    prompts_ori_length (list of lists): 包含 prompt_ori_length 的列表，每个元素是int。
    prompts_pad_length int: prompts_pad_length，int。

    返回:
    list of lists: 包含 position_ids 的列表，每个元素是一个列表。
    """
    position_ids_list = []
    for idx, input_ids in enumerate(input_ids_list):
        prompt_pad_length = prompts_pad_length - prompts_ori_length[idx]
        position_ids = [0] * prompt_pad_length + list(range(len(input_ids) - prompt_pad_length))
        position_ids_list.append(position_ids)

    return position_ids_list


def generate_attention_mask(input_ids_list, prompts_ori_length, prompts_pad_length, responses_ori_length,
                            responses_pad_length):
    """
    生成与 input_ids 对应的 attention_mask 列表。

    参数:
    input_ids_list (list of lists): 包含 input_ids 的列表，每个元素是一个列表。
    prompts_ori_length (list of lists): 包含 prompt_ori_length 的列表，每个元素是int。
    prompts_pad_length int: prompts_pad_length，int。
    responses_ori_length (list of lists): 包含 response_ori_length 的列表，每个元素是int。
    responses_pad_length int: responses_pad_length，int。

    返回:
    list of lists: 包含 attention_mask 的列表，每个元素是一个列表。
    """
    attention_mask_list = []

    for idx, input_ids in enumerate(input_ids_list):
        attention_mask = torch.ones_like(torch.tensor(input_ids))
        prompt_pad_length = prompts_pad_length - prompts_ori_length[idx]
        response_pad_length = responses_pad_length - responses_ori_length[idx]
        attention_mask[:prompt_pad_length] = 0
        if response_pad_length > 0:
            attention_mask[-response_pad_length:] = 0
        attention_mask_list.append(attention_mask.numpy().tolist())

    return attention_mask_list
