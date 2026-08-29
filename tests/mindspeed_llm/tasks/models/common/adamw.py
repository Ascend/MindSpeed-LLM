from typing import List, Optional, Tuple, Union
import torch
import torch_npu
from torch import Tensor
from torch.optim.optimizer import Optimizer


def _o2_adamw(params: List[Tensor],
              grads: List[Tensor],
              exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor],
              max_exp_avg_sqs: List[Tensor],
              step: int,
              *,
              amsgrad: bool,
              beta1: float,
              beta2: float,
              lr: float,
              weight_decay: float,
              eps: float,
              maximize: bool):
    """
    Functional API that performs AdamW algorithm computation specifically for o2 feature.
    """
    for i, param in enumerate(params):
        grad = grads[i]

        # start of megatron_adaptation, here we change exp_avg and exp_avg_aq into FP32,
        # avoiding precision problem in npu_apply_adam_w.
        exp_avg = exp_avgs[i].float()
        exp_avg_sq = exp_avg_sqs[i].float()
        # end of megatron_adaptation

        bias_correction1 = beta1 ** (step - 1)
        bias_correction2 = beta2 ** (step - 1)

        param.data, exp_avg, exp_avg_sq = torch_npu.npu_apply_adam_w(
            bias_correction1,
            bias_correction2,
            lr,
            weight_decay,
            beta1,
            beta2,
            eps,
            grad,
            None,
            amsgrad,
            maximize,
            out=(param.data, exp_avg, exp_avg_sq)
        )
        # start of megatron_adaptation, here we recover exp_avg back to exp_avgs
        exp_avgs[i].copy_(exp_avg)
        exp_avg_sqs[i].copy_(exp_avg_sq)
        # end of megatron_adaptation


class O2AdamW(Optimizer):
    """
    Adamw for o2 feature called O2AdamW.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(O2AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(O2AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # start of megatron_adaptation, here we initialize exp_avg and exp_avg_sq in bf16
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16, memory_format=torch.preserve_format)
                    # end of megatron_adaptation, here we initialize exp_avg and exp_avg_sq in bf16

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            _o2_adamw(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      group['step'],
                      amsgrad=amsgrad,
                      beta1=beta1,
                      beta2=beta2,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      eps=group['eps'],
                      maximize=group['maximize'])

        return loss
