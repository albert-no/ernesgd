import math
import torch
from torch.optim import Optimizer
import numpy as np


class AnchorAdam(Optimizer):
    """Implements optimistic Adam algorithm.
    It has been proposed in [XXX]
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Training GANs with Ahchoring: [XXX]
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, anchor_rate_mode='exponential', anchor_gamma = 1, anchor_start_from = 0, 
                 anchor_update_period = -1, inner_iters_per_epoch = 500, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AnchorAdam, self).__init__(params, defaults)

        self.anchor_rate_mode = anchor_rate_mode
        self.anchor_gamma = anchor_gamma
        self.anchor_start_from = anchor_start_from
        self.anchor_update_period = anchor_update_period
        self.inner_iters_per_epoch = inner_iters_per_epoch
        self.anchor_rate = 0

    def __setstate__(self, state):
        super(AnchorAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    # --- we initialize the anchor variable: anchor = 0 ---
                    state['anchor'] = p.data.clone()
                    state['anchor'].mul_(0)
                    # --- end ---

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                # ==================
                # Anchoring Algorithm
                # ==================

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] # m_{t-1}, v_{t-1}

                # -- turn on the anchor when t == desired iteration numbers --
                if state['step'] == self.anchor_start_from * self.inner_iters_per_epoch: # anchor is fixed
                    state['anchor'] = p.data.clone()

                    self.anchor_rate = self.anchor_gamma

                if self.anchor_update_period <= 0:
                    # === anchor is fixed but the anchor_rate is decaying ==
                    n_epoch = state['step']//self.inner_iters_per_epoch # the index of epoch

                    if n_epoch >= self.anchor_start_from:
                        count = n_epoch - self.anchor_start_from

                        # -- anchor decays in an exponential rate --
                        if  self.anchor_rate_mode == 'exponential':
                            self.anchor_rate = self.anchor_gamma*(0.97 ** count)
                        
                        # -- anchor decays in a 1/t rate --
                        elif self.anchor_rate_mode == 'one_over_k':
                            self.anchor_rate = self.anchor_gamma/(count + 1)

                        # -- anchor decays in a 1/sqrt(t) rate --
                        elif self.anchor_rate_mode == 'one_over_sqrt_k':
                            self.anchor_rate = self.anchor_gamma/np.sqrt(count + 1)

                else:
                    # === anchor is updated for every several epochs but the anchor_rate is fixed ===
                    if state['step'] % (self.anchor_update_period * self.inner_iters_per_epoch) == 0 and self.anchor_rate > 0:
                        # --- if the anchor is turned on and the update period arrives, we update the anchor ---
                        state['anchor'] = p.data.clone()

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']
                anchor = state['anchor']

                # # Examine the correctness of the code
                # if state['step'] % (self.inner_iters_per_epoch) == 0:
                #     print('anchor:', anchor)
                #     print('pdata:', p.data)
                #     print('anchor_beta:', self.anchor_gamma)
                #     print('anchor_rate:', self.anchor_rate)
                #     # print('anchor_coeff:', self.anchor_rate * step_size)

                # t = t + 1
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)               # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # v_t = beta2 * v_{t-1} + (1-beta2) * g_t * g_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps']) # sqrt(v_t) + epsilon

                bias_correction1 = 1 - beta1 ** state['step'] # 1 - beta1^t
                bias_correction2 = 1 - beta2 ** state['step'] # 1 - beta2^t
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 # scaled_stepsize = eta * sqrt(1 - beta2^t)/(1 - beta1^t)

                ## --- add anchor WITH adam ---
                ## scaled_stepsize = eta * sqrt(1 - beta2^t)/(1 - beta1^t)
                ## theta_t = theta_{t-1} - scaled_stepsize * m_t/[sqrt(v_t) + epsilon] + anchor_rate * scaled_stepsize * (anchor - theta_{t-1})
                ##         = (1 - anchor_rate * scaled_stepsize) * theta_{t-1} - scaled_stepsize * m_t/[sqrt(v_t) + epsilon] + anchor_rate * scaled_stepsize * anchor
                p.data.mul_(1 - self.anchor_rate * step_size)                      # in-place multiplication
                p.data.addcdiv_(-step_size, exp_avg, denom)     # in-place add&div  
                # p.data.add_(anchor_rate * step_size * anchor)                 # in-place addition
                p.data.add_(self.anchor_rate * step_size * anchor) 
                # --- end ---

                # --- add anchor WITH VARIANCE SCALING ADAM ---
                # scaled_stepsize = eta * sqrt(1 - beta2^t)/(1 - beta1^t)
                # theta_t = theta_{t-1} - scaled_stepsize * m_t/[sqrt(v_t) + epsilon] + anchor_rate * scaled_stepsize * (anchor - theta_{t-1})
                #         = (1 - anchor_rate * scaled_stepsize) * theta_{t-1} - scaled_stepsize * m_t/[sqrt(v_t) + epsilon] + anchor_rate * scaled_stepsize * anchor
                #p.data.mul_(1 - self.anchor_rate * step_size)                      # in-place multiplication
                
                #p.data.addcdiv_(-self.anchor_rate*step_size, p.data, denom)     # in-place add&div  
                #p.data.addcdiv_(-step_size, exp_avg, denom)     # in-place add&div  
                ## p.data.add_(anchor_rate * step_size * anchor)                 # in-place addition
                ##p.data.add_(self.anchor_rate * step_size * anchor)
                #p.data.addcdiv_(self.anchor_rate*step_size, anchor, denom)     # in-place add&div  
               # --- end ---

               
        return loss
