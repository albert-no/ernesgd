import math
import torch
from torch.optim import Optimizer


class OAdam(Optimizer):
    """Implements optimistic Adam algorithm.
    It has been proposed in `Training GANs with Optimism`_.
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
    .. _Training GANs with Optimism:
        https://arxiv.org/abs/1711.00141
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, anchor_gamma = 1, amsgrad=False):
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
        super(OAdam, self).__init__(params, defaults)
        self.anchor_gamma = anchor_gamma
    def __setstate__(self, state):
        super(OAdam, self).__setstate__(state)
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
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)


                # We follow the notation in ``Training GANs with Optimism''
                # exp_avg -- m_t
                # exp_avg_sq -- v_t
                # state['step'] -- t
                # group['lr'] -- eta
                # grad -- g
                # p.data -- theta

                # ==================
                # Optimism Algorithm
                # ==================
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] # m_{t-1}, v_{t-1}
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']

                if state['step'] > 0: 
                    bias_correction1 = 1 - beta1 ** state['step'] # 1 - beta1^{t-1}
                    bias_correction2 = 1 - beta2 ** state['step'] # 1 - beta2^{t-1}
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 # eta * sqrt(1 - beta2^{t-1})/(1 - beta1^{t-1})

                    # add optimism: theta_{t - 1/2} = theta_{t-1} + eta * sqrt(1 - beta2^{t-1})/(1 - beta1^{t-1}) * m_{t-1}/[sqrt(v_{t-1}) + epsilon]
                    p.data.addcdiv_(step_size*self.anchor_gamma, exp_avg, exp_avg_sq.sqrt().add(group['eps'])) 

                # t = t + 1
                state['step'] += 1

                # not sure when to update this code
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # # Optimistic update :)
                # p.data.addcdiv_(step_size, exp_avg, exp_avg_sq.sqrt().add(group['eps']))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad) # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) # v_t = beta2 * v_{t-1} + (1-beta2) * g_t * g_t
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps']) # sqrt(v_t) + epsilon

                bias_correction1 = 1 - beta1 ** state['step'] # 1 - beta1^t
                bias_correction2 = 1 - beta2 ** state['step'] # 1 - beta2^t
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 # eta * sqrt(1 - beta2^t)/(1 - beta1^t)

                # theta_t = theta_{t-1/2} - 2 * eta * sqrt(1 - beta2^t)/(1 - beta1^t) * m_t/[sqrt(v_t) + epsilon]
                p.data.addcdiv_(-(1.0+self.anchor_gamma) * step_size, exp_avg, denom) 
        return loss
