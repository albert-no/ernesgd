import torch
from torch.optim.optimizer import Optimizer, required


class ERNESGD(Optimizer):
    # XXX (TBD) update comments
    r"""Implements NEW Ernest stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.half_step()
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.full_step()
        XXX Citationneeded)
    .. note:: XXX (TBD) update comments
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ERNESGD, self).__init__(params, defaults)
        # XXX New part (store the original values)
        self.param_groups_prev = self.param_groups[:]

    def __setstate__(self, state):
        super(ERNESGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def half_step(self, closure=None):
        """Performs a half optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # XXX need to update param_group_prev after fisrt loss
        self.param_groups_prev[:] = self.param_groups[:]

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    # XXX when does this happen? Should we put an "assert" here?
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def full_step(self, closure=None):
        """Performs a second half (full) optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # XXX New part in subtract the gradient from Z_k
        for idx, group in enumerate(self.param_groups):
            # XXX get corresponding group_prev
            group_prev = self.param_groups_prev[idx]

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param_idx, p in enumerate(group['params']):
                # XXX get corresponding data point
                prev_p = group_prev['params'][param_idx]
                if p.grad is None:
                    continue
                # Use new gradient (unchanged)
                d_p = p.grad.data
                if weight_decay != 0:
                    # XXX starting point is prev_p
                    d_p.add_(weight_decay, prev_p.data)

                # XXX nonzero momentum case has not been changed
                # (need to update if we want to use this)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # XXX update prev_p since the starting point is prev_p
                prev_p.data.add_(-group['lr'], d_p)
        # XXX New part update original param_groups
        self.param_groups[:] = self.param_groups_prev[:]
        # XXX Note: in order to use inplace add_, we update param_groups_prev first
        # then update param_groups. This might cause an issue of gradients
        # Or it may not be an issue because grad is computed in loss backward anyways

        # XXX XXX need to update param_group_prev after fisrt loss

        return loss
