"""
Sharpness-Aware Minimization (SAM) optimizer and BatchNorm helpers for two-pass training.

See sam.md in project root for usage and references (davda54/sam, BN tip, multi-GPU, LR scheduling).
"""

import torch
import torch.nn as nn


class SAM(torch.optim.Optimizer):
    """
    SAM: Sharpness-Aware Minimization.
    Wraps a base_optimizer (e.g. AdamW, SGD) and performs two-step updates
    to minimize loss and loss sharpness. Use first_step() after first backward,
    then second forward-backward, then second_step().
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        # avoid huge scale when grad_norm is tiny (would cause overflow -> NaN)
        grad_norm = torch.clamp(grad_norm, min=1e-6)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def enable_running_stats(model: nn.Module) -> None:
    """
    Enable BatchNorm running stats update (use for first forward pass in SAM).
    Restores momentum if it was previously disabled.
    """
    def _enable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum
    model.apply(_enable)


def disable_running_stats(model: nn.Module) -> None:
    """
    Disable BatchNorm running stats update (use for second forward pass in SAM).
    Sets momentum to 0 so running mean/var are not updated during the second pass.
    """
    def _disable(module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)
