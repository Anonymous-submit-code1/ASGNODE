"""Minimal ODE base classes required by the main ASGNODE model."""

import torch
from torch import nn


class ODEblock(nn.Module):
    def __init__(self, odefunc, opt, data, device, t):
        super().__init__()
        self.opt = opt
        self.device = device
        self.t = t
        self.odefunc = odefunc(
            opt["hidden_dim"],
            opt["hidden_dim"],
            opt,
            data,
            device,
        )

        from torchdiffeq import odeint
        self.integrator = odeint
        self.atol = float(opt["tol_scale"]) * 1e-7
        self.rtol = float(opt["tol_scale"]) * 1e-9

    def set_x0(self, x0):
        self.odefunc.x0 = x0.detach().clone()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(Time Interval "
            f"{self.t[0].item()} -> {self.t[1].item()})"
        )


class ODEFunc(nn.Module):
    def __init__(self, opt, data, device):
        super().__init__()
        self.opt = opt
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.lambda2 = None
        self.lambda_max = None
        self.alpha_train = nn.Parameter(
            torch.full(
                (1, int(opt["hidden_dim"])),
                float(opt.get("gain_init", -5.0)),
            )
        )
        self.beta_train = nn.Parameter(torch.tensor(0.0))
        self.x0 = None
        self.nfe = 0

    def __repr__(self):
        return self.__class__.__name__
