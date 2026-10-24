"""Constant propagation-graph ODE block used by ASGNODE."""

import torch

from base_classes import ODEblock


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, opt, data, device, t=torch.tensor([0.0, 1.0])):
        super().__init__(odefunc, opt, data, device, t)
        self.odefunc.edge_index = data.edge_index.to(device)
        self.odefunc.edge_weight = data.edge_attr.to(device)
        self.odefunc.lambda2 = float(data.lambda2)
        self.odefunc.lambda_max = float(data.lambda_max)

    def _solver_options(self):
        if self.opt["method"] == "rk4":
            return {"step_size": self.opt["step_size"]}
        return {
            "first_step": self.opt["dt"],
            "min_step": self.opt["dt_min"],
        }

    def trajectory(self, x, time):
        return self.integrator(
            self.odefunc,
            x,
            time.type_as(x),
            method=self.opt["method"],
            options=self._solver_options(),
            atol=self.atol,
            rtol=self.rtol,
        )

    def forward(self, x):
        state = self.trajectory(x, self.t)
        return state[-1]
