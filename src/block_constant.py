import torch

from base_classes import ODEblock


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, opt, data, device, t=torch.tensor([0.0, 1.0])):
        super().__init__(odefunc, opt, data, device, t)
        self.odefunc.edge_index = data.edge_index.to(device)
        self.odefunc.edge_weight = data.edge_attr.to(device)
        self.odefunc.lambda2 = float(data.lambda2)
        self.odefunc.lambda_max = float(data.lambda_max)

    def forward(self, x):
        time = self.t.type_as(x)
        if self.opt["method"] == "rk4":
            options = {"step_size": self.opt["step_size"]}
        else:
            options = {
                "first_step": self.opt["dt"],
                "min_step": self.opt["dt_min"],
            }
        state = self.integrator(
            self.odefunc,
            x,
            time,
            method=self.opt["method"],
            options=options,
            atol=self.atol,
            rtol=self.rtol,
        )
        return state[-1]
