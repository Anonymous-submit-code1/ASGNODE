import torch
import torch.nn.functional as F
from torch import nn

from block_constant import ConstantODEblock
from function_laplacian_diffusion import LaplacianODEFunc


class ASGNODE(nn.Module):
    """Build a class-space state, then evolve it with the ASGNODE ODE."""

    def __init__(self, opt, dataset, device=torch.device("cpu")):
        super().__init__()
        self.opt = opt
        self.device = device
        self.num_classes = int(dataset.num_classes)
        self.num_features = int(dataset.data.num_features)
        self.beltrami = bool(opt.get("beltrami", False))

        if self.beltrami:
            self.feature_encoder = nn.Linear(
                self.num_features,
                int(opt["feat_hidden_dim"]),
            )
            self.position_encoder = nn.Linear(
                int(opt["position_dim"]),
                int(opt["pos_enc_hidden_dim"]),
            )
            self.class_projection = nn.Linear(
                int(opt["feat_hidden_dim"]) + int(opt["pos_enc_hidden_dim"]),
                self.num_classes,
            )
        else:
            self.class_projection = nn.Linear(
                self.num_features,
                self.num_classes,
            )

        self.input_batch_norm = (
            nn.BatchNorm1d(self.num_classes)
            if opt.get("batch_norm", False)
            else nn.Identity()
        )

        time = torch.tensor([0.0, float(opt["time"])], device=device)
        self.odeblock = ConstantODEblock(
            LaplacianODEFunc,
            opt,
            dataset.data,
            device,
            t=time,
        ).to(device)

    def encode_initial_state(self, features, position_encoding=None):
        if self.beltrami:
            if position_encoding is None:
                raise ValueError("Beltrami forward requires position_encoding.")
            features = F.dropout(
                features,
                p=float(self.opt["input_dropout"]),
                training=self.training,
            )
            position_encoding = F.dropout(
                position_encoding,
                p=float(self.opt["input_dropout"]),
                training=self.training,
            )
            fused_state = torch.cat(
                (
                    self.feature_encoder(features),
                    self.position_encoder(position_encoding),
                ),
                dim=-1,
            )
            initial_state = self.class_projection(fused_state)
        else:
            features = F.dropout(
                features,
                p=float(self.opt["input_dropout"]),
                training=self.training,
            )
            initial_state = self.class_projection(features)

        return self.input_batch_norm(initial_state)

    def forward(self, features, position_encoding=None):
        initial_state = self.encode_initial_state(features, position_encoding)
        self.odeblock.set_x0(initial_state)
        state = self.odeblock(initial_state)
        return F.dropout(
            state,
            p=float(self.opt["dropout"]),
            training=self.training,
        )

    def trajectory(self, features, position_encoding, times):
        initial_state = self.encode_initial_state(features, position_encoding)
        self.odeblock.set_x0(initial_state)
        return self.odeblock.trajectory(initial_state, times)

    def reset_nfe(self):
        self.odeblock.odefunc.nfe = 0
