"""Tensor-valued stability quantities used by the ASGNODE vector field."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class StabilityParameters:
    kappa: Tensor
    mu: Tensor
    separation_epsilon: Tensor
    stability_margin: Tensor
    lambda2: Tensor
    lambda_max: Tensor
    rho_bar: Tensor
    contraction_margin: Tensor
    budget_ratio: Tensor
