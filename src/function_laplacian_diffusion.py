import torch
from torch import Tensor
import torch.nn.functional as F
import torch_sparse

from base_classes import ODEFunc
from stability import StabilityParameters
from graph_type import edge_weight_baseline
from main_hyperparameters import continuous_field_certificate


class MaxNFEException(RuntimeError):
    """Raised when one forward solve exceeds its function-evaluation budget."""


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt, data, device):
        super(LaplacianODEFunc, self).__init__(opt, data, device)

        self.data = data
        self.in_features = in_features
        self.out_features = out_features
        self._edge_reference = edge_weight_baseline(
            opt.get('dataset', ''),
            opt.get('graph_type'),
        )

    def sparse_multiply(self, x):
        edge_p = self.edge_weight - self._edge_reference
        return torch_sparse.spmm(
            self.edge_index,
            edge_p,
            x.shape[0],
            x.shape[0],
            x,
        )

    def get_stability_parameters(self, x: Tensor) -> StabilityParameters:
        certificate = continuous_field_certificate(
            kappa=self.opt['kappa'],
            separation_epsilon=self.opt['separation_epsilon'],
            stability_margin=self.opt['stability_margin'],
            lambda2=self.opt['lambda2'],
            rho_bar=self.opt['rho_bar'],
            mu=self.opt['mu'],
        )

        def as_tensor(value):
            return torch.as_tensor(value, dtype=x.dtype, device=x.device)

        return StabilityParameters(
            kappa=as_tensor(certificate.kappa),
            mu=as_tensor(certificate.mu),
            separation_epsilon=as_tensor(certificate.separation_epsilon),
            stability_margin=as_tensor(certificate.stability_margin),
            lambda2=as_tensor(certificate.lambda2),
            lambda_max=as_tensor(certificate.lambda_max),
            rho_bar=as_tensor(certificate.rho_bar),
            contraction_margin=as_tensor(certificate.contraction_margin),
            budget_ratio=as_tensor(certificate.budget_ratio),
        )

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        if self.nfe >= self.opt["max_nfe"]:
            raise MaxNFEException
        self.nfe += 1

        return self.velocity_components(x)['field']

    def velocity_components(self, x: Tensor):
        """Return attractive, separation, and full velocity components."""
        params = self.get_stability_parameters(x)

        v_att = self.sparse_multiply(x) - x
        channel_gain = params.kappa + F.softplus(self.alpha_train)
        attractive = v_att * channel_gain
        separation = local_separation_velocity(
            x,
            self.edge_index,
            self.edge_weight,
            params.separation_epsilon.item(),
        )
        separation_constraint = params.mu * separation
        v = attractive - separation_constraint
        source = torch.zeros_like(x)
        if self.opt.get('add_source', False):
            if self.x0 is None:
                raise RuntimeError("add_source requires the ODE block to set x0.")
            source = self.beta_train * self.x0
            v = v + source

        return {
            'attractive': attractive,
            'separation': separation,
            'separation_constraint': separation_constraint,
            'source': source,
            'field': v,
        }


def local_separation_velocity(
    Z: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor = None,
    epsilon: float = 1e-6,
) -> Tensor:
    """
    Compute the local separation correction S_minus(t).

    Formula:
        s_{i,-}(t)
        = sum_{j in N_i} a_ij * (z_j(t) - z_i(t))
          / (||z_j(t) - z_i(t)||_2^2 + epsilon)

    Parameters
    ----------
    Z:
        Node states with shape [N, C].
        N is the number of nodes and C is the number of classes.
    edge_index:
        Propagation matrix coordinates with shape [2, E]:
            edge_index[0, e] = i  # row / center node
            edge_index[1, e] = j  # column / neighboring node
    edge_weight:
        Optional normalized edge weights with shape [E].
        If omitted, all edges receive weight 1.
    epsilon:
        Positive numerical-stability constant.

    Returns
    -------
    Tensor:
        S_minus with shape [N, C].
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must have shape [N, C], but received {Z.shape}.")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, E], "
            f"but received {edge_index.shape}."
        )

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    edge_index = edge_index.to(device=Z.device, dtype=torch.long)

    center = edge_index[0]  # row i of the propagation operator
    neighbor = edge_index[1]  # neighboring node j

    # Relative difference for each edge: z_j - z_i
    diff = Z[neighbor] - Z[center]  # [E, C]

    # Squared Euclidean distance: ||z_j - z_i||_2^2
    squared_distance = torch.sum(diff * diff, dim=-1, keepdim=True)  # [E, 1]

    # Local separation message along each edge
    messages = diff / (squared_distance + epsilon)  # [E, C]

    if edge_weight is not None:
        if edge_weight.ndim != 1 or edge_weight.shape[0] != center.shape[0]:
            raise ValueError(
                "edge_weight must have shape [E], "
                f"but received {edge_weight.shape}."
            )

        edge_weight = edge_weight.to(device=Z.device, dtype=Z.dtype)
        messages = messages * edge_weight.unsqueeze(-1)

    # Aggregate messages from neighboring nodes j to each center node i
    S_minus = torch.zeros_like(Z)
    S_minus.index_add_(dim=0, index=center, source=messages)

    return S_minus
