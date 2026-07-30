# ASGNODE

This directory contains the cleaned code for the paper's main semi-supervised
node-classification experiment. The original research workspace is unchanged.
Ablations, plotting scripts, external baselines, diagnostics, caches, and
historical outputs are excluded.

Install the runtime dependencies from `requirements.txt` in a PyTorch/PyG
environment compatible with your CPU or CUDA installation.

## Main experiment

Run the experiment from this directory with:

```bash
python src/run_GNN.py \
  --dataset Cora
```

The dataset configuration explicitly contains all six certificate values
reported in Table 6:

- `--kappa` is the positive lower bound of the attractive gain;
- `--separation-epsilon` is the epsilon in the local separation denominator;
- `--stability-margin` is the reserved margin `m` in Eq. (19).
- `--lambda2` is the certificate value `lambda_2(L)`;
- `--rho-bar` is the separation one-sided Lipschitz bound;
- `--mu` is the strength of the separation constraint.

All six are parser options. Supplying one explicitly overrides only that
Table 6 default. A fully manual run can therefore use:

```bash
python src/run_GNN.py --dataset Cora \
  --kappa 0.25 --separation-epsilon 0.2 --stability-margin 0.5 \
  --lambda2 0.0148 --rho-bar 845 --mu 2.19e-6
```

When `rho_bar` or `mu` is omitted in direct API use, it can still be derived
from Eqs. (16) and (19):

```text
rho_bar = lambda_max(L) / epsilon
mu      = (1 - m) * kappa * lambda_2(L) / rho_bar
field   = attractive - mu * separation
```

In the main experiment, the explicit Table 6 `mu` is used directly, so the
normalized separation constraint is always present in the ODE.

The original ACSG source term is also available explicitly:

```text
field = attractive - mu * separation + beta * x0
```

Use `--add-source` or `--no-add-source` to override its dataset default. It is
enabled by default for Cora, Citeseer, Pubmed, CoauthorCS, and Photo, matching
the original main-parameter table, and disabled for the other datasets. The
scalar `beta` is learned from zero and `x0` is the detached initial ODE state.

Beltrami encoding is enabled in the main configuration and feeds an explicit
class-space projection before the ODE:

```text
X_f = Linear(features)
X_p = Linear(position_encoding)
Z(0) = class_projection(concat(X_f, X_p))
logits = ODE(Z(0)), with Z(0) in R^(N x num_classes)
```

The parser exposes `--beltrami`/`--no-beltrami`, the feature and positional
hidden dimensions, positional-encoding type/orientation, PPR parameters,
`--exact`/`--approximate`, and class-space batch normalization. GDC encodings
are generated and cached under
`data/pos_encodings`. Original `DW*` encodings are loaded from the matching
pickle cache because they were precomputed assets in the original project.

Legacy alternatives such as `stability_r`, `stability_mu`, `mu_scale`,
`stability_delta`, and diagnostic/ablation controls are not part of the main
experiment parser.

Training and solver defaults are selected per dataset from the original ACSG
main-experiment protocol. Every setting remains a command-line option, so an
explicit value overrides the dataset default. The propagation graph uses the
performance-oriented random-walk normalization from ACSG. The certificate
spectrum uses the corresponding unnormalized symmetric base Laplacian, as in
Table 6; self-loops cancel in this Laplacian and do not change its spectrum.

Multi-run experiments use fresh random seeds by default. Pass `--seed N` only
when a reproducible sequence `N, N+1, ...` is required. A single run without
`--seed` keeps seed 0 for convenient debugging.

`--dt` is the initial adaptive step used by Dopri5 (`first_step` in
`torchdiffeq`) and `--dt-min` is its minimum step (`min_step`); RK4 continues
to use `--step-size`. `--dropout` is applied to the class-space ODE output
during training and is disabled during evaluation.

The class-space defaults were validated separately from the six Table 6
certificate values. Cora and Citeseer use zero output-logit dropout. Film uses
`time=0.25`, `input_dropout=0.3`, an 8-dimensional positional branch, and
approximate GDC; these performance settings do not alter any Table 6 value.

The learnable attractive gain starts close to its prescribed lower bound
(`alpha=-5`) instead of at zero. This avoids adding an initial
`softplus(0)=0.693` gain on top of `kappa`, while retaining the paper's
parameterization `K = kappa I + diag(softplus(alpha))`.

## Source layout

- `src/run_GNN.py`: main parser and training loop;
- `src/main_configs.py`: per-dataset training and solver defaults;
- `src/model.py`: class-space ASGNODE model;
- `src/function_laplacian_diffusion.py`: attraction-separation vector field;
- `src/main_hyperparameters.py`: validation and formula-derived quantities;
- `src/data.py`: the ten main datasets and Laplacian-spectrum preparation;
- `src/graph_type.py`: implicit homophilic/heterophilic edge-weight policy.

The edge propagation rule keeps homophilic weights unchanged and internally
subtracts `0.5` from heterophilic weights without reading node labels.
