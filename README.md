# Towards Asymptotic Stability: Attraction-Separation Dynamics against Oversmoothing in Graph Neural Networks

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
