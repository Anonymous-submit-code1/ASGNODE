# Towards Asymptotic Stability: Attraction-Separation Dynamics against Oversmoothing in Graph Neural Networks

![overview of ASGNODE](overview.png)

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
