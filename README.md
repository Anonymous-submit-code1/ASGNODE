# Asymptotically Stable Graph Diffusion: Selective Smoothing at Matched Energy

![overview of ASGNODE](overview.png)

Install the runtime dependencies from `requirements.txt` in a PyTorch/PyG
environment compatible with your CPU or CUDA installation.

## Main experiment

Run the experiment from this directory with:

```bash
python src/run_GNN.py
```

The dataset configuration explicitly contains all six certificate values
reported in Table 6:


| Dataset | lr | weight_decay | _T_ | _κ_ | _ε_ | _β_ | 
|---------|------|--------------|------|-------|-------|-------|
| Cora | 0.0229 | 0.005 | 18.3 | 0.25 | 0.2 | 0.9 |
| CiteSeer | 0.0086 | 0.1 | 3.5 | 2 | 0.025 | 0.9 |
| PubMed | 0.0146 | 0.001 | 12.94 | 0.25 | 0.2 | 0.9 |
| Coauthor CS | 0.0009 | 0.004 | 3.13 | 0.25 | 0.2 | 0.9 |
| Computers | 0.0035 | 0.007 | 3.25 | 0.25 | 0.2 | 0.9 |
| Photo | 0.0055 | 0.004 | 3.58 | 0.5 | 0.1 | 0.9 |
| Texas | 0.02 | 0.0005 | 1.5 | 0.5 | 0.1 | 0.9 |
| Wisconsin | 0.01 | 0.0005 | 2 | 0.1 | 0.5 | 0.9 |
| Cornell | 0.01 | 0.001 | 1.3 | 0.1 | 0.5 | 0.9 |
| Film | 0.05 | 0.001 | 0.25 | 0.25 | 0.2 | 0.9 |

- `--lr` is the learning rate;
- `--κ` is the positive lower bound of the attractive gain;
- `--ε` controls the transition scale of the distance-dependent gate;
- `--β` scales the separation correction through _μ_=_βκ_, with 0<_β_<1 ensuring the stated sufficient stability condition;
- `--T` is the terminal time at which the ODE solution is used for prediction.
