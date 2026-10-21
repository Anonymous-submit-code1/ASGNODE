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

## Supplementary Experiment

### Propagation Dynamics under a Shared Physical-Time Grid

[Propagation dynamics under a shared physical-time grid. Columns show normalized disagreement energy, test accuracy, normalized classification margin, and degree-weighted centroid drift. Curves represent means over ten checkpoints; shaded bands indicate one sample standard deviation. The vertical dotted line marks $T_{\mathrm{ref}}=0.5$.](E4_shared_time_controls.png)


We compare full gating, ungated propagation, and spatially uniform gating on a shared physical-time grid, freezing the encoder, initial state, channel gains, graph, and readout within each checkpoint. Figure \ref{E4_shared} reports disagreement energy, test accuracy, normalized classification margin, and degree-weighted centroid drift over $t\in[0,8T_{\mathrm{ref}}]$. The three flows exhibit different energy-decay profiles, while their predictive differences depend on the dataset and propagation time. In particular, slower energy decay does not consistently imply better classification margins or accuracy. Centroid drift remains near numerical precision for all three flows, consistent with the conservation property of the symmetric graph dynamics. These frozen-checkpoint comparisons characterize propagation behavior and do not replace independently trained, time-tuned ablations.
