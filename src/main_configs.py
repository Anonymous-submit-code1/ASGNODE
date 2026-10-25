"""Dataset defaults for the main ASGNODE node-classification experiment.

Certified defaults use the current manuscript Table 8 and its solver choices.
Other training settings are inherited from the original main implementation.
Historical values are available only through explicit legacy mode.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class MainExperimentConfig:
    epochs: int
    optimizer: str
    lr: float
    weight_decay: float
    input_dropout: float
    time: float
    method: str
    step_size: float
    max_nfe: int
    tol_scale: float
    kappa: float
    separation_epsilon: float
    stability_margin: float
    lambda2: float
    rho_bar: float
    mu: float
    add_source: bool
    beltrami: bool
    feat_hidden_dim: int
    pos_enc_hidden_dim: int
    pos_enc_type: str
    pos_enc_orientation: str
    ppr_alpha: float
    position_eps: float
    exact: bool
    batch_norm: bool
    dt: float
    dt_min: float
    dropout: float
    eval_horizon_multiplier: float
    max_test_steps: int
    gain_init: float
    separation_beta: float = 0.5


LEGACY_EXPERIMENT_CONFIGS = {
    "Cora": MainExperimentConfig(
        100, "adamax", 0.022924849756740397, 0.00507685443154266,
        0.5, 18.3, "rk4", 1.0, 2000, 821.9773048827274,
        0.25, 0.2, 0.5, 1.48e-2, 8.45e2, 2.19e-6, True,
        True, 64, 16, "GDC", "row", 0.05, 0.01, True,
        False, 0.001, 1e-5, 0.0, 1.0, 100, -5.0,
    ),
    "Citeseer": MainExperimentConfig(
        250, "adam", 0.00863585231323069, 0.1,
        0.6803233752085334, 3.5, "rk4", 1.0, 3000, 2.9010446330432815,
        2.0, 0.025, 0.5, 6.52e-3, 4.01e3, 1.63e-6, True,
        True, 64, 16, "GDC", "row", 0.05, 0.01, True,
        False, 0.001, 1e-5, 0.0, 1.0, 100, -6.0,
    ),
    "Pubmed": MainExperimentConfig(
        600, "adamax", 0.014669345840305131, 0.0018236722171703636,
        0.5, 12.942327880200853, "dopri5", 1.0, 5000, 1991.0688305523001,
        0.25, 0.2, 0.5, 2.75e-2, 8.61e2, 3.99e-6, True,
        True, 64, 16, "GDC", "row", 0.05, 0.01, False,
        False, 0.001, 1e-5, 0.07191100715473969, 1.0, 100, -5.0,
    ),
    "CoauthorCS": MainExperimentConfig(
        250, "rmsprop", 0.0009342860080741642, 0.004738413087298854,
        0.5275042493231822, 3.126400580172773, "dopri5", 1.0,
        3000, 9348.983916372074,
        0.25, 0.2, 0.5, 1.02e-2, 6.86e2, 1.88e-6, True,
        True, 128, 32, "GDC", "row", 0.05, 0.0001, False,
        False, 0.001, 1e-5, 0.6857774850321, 1.0, 100, -5.0,
    ),
    "Computers": MainExperimentConfig(
        100, "adam", 0.0035304663972281548, 0.007674669913252157,
        0.5973137276937647, 3.249016177876166, "dopri5", 1.0,
        500, 127.46369887079446,
        0.25, 0.2, 0.5, 1.17e-1, 1.49e4, 9.77e-7, False,
        True, 64, 32, "DW128", "row", 0.05, 0.01, False,
        False, 0.001, 1e-5, 0.08732611854459256, 1.0, 100, -5.0,
    ),
    "Photo": MainExperimentConfig(
        100, "adam", 0.005560726683883279, 0.004707800883497945,
        0.42903126506740247, 3.5824027975386623, "dopri5", 1.0,
        500, 2086.525473167121,
        0.5, 0.1, 0.5, 5.54e-2, 1.43e4, 9.65e-7, True,
        True, 64, 16, "DW128", "row", 0.05, 0.01, False,
        True, 0.001, 1e-5, 0.46502284638600183, 1.0, 100, -5.0,
    ),
    "texas": MainExperimentConfig(
        200, "adamax", 0.02, 0.0005,
        0.5, 1.5, "rk4", 1.0, 2000, 11353.0,
        0.5, 0.1, 0.5, 1.11e-1, 1.05e3, 2.64e-5, False,
        True, 64, 16, "GDC", "row", 0.05, 0.01, True,
        False, 0.001, 1e-5, 0.0, 1.0, 100, -5.0,
    ),
    "wisconsin": MainExperimentConfig(
        150, "adamax", 0.01, 0.0005,
        0.5, 2.0, "rk4", 1.0, 2000, 821.9773048827274,
        0.1, 0.5, 0.5, 2.21e-1, 2.46e2, 4.50e-5, False,
        True, 64, 16, "GDC", "row", 0.05, 0.01, True,
        False, 0.001, 1e-5, 0.0, 1.0, 100, -5.0,
    ),
    "cornell": MainExperimentConfig(
        100, "adamax", 0.01, 0.001,
        0.5, 1.3, "rk4", 1.0, 2000, 821.9773048827274,
        0.1, 0.5, 0.5, 1.94e-1, 1.90e2, 5.10e-5, False,
        True, 64, 16, "GDC", "row", 0.05, 0.01, True,
        False, 0.001, 1e-5, 0.0, 1.0, 100, -5.0,
    ),
    "film": MainExperimentConfig(
        500, "adamax", 0.05, 0.001,
        0.3, 0.25, "rk4", 1.0, 2000, 821.9773048827274,
        0.25, 0.2, 0.5, 8.64e-2, 6.52e3, 1.66e-6, False,
        True, 64, 8, "GDC", "row", 0.05, 0.01, False,
        False, 0.001, 1e-5, 0.5, 1.0, 100, -5.0,
    ),
}


_PAPER_TABLE = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "paper_table8.json").read_text(encoding="utf-8")
)["datasets"]
MAIN_EXPERIMENT_CONFIGS = {
    dataset: replace(
        config,
        **_PAPER_TABLE[dataset],
        mu=_PAPER_TABLE[dataset]["separation_beta"] * _PAPER_TABLE[dataset]["separation_epsilon"] * _PAPER_TABLE[dataset]["kappa"],
        stability_margin=1.0 - _PAPER_TABLE[dataset]["separation_beta"],
        add_source=False,
        beltrami=False,
        batch_norm=False,
    )
    for dataset, config in LEGACY_EXPERIMENT_CONFIGS.items()
}


def main_experiment_config(dataset: str, field_mode: str = "certified") -> MainExperimentConfig:
    if field_mode not in {"certified", "legacy"}:
        raise ValueError("field_mode must be certified or legacy")
    configs = MAIN_EXPERIMENT_CONFIGS if field_mode == "certified" else LEGACY_EXPERIMENT_CONFIGS
    try:
        return configs[dataset]
    except KeyError as error:
        choices = ", ".join(MAIN_EXPERIMENT_CONFIGS)
        raise ValueError(
            f"Unknown main-experiment dataset: {dataset}. Choices: {choices}"
        ) from error
