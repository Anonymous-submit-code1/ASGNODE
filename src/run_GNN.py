import argparse
import random
from pathlib import Path


def _positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _unit_interval(value):
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must lie strictly between 0 and 1")
    return parsed


def _probability(value):
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must lie in [0, 1)")
    return parsed


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the main ASGNODE node-classification experiment."
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        choices=(
            "Cora",
            "Citeseer",
            "Pubmed",
            "CoauthorCS",
            "Computers",
            "Photo",
            "texas",
            "wisconsin",
            "cornell",
            "film",
        ),
        default="Citeseer",
    )
    data.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Dataset cache directory.",
    )
    data.add_argument("--split-index", type=int, default=0)
    data.add_argument("--runs", type=int, default=10)
    data.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional reproducibility seed. Multi-run experiments use fresh "
            "random seeds when this option is omitted."
        ),
    )

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=None)
    training.add_argument(
        "--optimizer",
        choices=("sgd", "rmsprop", "adagrad", "adam", "adamax"),
        default=None,
    )
    training.add_argument("--lr", type=_positive_float, default=None)
    training.add_argument("--weight-decay", type=float, default=None)
    training.add_argument("--input-dropout", type=_probability, default=None)
    training.add_argument(
        "--dropout",
        type=_probability,
        default=None,
        help="Dropout applied to the class-space ODE output during training.",
    )
    training.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit device such as cuda:0.",
    )

    solver = parser.add_argument_group("ODE solver")
    solver.add_argument("--time", type=_positive_float, default=None)
    solver.add_argument(
        "--method",
        choices=("rk4", "dopri5"),
        default=None,
    )
    solver.add_argument("--step-size", type=_positive_float, default=None)
    solver.add_argument(
        "--dt",
        type=_positive_float,
        default=None,
        help="Initial adaptive step passed to Dopri5 as first_step.",
    )
    solver.add_argument(
        "--dt-min",
        "--dt_min",
        dest="dt_min",
        type=_positive_float,
        default=None,
        help="Minimum adaptive step passed to Dopri5 as min_step.",
    )
    solver.add_argument("--max-nfe", type=int, default=None)
    solver.add_argument("--tol-scale", type=_positive_float, default=None)
    solver.add_argument(
        "--eval-horizon-multiplier",
        "--earlystopxT",
        dest="eval_horizon_multiplier",
        type=_positive_float,
        default=None,
        help=(
            "Evaluate intermediate ODE times up to this multiple of the "
            "training horizon; this never interrupts training."
        ),
    )
    solver.add_argument(
        "--max-test-steps",
        "--max_test_steps",
        dest="max_test_steps",
        type=_positive_int,
        default=None,
        help="Maximum number of evaluation-time ODE steps.",
    )

    method = parser.add_argument_group("ASGNODE continuous field")
    method.add_argument(
        "--kappa",
        type=_positive_float,
        default=None,
        help="Attractive-gain lower bound; defaults to the dataset's Table 6 value.",
    )
    method.add_argument(
        "--separation-epsilon",
        type=_positive_float,
        default=None,
        help="Separation denominator epsilon; defaults to the Table 6 value.",
    )
    method.add_argument(
        "--stability-margin",
        type=_unit_interval,
        default=None,
        help="Reserved margin m in Eq. (19); defaults to the Table 6 value.",
    )
    method.add_argument(
        "--lambda2",
        type=_positive_float,
        default=None,
        help="Certificate lambda_2(L); defaults to the Table 6 value.",
    )
    method.add_argument(
        "--rho-bar",
        type=_positive_float,
        default=None,
        help="Separation one-sided Lipschitz bound; defaults to Table 6.",
    )
    method.add_argument(
        "--mu",
        type=_positive_float,
        default=None,
        help="Explicit separation-constraint strength; defaults to Table 6.",
    )
    method.add_argument(
        "--gain-init",
        type=float,
        default=None,
        help="Initial unconstrained alpha in K = kappa I + diag(softplus(alpha)).",
    )
    source = method.add_mutually_exclusive_group()
    source.add_argument(
        "--add-source",
        "--add_source",
        dest="add_source",
        action="store_true",
        default=None,
        help="Add the original ACSG learnable source term beta*x0.",
    )
    source.add_argument(
        "--no-add-source",
        dest="add_source",
        action="store_false",
        default=None,
        help="Disable the original ACSG source term.",
    )

    beltrami = parser.add_argument_group("Beltrami encoder")
    beltrami_toggle = beltrami.add_mutually_exclusive_group()
    beltrami_toggle.add_argument(
        "--beltrami",
        dest="beltrami",
        action="store_true",
        default=None,
        help=(
            "Fuse feature/position projections, project to classes, and "
            "evolve the class-space state."
        ),
    )
    beltrami_toggle.add_argument(
        "--no-beltrami",
        dest="beltrami",
        action="store_false",
        default=None,
        help="Use the direct class-space encoder.",
    )
    beltrami.add_argument("--feat-hidden-dim", type=_positive_int, default=None)
    beltrami.add_argument("--pos-enc-hidden-dim", type=_positive_int, default=None)
    beltrami.add_argument(
        "--pos-enc-type",
        choices=("GDC", "DW64", "DW128", "DW256"),
        default=None,
    )
    beltrami.add_argument(
        "--pos-enc-orientation",
        choices=("row", "col"),
        default=None,
    )
    beltrami.add_argument("--ppr-alpha", type=_unit_interval, default=None)
    beltrami.add_argument("--position-eps", type=_positive_float, default=None)
    exact = beltrami.add_mutually_exclusive_group()
    exact.add_argument(
        "--exact",
        dest="exact",
        action="store_true",
        default=None,
        help="Use exact dense PPR for a GDC positional encoding.",
    )
    exact.add_argument(
        "--approximate",
        dest="exact",
        action="store_false",
        default=None,
        help="Use approximate PPR for a GDC positional encoding.",
    )
    batch_norm = beltrami.add_mutually_exclusive_group()
    batch_norm.add_argument(
        "--batch-norm",
        dest="batch_norm",
        action="store_true",
        default=None,
    )
    batch_norm.add_argument(
        "--no-batch-norm",
        dest="batch_norm",
        action="store_false",
        default=None,
    )
    return parser


def _set_seed(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_seeds(seed, runs):
    """Choose fresh multi-run seeds unless reproducibility was requested."""

    if seed is not None:
        return [int(seed) + index for index in range(runs)]
    if runs == 1:
        return [0]
    return random.SystemRandom().sample(range(2**31), runs)


def _optimizer(name, parameters, *, lr, weight_decay, torch):
    classes = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamax": torch.optim.Adamax,
    }
    return classes[name](parameters, lr=lr, weight_decay=weight_decay)


def _apply_dataset_defaults(args):
    from main_configs import main_experiment_config

    config = main_experiment_config(args.dataset)
    for name in (
        "epochs",
        "optimizer",
        "lr",
        "weight_decay",
        "input_dropout",
        "dropout",
        "time",
        "method",
        "step_size",
        "dt",
        "dt_min",
        "max_nfe",
        "tol_scale",
        "eval_horizon_multiplier",
        "max_test_steps",
        "kappa",
        "separation_epsilon",
        "stability_margin",
        "lambda2",
        "rho_bar",
        "mu",
        "gain_init",
        "add_source",
        "beltrami",
        "feat_hidden_dim",
        "pos_enc_hidden_dim",
        "pos_enc_type",
        "pos_enc_orientation",
        "ppr_alpha",
        "position_eps",
        "exact",
        "batch_norm",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(config, name))
    return args


def _model_options(args, dataset, position_encoding=None):
    hidden_dim = int(dataset.num_classes)
    position_dim = None
    if args.beltrami:
        if position_encoding is None:
            raise ValueError("Beltrami requires a positional encoding.")
        position_dim = int(position_encoding.shape[1])
    return {
        "dataset": args.dataset,
        "hidden_dim": hidden_dim,
        "input_dropout": args.input_dropout,
        "dropout": args.dropout,
        "time": args.time,
        "method": args.method,
        "step_size": args.step_size,
        "dt": args.dt,
        "dt_min": args.dt_min,
        "max_nfe": args.max_nfe,
        "tol_scale": args.tol_scale,
        "kappa": args.kappa,
        "separation_epsilon": args.separation_epsilon,
        "stability_margin": args.stability_margin,
        "lambda2": args.lambda2,
        "rho_bar": args.rho_bar,
        "mu": args.mu,
        "gain_init": args.gain_init,
        "add_source": args.add_source,
        "beltrami": args.beltrami,
        "feat_hidden_dim": args.feat_hidden_dim,
        "pos_enc_hidden_dim": args.pos_enc_hidden_dim,
        "position_dim": position_dim,
        "batch_norm": args.batch_norm,
    }


def _evaluation_times(args, torch, device, dtype):
    """Reproduce ACSG's validation-selected ODE horizon without stopping training."""

    end_time = float(args.time) * float(args.eval_horizon_multiplier)
    if float(args.eval_horizon_multiplier) <= 1.0:
        return torch.tensor([0.0, float(args.time)], device=device, dtype=dtype)

    if args.method == "rk4":
        step = float(args.step_size)
        step_count = min(int(end_time // step), int(args.max_test_steps))
        times = torch.arange(
            step_count + 1,
            device=device,
            dtype=dtype,
        ) * step
        if step_count < int(args.max_test_steps) and float(times[-1]) < end_time:
            times = torch.cat((times, times.new_tensor([end_time])))
        return times

    sample_count = min(
        int(args.max_test_steps),
        max(1, int(round(float(args.eval_horizon_multiplier) * 10.0))),
    )
    return torch.linspace(
        0.0,
        end_time,
        sample_count + 1,
        device=device,
        dtype=dtype,
    )


def _evaluate_over_horizon(
    model,
    data,
    position_encoding,
    args,
    evaluate_logits,
    torch,
):
    times = _evaluation_times(args, torch, data.x.device, data.x.dtype)
    trajectory = model.trajectory(data.x, position_encoding, times)
    best_scores = None
    for logits in trajectory[1:]:
        scores = tuple(evaluate_logits(logits, data, "accuracy"))
        if best_scores is None or scores[1] > best_scores[1]:
            best_scores = scores
    return best_scores


def _run_once(args, run_seed, split_index=None):
    import numpy as np
    import torch

    from data import (
        load_dataset,
        load_position_encoding,
        prepare_propagation_graph,
    )
    from evaluation import evaluate_logits
    from model import ASGNODE

    _set_seed(run_seed, torch, np)
    dataset = load_dataset(
        args.dataset,
        args.data_dir,
        split_index=args.split_index if split_index is None else split_index,
        split_seed=run_seed,
    )
    position_encoding = None
    if args.beltrami:
        position_encoding = load_position_encoding(
            dataset.data,
            {
                "dataset": args.dataset,
                "pos_enc_type": args.pos_enc_type,
                "pos_enc_orientation": args.pos_enc_orientation,
                "ppr_alpha": args.ppr_alpha,
                "position_eps": args.position_eps,
                "exact": args.exact,
            },
            args.data_dir,
        )
    prepare_propagation_graph(dataset.data)

    from main_hyperparameters import continuous_field_certificate

    continuous_field_certificate(
        kappa=args.kappa,
        separation_epsilon=args.separation_epsilon,
        stability_margin=args.stability_margin,
        lambda2=args.lambda2,
        rho_bar=args.rho_bar,
        mu=args.mu,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    options = _model_options(args, dataset, position_encoding)
    model = ASGNODE(options, dataset, device=device).to(device)
    data = dataset.data.to(device)
    if position_encoding is not None:
        position_encoding = position_encoding.to(device)
    optimizer = _optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        torch=torch,
    )

    best_validation = float("-inf")
    best_scores = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.reset_nfe()
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, position_encoding)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask],
            data.y.view(-1)[data.train_mask],
        )
        loss.backward()
        optimizer.step()
        model.reset_nfe()

        model.eval()
        with torch.no_grad():
            scores = _evaluate_over_horizon(
                model,
                data,
                position_encoding,
                args,
                evaluate_logits,
                torch,
            )
        model.reset_nfe()
        if scores[1] > best_validation:
            best_validation = scores[1]
            best_scores = tuple(scores)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    train_score, validation_score, test_score = best_scores
    print(
        f"train={train_score:.4f} val={validation_score:.4f} test={test_score:.4f}"
    )
    return test_score


def main(args=None):
    import numpy as np

    parsed = _apply_dataset_defaults(build_parser().parse_args(args))
    if parsed.runs <= 0:
        raise ValueError("runs must be positive.")
    if parsed.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if parsed.max_nfe <= 0:
        raise ValueError("max_nfe must be positive.")
    if parsed.weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative.")

    heterophilic = {"texas", "wisconsin", "cornell", "film"}
    scores = []
    run_seeds = _run_seeds(parsed.seed, parsed.runs)
    for run_index, run_seed in enumerate(run_seeds):
        split_index = parsed.split_index
        if parsed.dataset in heterophilic:
            split_index = (parsed.split_index + run_index) % 10
        scores.append(
            _run_once(parsed, run_seed, split_index=split_index)
        )
    print(f"test={np.mean(scores):.4f}+/-{np.std(scores):.4f} ({len(scores)} runs)")
    return scores


if __name__ == "__main__":
    main()
