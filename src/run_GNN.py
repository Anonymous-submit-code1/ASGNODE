"""Main semi-supervised node-classification experiment for ASGNODE."""

import argparse
import json
import math
import random
import time
import hashlib
from pathlib import Path


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
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


def _nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
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
    training.add_argument("--result-json", default=None, help="Write configurations and per-run results.")
    training.add_argument("--checkpoint-dir", default=None)
    training.add_argument("--validation-only", action="store_true", help="Do not compute test accuracy during parameter selection.")
    training.add_argument("--show-config", action="store_true", help="Print effective settings without downloading data or training.")
    training.add_argument("--log-every", type=_positive_int, default=20)
    training.add_argument("--diagnostics", action="store_true", help="Measure field and weighted-energy diagnostics at the selected checkpoint.")
    training.add_argument("--num-threads", type=_positive_int, default=4)
    training.add_argument("--anchor-loss-weight", type=_nonnegative_float, default=0.0,
                          help="Training-label-only cross entropy on Z(0), added to terminal loss.")

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
        "--field-mode", choices=("certified", "legacy"), default="certified",
        help="Certified: mu=beta*epsilon*kappa, common nonnegative RW weights, no source. Legacy reproduces the supplied code.",
    )
    method.add_argument(
        "--separation-beta", type=_probability, default=None,
        help="Energy budget beta in [0,1); certified defaults use the dataset's Table 8 value.",
    )
    method.add_argument(
        "--separation-kernel", choices=("inverse", "boundary", "linear"), default=None,
        help="Certified default: increasing-distance boundary correction. Inverse and linear require explicit selection.",
    )
    method.add_argument(
        "--kappa",
        type=_positive_float,
        default=None,
        help="Attractive-gain lower bound; certified defaults use the dataset's Table 8 value.",
    )
    method.add_argument(
        "--separation-epsilon",
        type=_positive_float,
        default=None,
        help="Gate scale epsilon; certified defaults use the dataset's Table 8 value.",
    )
    method.add_argument(
        "--stability-margin",
        type=_unit_interval,
        default=None,
        help="Legacy-only margin m in Eq. (19). Certified mode uses separation-beta.",
    )
    method.add_argument(
        "--lambda2",
        type=_positive_float,
        default=None,
        help="Legacy-only Table 6 lambda_2. Certified mode computes the normalized gap.",
    )
    method.add_argument(
        "--rho-bar",
        type=_positive_float,
        default=None,
        help="Legacy-only Table 6 bound. Certified mode derives it from the actual graph.",
    )
    method.add_argument(
        "--mu",
        type=_positive_float,
        default=None,
        help="Explicit mu override. Certified default is beta*epsilon*kappa; legacy uses Table 6.",
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

    config = main_experiment_config(args.dataset, args.field_mode)
    explicit_mu = args.mu
    explicit_beta = args.separation_beta
    if args.separation_kernel is None:
        args.separation_kernel = (
            "boundary" if args.field_mode == "certified"
            else "inverse"
        )
    if args.field_mode == "certified":
        if args.add_source:
            raise ValueError("--add-source requires --field-mode legacy; it is outside the consensus certificate.")
        if args.lambda2 is not None or args.rho_bar is not None:
            raise ValueError("Certified mode computes the normalized spectrum from the actual graph; lambda2/rho-bar overrides are legacy only.")
        if args.stability_margin is not None:
            raise ValueError("Use --separation-beta in certified mode; --stability-margin is legacy only.")
        if explicit_mu is not None and explicit_beta is not None:
            raise ValueError("Specify either --mu or --separation-beta, not both.")
        args.add_source = False
    elif explicit_beta is not None:
        raise ValueError("--separation-beta is only defined for the certified field.")
    if args.field_mode == "legacy" and args.separation_kernel != "inverse":
        raise ValueError("Alternative separation kernels require certified mode.")
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
    if args.field_mode == "certified":
        args.separation_beta = (
            explicit_mu / (args.separation_epsilon * args.kappa)
            if explicit_mu is not None else (
                config.separation_beta
                if explicit_beta is None else explicit_beta
            )
        )
        if not 0 <= args.separation_beta < 1:
            raise ValueError("Certified separation requires 0 <= mu/(epsilon*kappa) < 1.")
        args.mu = args.separation_beta * args.separation_epsilon * args.kappa
        args.stability_margin = 1.0 - args.separation_beta
        # Replaced with the actual normalized spectrum after graph preparation.
        args.lambda2 = None
        args.rho_bar = None
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
        "field_mode": args.field_mode,
        "separation_beta": args.separation_beta,
        "separation_kernel": args.separation_kernel,
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
    return_details=False,
):
    times = _evaluation_times(args, torch, data.x.device, data.x.dtype)
    trajectory = model.trajectory(data.x, position_encoding, times)
    best_scores = None
    best_logits, best_time = None, None
    for current_time, logits in zip(times[1:], trajectory[1:]):
        scores = tuple(evaluate_logits(logits, data, "accuracy"))
        if best_scores is None or scores[1] > best_scores[1]:
            best_scores = scores
            best_logits, best_time = logits, float(current_time)
    if return_details:
        return best_scores, best_logits, best_time
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

    torch.set_num_threads(args.num_threads)
    _set_seed(run_seed, torch, np)
    started = time.perf_counter()
    effective_split_index = args.split_index if split_index is None else split_index
    dataset = load_dataset(
        args.dataset,
        args.data_dir,
        split_index=effective_split_index,
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
    prepare_propagation_graph(dataset.data, certified=args.field_mode == "certified")

    from main_hyperparameters import continuous_field_certificate, pairwise_field_certificate

    if args.field_mode == "certified":
        args.lambda2 = dataset.data.normalized_lambda2
        args.rho_bar = dataset.data.normalized_lambda_max / args.separation_epsilon
        certificate = pairwise_field_certificate(
            kappa=args.kappa, separation_epsilon=args.separation_epsilon,
            separation_beta=args.separation_beta, lambda2=args.lambda2,
            lambda_max=dataset.data.normalized_lambda_max, mu=args.mu,
            separation_kernel=args.separation_kernel,
        )
        args.rho_bar = certificate.rho_bar
    else:
        certificate = continuous_field_certificate(
            kappa=args.kappa, separation_epsilon=args.separation_epsilon,
            stability_margin=args.stability_margin, lambda2=args.lambda2,
            rho_bar=args.rho_bar, mu=args.mu,
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    options = _model_options(args, dataset, position_encoding)
    model = ASGNODE(options, dataset, device=device).to(device)
    data = dataset.data.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    best_logits, best_epoch, best_time = None, None, None
    history = []

    def validation_scores(logits, graph, metric):
        predictions = logits.argmax(-1)
        labels = graph.y.view(-1)
        return [float((predictions[mask] == labels[mask]).float().mean())
                for mask in (graph.train_mask, graph.val_mask)]

    print(f"seed={run_seed} mode={args.field_mode} source={args.add_source} "
          f"beta={args.separation_beta} mu={args.mu:.8g} T={args.time}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.reset_nfe()
        model.train()
        optimizer.zero_grad()
        logits, initial_logits = model(data.x, position_encoding, return_initial=True)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask],
            data.y.view(-1)[data.train_mask],
        )
        terminal_loss = loss
        anchor_loss = torch.nn.functional.cross_entropy(
            initial_logits[data.train_mask], data.y.view(-1)[data.train_mask],
        ) if args.anchor_loss_weight else loss.new_zeros(())
        loss = terminal_loss + args.anchor_loss_weight * anchor_loss
        loss.backward()
        if not torch.isfinite(loss) or any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise FloatingPointError(f"Non-finite loss/gradient at epoch {epoch}")
        optimizer.step()
        model.reset_nfe()

        model.eval()
        with torch.no_grad():
            scores, candidate_logits, candidate_time = _evaluate_over_horizon(
                model,
                data,
                position_encoding,
                args,
                validation_scores,
                torch,
                return_details=True,
            )
        model.reset_nfe()
        if scores[1] > best_validation:
            best_validation = scores[1]
            best_scores = tuple(scores)
            best_logits = candidate_logits.detach().clone()
            best_epoch, best_time = epoch, candidate_time
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        history.append({"epoch": epoch, "loss": float(loss.detach()), "validation": scores[1],
                        "terminal_loss": float(terminal_loss.detach()),
                        "anchor_loss": float(anchor_loss.detach())})
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={float(loss.detach()):.5f} val={scores[1]:.4f} "
                  f"best_val={best_validation:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_score, validation_score = best_scores
    test_score = None
    if not args.validation_only:
        test_score = evaluate_logits(best_logits, data)[2]
    record = {
        "seed": run_seed, "train": train_score, "validation": validation_score,
        "effective_split_index": effective_split_index,
        "test": test_score, "best_epoch": best_epoch, "selected_time": best_time,
        "seconds": time.perf_counter()-started, "nodes": int(data.num_nodes),
        "split_sizes": {name: int(getattr(data, name+"_mask").sum()) for name in ("train", "val", "test")},
        "split_hash": hashlib.sha256(b"".join(getattr(data, name+"_mask").cpu().numpy().tobytes() for name in ("train", "val", "test"))).hexdigest(),
        "certificate": vars(certificate), "history": history,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "device": str(device), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }
    if args.diagnostics:
        from diagnostics import field_diagnostics
        record["diagnostics"] = field_diagnostics(model, data, position_encoding, best_time)
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / f"{args.dataset}_seed{run_seed}.pt"
        torch.save({"state_dict": best_state, "options": options, "args": vars(args),
                    "seed": run_seed, "best_epoch": best_epoch, "selected_time": best_time,
                    "effective_split_index": effective_split_index,
                    "split_hash": record["split_hash"]}, checkpoint_path)
        record["checkpoint"] = str(checkpoint_path.resolve())
    print(f"train={train_score:.4f} val={validation_score:.4f} "
          + ("test=withheld" if test_score is None else f"test={test_score:.4f}"), flush=True)
    return record


def main(args=None):
    parsed = _apply_dataset_defaults(build_parser().parse_args(args))
    if parsed.show_config:
        print(json.dumps(vars(parsed), indent=2))
        return vars(parsed)

    import numpy as np
    if parsed.runs <= 0:
        raise ValueError("runs must be positive.")
    if parsed.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if parsed.max_nfe <= 0:
        raise ValueError("max_nfe must be positive.")
    if parsed.weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative.")

    heterophilic = {"texas", "wisconsin", "cornell", "film"}
    records = []
    run_seeds = _run_seeds(parsed.seed, parsed.runs)
    for run_index, run_seed in enumerate(run_seeds):
        split_index = parsed.split_index
        if parsed.dataset in heterophilic:
            split_index = (parsed.split_index + run_index) % 10
        records.append(
            _run_once(parsed, run_seed, split_index=split_index)
        )
        if parsed.result_json:
            output = Path(parsed.result_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"args": vars(parsed), "runs": records}, indent=2, allow_nan=False), encoding="utf-8")
    key = "validation" if parsed.validation_only else "test"
    scores = [record[key] for record in records]
    print(f"{key}={np.mean(scores):.4f}+/-{np.std(scores):.4f} ({len(scores)} runs)")
    return scores


if __name__ == "__main__":
    main()
