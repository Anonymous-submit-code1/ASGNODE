"""Pairwise contraction certificate and historical Table 6 parameter checks."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContinuousFieldCertificate:
    kappa: float
    separation_epsilon: float
    stability_margin: float
    lambda2: float
    lambda_max: float
    rho_bar: float
    mu: float
    contraction_margin: float
    budget_ratio: float


def pairwise_field_certificate(
    *, kappa, separation_epsilon, separation_beta, lambda2, lambda_max, mu=None,
    separation_kernel="boundary",
) -> ContinuousFieldCertificate:
    """Certificate in the degree-weighted norm for reversible RW propagation.

    lambda2/lambda_max belong to I-D^(-1/2) A D^(-1/2), including the
    self-loop degrees used by the actual propagation matrix. beta bounds the
    energy contribution ratio, NOT the unweighted velocity norm ratio Q.
    """
    kappa, epsilon, beta = map(float, (kappa, separation_epsilon, separation_beta))
    lambda2, lambda_max = float(lambda2), float(lambda_max)
    if not all(math.isfinite(v) for v in (kappa, epsilon, beta, lambda2, lambda_max)):
        raise ValueError("Certificate parameters must be finite.")
    if kappa <= 0 or epsilon <= 0:
        raise ValueError("kappa and separation_epsilon must be positive.")
    if not 0 <= beta < 1:
        raise ValueError("separation_beta must lie in [0, 1).")
    if lambda2 <= 0 or lambda_max < lambda2:
        raise ValueError("A connected graph with a valid normalized spectrum is required.")
    mu = beta * epsilon * kappa if mu is None else float(mu)
    if not math.isfinite(mu) or not 0 <= mu < epsilon * kappa:
        raise ValueError("Certified separation requires 0 <= mu < epsilon * kappa.")
    ratio = mu / (epsilon * kappa)
    if separation_kernel not in {"inverse", "boundary", "linear"}:
        raise ValueError("Unknown separation kernel.")
    return ContinuousFieldCertificate(
        kappa=kappa, separation_epsilon=epsilon, stability_margin=1-ratio,
        lambda2=lambda2, lambda_max=lambda_max,
        rho_bar=lambda_max/epsilon,
        mu=mu, contraction_margin=(kappa-mu/epsilon)*lambda2,
        budget_ratio=ratio,
    )


def continuous_field_certificate(
    *,
    kappa: float,
    separation_epsilon: float,
    stability_margin: float,
    lambda2: float,
    lambda_max: float = None,
    rho_bar: float = None,
    mu: float = None,
) -> ContinuousFieldCertificate:
    """Validate explicit values, deriving omitted certificate quantities."""
    values = {
        "kappa": float(kappa),
        "separation_epsilon": float(separation_epsilon),
        "stability_margin": float(stability_margin),
        "lambda2": float(lambda2),
    }

    if values["kappa"] <= 0.0:
        raise ValueError("kappa must be positive.")
    if values["separation_epsilon"] <= 0.0:
        raise ValueError("separation_epsilon must be positive.")
    if not 0.0 < values["stability_margin"] < 1.0:
        raise ValueError("stability_margin must lie strictly between 0 and 1.")
    if values["lambda2"] <= 0.0:
        raise ValueError("lambda2 must be positive; the graph must be connected.")
    if rho_bar is None:
        if lambda_max is None:
            raise ValueError("Either rho_bar or lambda_max must be provided.")
        lambda_max = float(lambda_max)
        if lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive.")
        rho_bar = lambda_max / values["separation_epsilon"]
    else:
        rho_bar = float(rho_bar)
        if rho_bar <= 0.0:
            raise ValueError("rho_bar must be positive.")
        lambda_max = rho_bar * values["separation_epsilon"]

    if mu is None:
        mu = (
            (1.0 - values["stability_margin"])
            * values["kappa"]
            * values["lambda2"]
            / rho_bar
        )
    else:
        mu = float(mu)
        if mu <= 0.0:
            raise ValueError("mu must be positive.")

    attractive_contraction = values["kappa"] * values["lambda2"]
    separation_bound = mu * rho_bar
    if separation_bound >= attractive_contraction:
        raise ValueError(
            "The parameters violate the legacy sufficient certificate: "
            "mu * rho_bar must be smaller than kappa * lambda2."
        )

    return ContinuousFieldCertificate(
        **values,
        lambda_max=lambda_max,
        rho_bar=rho_bar,
        mu=mu,
        contraction_margin=attractive_contraction - separation_bound,
        budget_ratio=separation_bound / attractive_contraction,
    )
