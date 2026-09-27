"""
Bates (1996) jump-diffusion with stochastic volatility.

Implements the coupled SDE system used in Assumption 3.1 of the paper:

    dS_t / S_{t-} = (mu - lambda * kappa) dt
                    + sqrt(v_t) dW_t^S
                    + (J_t - 1) dN_t

    dv_t = kappa_v (theta - v_t) dt
           + sigma_v sqrt(v_t) dW_t^v

with Corr(dW^S, dW^v) = rho, N_t Poisson(lambda),
ln J_t ~ Normal(mu_J, sigma_J^2), and
kappa = E[J - 1] = exp(mu_J + sigma_J^2/2) - 1.

Uses a Milstein-Euler discretization for v with full-truncation to
maintain positivity (Lord, Koekkoek, van Dijk 2010) and log-Euler
for S. Vectorised across paths.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BatesParams:
    """Bates model parameters. All rates in annualised, continuous
    compounding conventions."""
    S0: float          # initial spot
    mu: float          # drift (r - q for FX or dividends)
    v0: float          # initial variance
    kappa_v: float     # variance mean-reversion speed
    theta: float       # long-run variance
    sigma_v: float     # vol of vol
    rho: float         # correlation between W^S and W^v (in [-1, 1])
    lam: float         # jump intensity per year
    mu_J: float        # mean of log-jump size
    sigma_J: float     # std of log-jump size

    @property
    def kappa(self) -> float:
        """Mean relative jump size E[J - 1]."""
        return np.exp(self.mu_J + 0.5 * self.sigma_J ** 2) - 1.0

    def check_feller(self) -> bool:
        """Feller condition 2 kappa_v theta >= sigma_v^2."""
        return 2.0 * self.kappa_v * self.theta >= self.sigma_v ** 2


def simulate_paths(
    params: BatesParams,
    n_paths: int,
    horizon_years: float,
    n_steps: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_paths sample paths of (S_t, v_t) over [0, horizon_years]
    at n_steps time steps.

    Returns
    -------
    S : ndarray of shape (n_paths, n_steps + 1)
        Spot price paths.
    v : ndarray of shape (n_paths, n_steps + 1)
        Variance paths.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = horizon_years / n_steps
    sqrt_dt = np.sqrt(dt)

    # Storage
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = params.S0
    v[:, 0] = params.v0

    # Cholesky for correlated Brownians
    rho = params.rho
    corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(corr_matrix)

    for t in range(n_steps):
        # Correlated standard normals
        Z = rng.standard_normal((n_paths, 2))
        dW = Z @ L.T * sqrt_dt  # shape (n_paths, 2)
        dW_S = dW[:, 0]
        dW_v = dW[:, 1]

        # Variance update: full-truncation Euler (positivity)
        v_prev = np.maximum(v[:, t], 0.0)
        v_new = (
            v[:, t]
            + params.kappa_v * (params.theta - v_prev) * dt
            + params.sigma_v * np.sqrt(v_prev) * dW_v
        )
        v[:, t + 1] = np.maximum(v_new, 0.0)

        # Jump component
        n_jumps = rng.poisson(params.lam * dt, size=n_paths)
        log_jump = np.where(
            n_jumps > 0,
            rng.normal(params.mu_J, params.sigma_J, size=n_paths)
            * np.sqrt(np.maximum(n_jumps, 1)),
            0.0,
        )
        # Compensated drift adjustment
        drift = (params.mu - params.lam * params.kappa) * dt

        # Log-Euler for S with correlated Brownian
        log_return = (
            drift
            - 0.5 * v_prev * dt
            + np.sqrt(v_prev) * dW_S
            + log_jump
        )
        S[:, t + 1] = S[:, t] * np.exp(log_return)

    return S, v


# ------------------------------------------------------------------
#  Preset calibrations for the four underlyings used in Section 6.
#  Parameters are consistent with published FX and equity vol
#  studies (Andersen & Benzoni 2009, Bates 1996, empirical option
#  implied vol surfaces). These are illustrative calibrations used
#  for the paper's Monte Carlo comparison; they are not fit to any
#  proprietary dataset.
# ------------------------------------------------------------------

def preset(name: str) -> BatesParams:
    """Return a preset Bates calibration for a named underlying."""
    presets = {
        "EURUSD": BatesParams(
            S0=1.10, mu=0.0, v0=0.0064, kappa_v=2.0, theta=0.0064,
            sigma_v=0.15, rho=-0.10, lam=1.5,
            mu_J=-0.002, sigma_J=0.012,
        ),
        "USDJPY": BatesParams(
            S0=150.0, mu=0.0, v0=0.0081, kappa_v=1.8, theta=0.0081,
            sigma_v=0.18, rho=0.05, lam=2.0,
            mu_J=0.001, sigma_J=0.015,
        ),
        "AUDUSD": BatesParams(
            S0=0.66, mu=0.0, v0=0.0121, kappa_v=2.2, theta=0.0121,
            sigma_v=0.22, rho=-0.20, lam=2.5,
            mu_J=-0.003, sigma_J=0.018,
        ),
        "SPX": BatesParams(
            S0=5000.0, mu=0.05, v0=0.0324, kappa_v=3.0, theta=0.0324,
            sigma_v=0.40, rho=-0.65, lam=3.0,
            mu_J=-0.015, sigma_J=0.035,
        ),
    }
    return presets[name]
