"""
Contract mechanics for the retail CFD and the Sharikat al-'Inan
market-access architectures analysed in the paper.

Implements:
  - CFD payoff, Tom-Next financing, margin dynamics, margin-call
    stopping time (Section 3 of the paper);
  - Sharikat al-'Inan pooled-equity payoff (Section 5.4);
  - Broker revenue decomposition (Section 3.5).

All functions operate on ndarray of simulated price paths
returned by src.bates.simulate_paths().
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ------------------------------------------------------------------
#  CFD contract
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CFDParams:
    """CFD contract specification following Definition 3.1."""
    notional: float       # N: units of the underlying
    leverage: float       # L
    direction: int        # +1 long, -1 short
    spread_bps: float     # half-spread in basis points on entry+exit
    alpha_maint: float    # maintenance margin threshold coefficient
    r_ccy1: float         # base-currency overnight rate (annualised)
    r_ccy2: float         # quote-currency overnight rate (annualised)
    broker_markup: float  # broker's markup m over interbank differential


def cfd_payoff(
    S_paths: np.ndarray,
    params: CFDParams,
    dt: float,
    t_exit_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CFD gross and net payoff along each simulated path.

    Returns
    -------
    gross : ndarray of shape (n_paths,)
        Gross payoff: delta * N * (S_{t_1} - S_{t_0})
    financing : ndarray of shape (n_paths,)
        Accumulated Tom-Next swap over the holding period.
    net : ndarray of shape (n_paths,)
        Net payoff = gross - financing - transaction costs.
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    if t_exit_idx is None:
        t_exit_idx = n_steps_plus_1 - 1

    S0 = S_paths[:, 0]
    S1 = S_paths[:, t_exit_idx]

    # Gross payoff
    gross = params.direction * params.notional * (S1 - S0)

    # Tom-Next financing (equation 3.9): F_s = delta * N * S_s *
    # (r_ccy2 - r_ccy1 + m)
    # Integrated: sum over days
    # For a long position the client pays if r_ccy2 > r_ccy1 + markup on
    # the short-currency leg; sign is set by direction and the
    # differential.
    diff = (params.r_ccy2 - params.r_ccy1)
    S_avg = S_paths[:, : t_exit_idx + 1].mean(axis=1)
    holding_years = t_exit_idx * dt
    # Financing charge (positive = paid by client)
    financing = (
        params.direction
        * params.notional
        * S_avg
        * (diff + params.broker_markup)
        * holding_years
    )
    # For a short position the sign of the differential flips; the
    # markup remains a cost. Handle with explicit direction * diff and
    # add markup as positive-costing to client:
    financing = (
        params.direction * params.notional * S_avg * diff * holding_years
        + params.notional * S_avg * params.broker_markup * holding_years
    )

    # Transaction cost: half-spread on entry + half-spread on exit
    txn_cost = (
        params.notional
        * (S0 + S1) / 2
        * 2 * params.spread_bps * 1e-4
    )

    net = gross - financing - txn_cost
    return gross, financing, net


def margin_call_time(
    S_paths: np.ndarray,
    params: CFDParams,
    dt: float,
) -> np.ndarray:
    """
    Return the first time index at which each path triggers a margin
    call, or -1 if no margin call over the horizon.

    Uses the barrier (equation 3.15):
        S_t <= S_0 * (1 - (1 - alpha) / L)   for a long position
        S_t >= S_0 * (1 + (1 - alpha) / L)   for a short position
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    S0 = S_paths[:, 0:1]
    threshold = (1.0 - params.alpha_maint) / params.leverage
    if params.direction == 1:
        # Long: hit when price drops
        barrier = S0 * (1.0 - threshold)
        hit = S_paths <= barrier
    else:
        # Short: hit when price rises
        barrier = S0 * (1.0 + threshold)
        hit = S_paths >= barrier

    hit[:, 0] = False  # exclude t = 0
    # First-hit index; -1 if never hit
    tau = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    return tau


def cfd_payoff_with_margin_call(
    S_paths: np.ndarray,
    params: CFDParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full CFD payoff accounting for margin-call forced liquidation.

    Loss is capped at the initial margin M_0 = N * S_0 / L for
    margin-called paths (assumes no negative-balance protection is
    contractually required beyond regulatory floor).
    """
    n_paths = S_paths.shape[0]
    tau = margin_call_time(S_paths, params, dt)
    S0 = S_paths[:, 0]

    # For paths not margin-called: use horizon exit
    horizon_idx = S_paths.shape[1] - 1
    exit_idx = np.where(tau == -1, horizon_idx, tau)

    S_exit = S_paths[np.arange(n_paths), exit_idx]

    # Gross P&L to exit
    gross = params.direction * params.notional * (S_exit - S0)

    # Financing accrued to exit
    exit_holding_years = exit_idx * dt
    diff = params.r_ccy2 - params.r_ccy1
    S_avg_to_exit = np.array([
        S_paths[i, : exit_idx[i] + 1].mean() for i in range(n_paths)
    ])
    financing = (
        params.direction * params.notional * S_avg_to_exit * diff
        * exit_holding_years
        + params.notional * S_avg_to_exit * params.broker_markup
        * exit_holding_years
    )
    # Transaction cost
    txn_cost = (
        params.notional * (S0 + S_exit) / 2
        * 2 * params.spread_bps * 1e-4
    )

    # Cap loss at initial margin M_0 = N * S_0 / L
    M0 = params.notional * S0 / params.leverage
    net = gross - financing - txn_cost
    net = np.maximum(net, -M0)  # negative-balance protection

    return gross, financing, net


# ------------------------------------------------------------------
#  Sharikat al-'Inan contract
# ------------------------------------------------------------------

@dataclass(frozen=True)
class SharikaParams:
    """Sharikat al-'Inan specification following Definition 5.4."""
    K_c: float          # client capital contribution
    K_b: float          # broker capital contribution
    ujrah_bps: float    # broker structuring fee (basis points of K_c)


def sharika_payoff(
    S_paths: np.ndarray,
    params: SharikaParams,
    t_exit_idx: int | None = None,
) -> np.ndarray:
    """
    Compute the client's Sharikat al-'Inan net payoff per
    Proposition 5.5:
        Pi_c = K_c * (S_{t_1} / S_{t_0} - 1) - u_c

    (Under proportional profit-and-loss sharing pi_c = K_c / (K_c + K_b).)
    """
    if t_exit_idx is None:
        t_exit_idx = S_paths.shape[1] - 1
    S0 = S_paths[:, 0]
    S1 = S_paths[:, t_exit_idx]

    ujrah = params.K_c * params.ujrah_bps * 1e-4
    payoff = params.K_c * (S1 / S0 - 1.0) - ujrah
    return payoff
