"""
Master simulation runner for the Section 6 empirical comparison.

Produces, for each (underlying, horizon, leverage) combination:
  - Cost-of-carry statistics for CFD and Sharikat
  - Margin-call trigger frequency for CFD
  - Return distribution moments and tail metrics
  - Bootstrap confidence intervals

Persists results as CSV tables consumed by figures.py and by the
LaTeX Section 6.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from bates import BatesParams, preset, simulate_paths
from contracts import (
    CFDParams, SharikaParams,
    cfd_payoff_with_margin_call, sharika_payoff, margin_call_time,
)
from metrics import summarise, bootstrap_ci, var_cvar, ks_test


# ------------------------------------------------------------------
#  Simulation configuration
# ------------------------------------------------------------------
UNDERLYINGS = ["EURUSD", "USDJPY", "AUDUSD", "SPX"]
HORIZONS_DAYS = [1, 5, 21, 63]  # 1D, 1W, 1M, 3M
LEVERAGES = [1, 5, 10, 30]
N_PATHS = 10_000
N_STEPS_PER_DAY = 1
TRADING_DAYS = 252
SEED_BASE = 20260928

# Interbank rate assumptions (illustrative; typical 2025 levels)
RATES = {
    # (base_ccy_rate, quote_ccy_rate)
    "EURUSD": (0.030, 0.045),
    "USDJPY": (0.045, 0.005),
    "AUDUSD": (0.038, 0.045),
    "SPX":    (0.000, 0.045),  # for SPX treat "financing" as USD funding cost
}

# Contract parameter defaults
SPREAD_BPS = 2.0
BROKER_MARKUP = 0.010   # 100 bps p.a. on top of interbank differential
ALPHA_MAINT = 0.50
CLIENT_CAPITAL = 10_000.0     # USD
UJRAH_BPS_PER_ANNUM = 100.0   # broker structuring fee, 1% p.a. of K_c


def run_cell(underlying: str, horizon_days: int, leverage: int,
             n_paths: int = N_PATHS, seed: int | None = None) -> dict:
    """Run one (underlying, horizon, leverage) simulation cell."""
    bates = preset(underlying)
    n_steps = horizon_days * N_STEPS_PER_DAY
    horizon_years = horizon_days / TRADING_DAYS
    dt = horizon_years / n_steps

    S_paths, v_paths = simulate_paths(
        bates, n_paths=n_paths, horizon_years=horizon_years,
        n_steps=n_steps, seed=seed,
    )

    # ---- CFD leg ----
    r1, r2 = RATES[underlying]
    N = CLIENT_CAPITAL * leverage / bates.S0
    cfd_params = CFDParams(
        notional=N, leverage=leverage, direction=1,
        spread_bps=SPREAD_BPS, alpha_maint=ALPHA_MAINT,
        r_ccy1=r1, r_ccy2=r2, broker_markup=BROKER_MARKUP,
    )
    gross_cfd, financing_cfd, net_cfd = cfd_payoff_with_margin_call(
        S_paths, cfd_params, dt
    )
    tau_mc = margin_call_time(S_paths, cfd_params, dt)
    mc_prob = float((tau_mc != -1).mean())

    # ---- Sharikat leg (no leverage; capital = CLIENT_CAPITAL) ----
    ujrah_bps = UJRAH_BPS_PER_ANNUM * horizon_days / TRADING_DAYS
    shr_params = SharikaParams(
        K_c=CLIENT_CAPITAL, K_b=0.0, ujrah_bps=ujrah_bps,
    )
    net_shr = sharika_payoff(S_paths, shr_params)

    # ---- Cost of carry (deterministic component, expected value) ----
    coc_cfd = float(np.mean(financing_cfd)) + \
              CLIENT_CAPITAL * leverage * 2 * SPREAD_BPS * 1e-4
    coc_shr = float(CLIENT_CAPITAL * ujrah_bps * 1e-4)

    # ---- Summaries ----
    summary_cfd = summarise(net_cfd)
    summary_shr = summarise(net_shr)

    # ---- KS test on distribution equivalence ----
    ks_stat, ks_p = ks_test(net_cfd, net_shr)

    # ---- Bootstrap CI on mean net payoff ----
    _, cfd_lo, cfd_hi = bootstrap_ci(net_cfd, np.mean, n_boot=500)
    _, shr_lo, shr_hi = bootstrap_ci(net_shr, np.mean, n_boot=500)

    return {
        "underlying": underlying,
        "horizon_days": horizon_days,
        "leverage": leverage,
        "mc_prob": mc_prob,
        "coc_cfd": coc_cfd,
        "coc_shr": coc_shr,
        "mean_cfd": summary_cfd["mean"],
        "mean_cfd_lo": cfd_lo, "mean_cfd_hi": cfd_hi,
        "mean_shr": summary_shr["mean"],
        "mean_shr_lo": shr_lo, "mean_shr_hi": shr_hi,
        "std_cfd": summary_cfd["std"], "std_shr": summary_shr["std"],
        "skew_cfd": summary_cfd["skew"], "skew_shr": summary_shr["skew"],
        "excess_kurt_cfd": summary_cfd["excess_kurt"],
        "excess_kurt_shr": summary_shr["excess_kurt"],
        "var95_cfd": summary_cfd["var_95"],
        "var95_shr": summary_shr["var_95"],
        "cvar95_cfd": summary_cfd["cvar_95"],
        "cvar95_shr": summary_shr["cvar_95"],
        "min_cfd": summary_cfd["min"], "min_shr": summary_shr["min"],
        "max_cfd": summary_cfd["max"], "max_shr": summary_shr["max"],
        "p_loss_cfd": summary_cfd["p_loss"],
        "p_loss_shr": summary_shr["p_loss"],
        "ks_stat": ks_stat, "ks_p": ks_p,
        "sharpe_cfd": summary_cfd["sharpe"],
        "sharpe_shr": summary_shr["sharpe"],
        "sortino_cfd": summary_cfd["sortino"],
        "sortino_shr": summary_shr["sortino"],
    }


def run_all(out_dir: Path) -> pd.DataFrame:
    """Run the full (underlying, horizon, leverage) grid."""
    records = []
    counter = 0
    for underlying in UNDERLYINGS:
        for horizon_days in HORIZONS_DAYS:
            for leverage in LEVERAGES:
                seed = SEED_BASE + counter
                rec = run_cell(underlying, horizon_days, leverage,
                               seed=seed)
                records.append(rec)
                counter += 1
                print(f"[done] {underlying:8s} h={horizon_days:>3d}d "
                      f"L={leverage:>3d}  MC={rec['mc_prob']:.3f}  "
                      f"KS={rec['ks_stat']:.3f}")
    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_grid.csv", index=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "n_paths": N_PATHS,
            "spread_bps": SPREAD_BPS,
            "broker_markup": BROKER_MARKUP,
            "alpha_maint": ALPHA_MAINT,
            "client_capital": CLIENT_CAPITAL,
            "ujrah_bps_per_annum": UJRAH_BPS_PER_ANNUM,
            "seed_base": SEED_BASE,
        }, f, indent=2)
    return df


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "tables"
    df = run_all(out)
    print("\nGrid summary saved to", out / "results_grid.csv")
    print(df.head())
