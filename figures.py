"""
Publication-quality figures for Section 6 of the paper.

Produces vector PDFs with Latin Modern Roman typography to match
the elsarticle paper body. Consistent color palette across figures:
  - Sharikat (proposed compliant architecture): blue (#2a78d6)
  - CFD (conventional baseline):                orange (#eb6834)
  - Grays for reference lines and axis chrome.

Requires results_grid.csv produced by simulate.py, plus fresh
path simulations for the distribution figures.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from bates import preset, simulate_paths
from contracts import (
    CFDParams, SharikaParams,
    cfd_payoff_with_margin_call, sharika_payoff, margin_call_time,
)

# ------------------------------------------------------------------
#  Typography and style
# ------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,       # embed as TrueType (searchable text)
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_SHR = "#2a78d6"     # Sharikat = blue
C_CFD = "#eb6834"     # CFD = orange
C_AXIS = "#52514e"
C_GRID = "#e1e0d9"
C_ANNOT = "#898781"


# ------------------------------------------------------------------
#  Figure 1: Cost-of-carry vs holding horizon (log--log)
# ------------------------------------------------------------------
def figure_1_cost_of_carry(df: pd.DataFrame, out_path: Path) -> None:
    """
    For EUR/USD: total cost of carry as a function of holding horizon,
    at three leverage levels for the CFD, versus the flat Sharikat
    ujrah.  Shows the leverage-driven cost divergence.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sub = df[df["underlying"] == "EURUSD"].copy()

    # Sharikat: flat cost curve (ujrah accrues linearly with horizon)
    shr = sub.drop_duplicates("horizon_days")
    ax.plot(shr["horizon_days"], shr["coc_shr"],
            color=C_SHR, marker="o", markersize=4, linewidth=1.4,
            label="Sharīkat al-ʿinān (any $L$)")

    # CFD at L=5, 10, 30
    line_styles = {5: "--", 10: "-.", 30: ":"}
    for L in [5, 10, 30]:
        c = sub[sub["leverage"] == L].sort_values("horizon_days")
        ax.plot(c["horizon_days"], c["coc_cfd"],
                color=C_CFD, linestyle=line_styles[L], linewidth=1.4,
                marker="s", markersize=3.5,
                label=fr"Leveraged CFD, $L={L}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Holding horizon (trading days, log scale)")
    ax.set_ylabel(r"Expected cost of carry, USD (log scale)")
    ax.set_xticks([1, 5, 21, 63])
    ax.set_xticklabels(["1", "5", "21", "63"])
    ax.grid(True, which="major", color=C_GRID, linewidth=0.4)
    ax.grid(True, which="minor", color=C_GRID, linewidth=0.2,
            alpha=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.tick_params(colors=C_AXIS)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ------------------------------------------------------------------
#  Figure 2: Margin-call probability heatmap
# ------------------------------------------------------------------
def figure_2_margin_call_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """
    Heatmap of P(margin call before horizon) over (leverage, horizon)
    for each of the four underlyings, in a 2x2 panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    underlyings = ["EURUSD", "USDJPY", "AUDUSD", "SPX"]
    horizons = [1, 5, 21, 63]
    leverages = [1, 5, 10, 30]

    from matplotlib.colors import LinearSegmentedColormap
    # Sequential blue ramp from palette.md
    blues = ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blues", blues)

    for i, u in enumerate(underlyings):
        sub = df[df["underlying"] == u]
        grid = np.zeros((len(leverages), len(horizons)))
        for j, L in enumerate(leverages):
            for k, h in enumerate(horizons):
                grid[j, k] = sub[(sub["leverage"] == L) &
                                 (sub["horizon_days"] == h)]["mc_prob"].iloc[0]
        im = axes[i].imshow(grid, cmap=cmap, aspect="auto",
                            origin="lower", vmin=0, vmax=1)
        for j in range(len(leverages)):
            for k in range(len(horizons)):
                v = grid[j, k]
                txt_color = "white" if v > 0.4 else "#0b0b0b"
                axes[i].text(k, j, f"{v*100:.1f}%",
                             ha="center", va="center",
                             color=txt_color, fontsize=7.5)
        axes[i].set_title(u, color=C_AXIS, pad=4)
        axes[i].set_xticks(range(len(horizons)))
        axes[i].set_xticklabels([str(h) for h in horizons])
        axes[i].set_yticks(range(len(leverages)))
        axes[i].set_yticklabels([str(L) for L in leverages])
        axes[i].tick_params(colors=C_AXIS)
        for s in axes[i].spines.values():
            s.set_color(C_AXIS)
            s.set_linewidth(0.4)

    fig.supxlabel(r"Holding horizon (trading days)",
                  fontsize=9, color=C_AXIS, y=0.02)
    fig.supylabel(r"Leverage $L$", fontsize=9, color=C_AXIS, x=0.02)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\mathbb{P}(\tau_{\mathrm{MC}} < t_1)$",
                   fontsize=8, color=C_AXIS)
    cbar.ax.tick_params(colors=C_AXIS, labelsize=7)
    cbar.outline.set_edgecolor(C_AXIS)
    cbar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.10,
                        wspace=0.10, hspace=0.20)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ------------------------------------------------------------------
#  Figure 3: Return distribution CDF comparison
# ------------------------------------------------------------------
def figure_3_return_cdf(out_path: Path,
                        underlying: str = "AUDUSD",
                        horizon_days: int = 21,
                        leverage: int = 10) -> None:
    """
    Empirical CDFs of net client P&L: leveraged CFD versus Sharikat,
    for one representative configuration.
    """
    from simulate import (
        RATES, SPREAD_BPS, BROKER_MARKUP, ALPHA_MAINT,
        CLIENT_CAPITAL, UJRAH_BPS_PER_ANNUM, TRADING_DAYS,
    )
    bates = preset(underlying)
    n_paths = 20_000
    horizon_years = horizon_days / TRADING_DAYS
    n_steps = horizon_days
    dt = horizon_years / n_steps

    S_paths, _ = simulate_paths(bates, n_paths, horizon_years, n_steps,
                                seed=12345)

    r1, r2 = RATES[underlying]
    N = CLIENT_CAPITAL * leverage / bates.S0
    cfd = CFDParams(notional=N, leverage=leverage, direction=1,
                    spread_bps=SPREAD_BPS, alpha_maint=ALPHA_MAINT,
                    r_ccy1=r1, r_ccy2=r2,
                    broker_markup=BROKER_MARKUP)
    _, _, net_cfd = cfd_payoff_with_margin_call(S_paths, cfd, dt)

    ujrah_bps = UJRAH_BPS_PER_ANNUM * horizon_days / TRADING_DAYS
    shr = SharikaParams(K_c=CLIENT_CAPITAL, K_b=0.0,
                        ujrah_bps=ujrah_bps)
    net_shr = sharika_payoff(S_paths, shr)

    # Sort for empirical CDF
    x_cfd = np.sort(net_cfd)
    x_shr = np.sort(net_shr)
    F = np.arange(1, len(x_cfd) + 1) / len(x_cfd)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(x_shr, F, color=C_SHR, linewidth=1.5,
            label="Sharīkat al-ʿinān ($L=1$)")
    ax.plot(x_cfd, F, color=C_CFD, linewidth=1.5,
            label=fr"Leveraged CFD, $L={leverage}$")

    # Reference line at F = 0.5 (median)
    ax.axhline(0.5, color=C_ANNOT, linewidth=0.4, linestyle=":")
    ax.axvline(0, color=C_ANNOT, linewidth=0.4, linestyle=":")

    ax.set_xlabel("Net client P&L, USD")
    ax.set_ylabel(r"Empirical CDF")
    ax.set_xlim([-CLIENT_CAPITAL * 1.05, CLIENT_CAPITAL * 1.0])
    ax.set_ylim([0, 1])
    ax.grid(True, color=C_GRID, linewidth=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.tick_params(colors=C_AXIS)
    for s in ax.spines.values():
        s.set_color(C_AXIS)
    ax.text(0.02, 0.98,
            f"{underlying}, {horizon_days}d horizon\n"
            f"$n=20{{,}}000$ paths",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, color=C_ANNOT)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ------------------------------------------------------------------
#  Figure 4: Left-tail decomposition (jumps vs diffusion)
# ------------------------------------------------------------------
def figure_4_tail_decomposition(out_path: Path,
                                underlying: str = "SPX",
                                horizon_days: int = 21) -> None:
    """
    Left tail (below 5th percentile) of the leveraged CFD P&L
    distribution, decomposed into contributions from the diffusion
    component vs the jump component. Shows that jumps dominate
    extreme losses.
    """
    from simulate import (
        RATES, SPREAD_BPS, BROKER_MARKUP, ALPHA_MAINT,
        CLIENT_CAPITAL, TRADING_DAYS,
    )
    n_paths = 30_000
    horizon_years = horizon_days / TRADING_DAYS
    n_steps = horizon_days
    dt = horizon_years / n_steps
    leverage = 10

    # Case A: full Bates
    bates_full = preset(underlying)
    S_full, _ = simulate_paths(bates_full, n_paths, horizon_years,
                               n_steps, seed=7)
    # Case B: zero-jump (lambda = 0)
    from dataclasses import replace
    bates_nojump = replace(bates_full, lam=0.0)
    S_nojump, _ = simulate_paths(bates_nojump, n_paths, horizon_years,
                                 n_steps, seed=7)

    r1, r2 = RATES[underlying]
    N = CLIENT_CAPITAL * leverage / bates_full.S0
    cfd = CFDParams(notional=N, leverage=leverage, direction=1,
                    spread_bps=SPREAD_BPS, alpha_maint=ALPHA_MAINT,
                    r_ccy1=r1, r_ccy2=r2,
                    broker_markup=BROKER_MARKUP)
    _, _, net_full = cfd_payoff_with_margin_call(S_full, cfd, dt)
    _, _, net_nojump = cfd_payoff_with_margin_call(S_nojump, cfd, dt)

    # Left tail (bottom 20% for visibility)
    q20 = np.quantile(net_full, 0.20)
    tail_full = net_full[net_full <= q20]
    tail_nojump = net_nojump[net_nojump <= q20]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bins = np.linspace(-CLIENT_CAPITAL, q20, 50)
    ax.hist(tail_full, bins=bins, color=C_CFD, alpha=0.55,
            edgecolor="none", label="With jumps ($\\lambda > 0$)")
    ax.hist(tail_nojump, bins=bins, color=C_SHR, alpha=0.55,
            edgecolor="none", label="No jumps ($\\lambda = 0$)")

    ax.set_xlabel("Net CFD client P&L, USD (left tail: bottom 20%)")
    ax.set_ylabel(r"Path count")
    ax.grid(True, color=C_GRID, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.tick_params(colors=C_AXIS)
    for s in ax.spines.values():
        s.set_color(C_AXIS)
    ax.text(0.98, 0.98,
            f"{underlying}, {horizon_days}d, $L={leverage}$\n"
            f"$n=30{{,}}000$ paths each",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=7, color=C_ANNOT)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
def main():
    root = Path(__file__).parent.parent
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "tables" / "results_grid.csv")

    figure_1_cost_of_carry(df, fig_dir / "fig1_cost_of_carry.pdf")
    print("[ok] fig1_cost_of_carry.pdf")

    figure_2_margin_call_heatmap(df, fig_dir / "fig2_margin_call.pdf")
    print("[ok] fig2_margin_call.pdf")

    figure_3_return_cdf(fig_dir / "fig3_return_cdf.pdf")
    print("[ok] fig3_return_cdf.pdf")

    figure_4_tail_decomposition(fig_dir / "fig4_tail_decomposition.pdf")
    print("[ok] fig4_tail_decomposition.pdf")


if __name__ == "__main__":
    main()
