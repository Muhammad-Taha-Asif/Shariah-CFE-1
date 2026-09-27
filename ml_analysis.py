"""
Section 6.4: Gradient-boosted classifier for margin-call prediction.

Trains an XGBoost classifier on path-level features to predict
margin-call events, then extracts feature importances to characterize
which price-path features drive CFD tail losses.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from bates import preset, simulate_paths
from contracts import CFDParams, margin_call_time


def build_features(S_paths: np.ndarray, v_paths: np.ndarray) -> pd.DataFrame:
    """
    Path-level features for margin-call prediction.
    Uses only information from the first day of each path so the
    classifier answers 'given the first-day dynamics, what is the
    margin-call probability over the remaining horizon?'
    """
    # First day = first 1 step (index 0 -> 1)
    log_return_d1 = np.log(S_paths[:, 1] / S_paths[:, 0])
    v_d1 = v_paths[:, 1]
    # First-day intraday range proxy (approximated as |log return|)
    range_d1 = np.abs(log_return_d1)
    # Longer-lookback features from days 1-3
    log_return_3d = np.log(S_paths[:, min(3, S_paths.shape[1]-1)]
                           / S_paths[:, 0])
    v_avg_3d = v_paths[:, :min(4, v_paths.shape[1])].mean(axis=1)
    # Realised variance over first 3 days
    log_incs = np.diff(np.log(S_paths[:, :min(4, S_paths.shape[1])]),
                       axis=1)
    rv_3d = (log_incs ** 2).sum(axis=1)

    return pd.DataFrame({
        "log_ret_d1": log_return_d1,
        "abs_ret_d1": range_d1,
        "v_d1": v_d1,
        "log_ret_3d": log_return_3d,
        "v_avg_3d": v_avg_3d,
        "rv_3d": rv_3d,
    })


def run_ml_analysis(out_path: Path) -> dict:
    """Train the classifier and save the feature-importance figure."""
    from simulate import (RATES, SPREAD_BPS, BROKER_MARKUP,
                          ALPHA_MAINT, CLIENT_CAPITAL, TRADING_DAYS)

    underlying = "SPX"
    horizon_days = 21
    leverage = 10
    n_paths = 30_000

    bates = preset(underlying)
    horizon_years = horizon_days / TRADING_DAYS
    dt = horizon_years / horizon_days
    S_paths, v_paths = simulate_paths(bates, n_paths, horizon_years,
                                      horizon_days, seed=99)

    r1, r2 = RATES[underlying]
    N = CLIENT_CAPITAL * leverage / bates.S0
    cfd = CFDParams(notional=N, leverage=leverage, direction=1,
                    spread_bps=SPREAD_BPS, alpha_maint=ALPHA_MAINT,
                    r_ccy1=r1, r_ccy2=r2,
                    broker_markup=BROKER_MARKUP)
    tau = margin_call_time(S_paths, cfd, dt)
    y = (tau != -1).astype(int)

    X = build_features(S_paths, v_paths)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        use_label_encoder=False, random_state=42,
    )
    clf.fit(X_train, y_train)
    p_hat = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_hat)

    importances = pd.Series(clf.feature_importances_,
                            index=X.columns).sort_values()

    # ---- Figure ----
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    C_SHR = "#2a78d6"
    C_AXIS = "#52514e"

    labels_pretty = {
        "log_ret_d1":  "Day-1 log return",
        "abs_ret_d1":  "Day-1 |log return|",
        "v_d1":        "Day-1 variance $v_1$",
        "log_ret_3d":  "3-day log return",
        "v_avg_3d":    "3-day mean variance",
        "rv_3d":       "3-day realised variance",
    }
    display_labels = [labels_pretty[k] for k in importances.index]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    y_pos = np.arange(len(importances))
    ax.barh(y_pos, importances.values, color=C_SHR, height=0.55,
            edgecolor="none")
    for i, v in enumerate(importances.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center",
                fontsize=7.5, color=C_AXIS)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_xlim([0, importances.max() * 1.20])
    ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.4)
    ax.tick_params(colors=C_AXIS)
    for s in ax.spines.values():
        s.set_color(C_AXIS)
    ax.text(0.98, 0.02,
            f"SPX, 21d, $L=10$; test AUC $={auc:.3f}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#898781")

    fig.savefig(out_path, format="pdf", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)

    return {"auc": float(auc), "n_features": int(len(X.columns)),
            "class_balance": float(y.mean())}


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent.parent / "figures" / "fig5_feature_importance.pdf"
    result = run_ml_analysis(out)
    print(f"ML analysis: {result}")
