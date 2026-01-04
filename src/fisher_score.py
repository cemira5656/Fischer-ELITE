from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def _pct_rank(s: pd.Series) -> pd.Series:
    # rank(pct=True) returns 0..1
    return s.rank(pct=True)


def fisher_proxy_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Builds a 0-100 FisherProxy score from percentile-ranked sub-scores.

    New (optional) momentum inputs (computed upstream in run_weekly.py):
      - ps_rank: 0..100 percentile of 12m return across universe
      - rs_rank: 0..100 percentile of (12m return - SPY 12m return) across universe
      - ret_12m: raw 12m return (fallback if ps_rank missing)
      - rs_12m: raw relative strength vs SPY (fallback if rs_rank missing)

    New (optional) growth fallback input:
      - revenueGrowth_filled: revenueGrowth where available, else momentum proxy (0..1)
    """
    out = df.copy()

    # ---------------------------
    # GROWTH: revenueGrowth_filled -> revenueGrowth -> earningsGrowth -> ps_rank/ret_12m fallback
    # ---------------------------
    growth_raw = out.get("revenueGrowth_filled")
    if growth_raw is None:
        growth_raw = out.get("revenueGrowth")

    if growth_raw is None:
        growth_raw = pd.Series(index=out.index, dtype=float)

    # If everything missing, fall back to earningsGrowth
    if growth_raw.isna().all() and "earningsGrowth" in out.columns:
        growth_raw = out["earningsGrowth"]

    # If still missing, fall back to momentum proxy:
    # prefer ps_rank (0..100) -> convert to 0..1, else ret_12m
    if growth_raw.isna().all():
        if "ps_rank" in out.columns:
            growth_raw = (out["ps_rank"] / 100.0).astype(float)
        elif "ret_12m" in out.columns:
            growth_raw = out["ret_12m"].astype(float)

    growth = _zscore(growth_raw.fillna(0.0))

    # ---------------------------
    # QUALITY: margins, roic, stability, reinvestment, dilution (same as you had)
    # ---------------------------
    gross = out.get("grossMargins", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    opm = out.get("operatingMargins", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    margins = _zscore(0.6 * gross + 0.4 * opm)

    roe = out.get("returnOnEquity", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    roa = out.get("returnOnAssets", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    roic = _zscore(0.7 * roe + 0.3 * roa)

    vol = out.get("volatility1y", pd.Series(index=out.index, dtype=float)).replace(0, np.nan)
    stability = _zscore((1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0))

    rnd = out.get("rndIntensity", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    reinvestment = _zscore(rnd)

    dilution_raw = out.get("sharesChange3y", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    dilution = _zscore(-dilution_raw)

    # ---------------------------
    # MOMENTUM: RS rank + Price strength rank
    # We treat rs_rank/ps_rank as already "rank-like" inputs (0..100).
    # We'll convert them into 0..1 percentiles and percentile-rank them again for robustness.
    # ---------------------------
    rs_rank_raw = out.get("rs_rank")
    ps_rank_raw = out.get("ps_rank")

    # If rs_rank missing, try rs_12m (raw), else zeros
    if rs_rank_raw is None:
        if "rs_12m" in out.columns:
            rs_rank_raw = out["rs_12m"].astype(float)
        else:
            rs_rank_raw = pd.Series(0.0, index=out.index)

    # If ps_rank missing, try ret_12m (raw), else zeros
    if ps_rank_raw is None:
        if "ret_12m" in out.columns:
            ps_rank_raw = out["ret_12m"].astype(float)
        else:
            ps_rank_raw = pd.Series(0.0, index=out.index)

    # If they are 0..100, map to 0..1, otherwise treat as raw and zscore then rank
    def normalize_rankish(x: pd.Series) -> pd.Series:
        x = x.replace([np.inf, -np.inf], np.nan).astype(float)
        # heuristic: if max > 3, likely percent (0..100) or raw returns; handle both safely
        if np.nanmax(x.values) > 3.0:
            # could be 0..100 ranks OR raw returns; distinguish by typical range
            # ranks usually within [0,100]; clamp if so
            if np.nanmax(x.values) <= 100.0 and np.nanmin(x.values) >= 0.0:
                x01 = (x / 100.0).clip(0.0, 1.0)
                return _pct_rank(x01.fillna(0.0))
            # raw returns: zscore then rank
            return _pct_rank(_zscore(x.fillna(0.0)))
        else:
            # small-magnitude: probably raw returns already
            return _pct_rank(_zscore(x.fillna(0.0)))

    rs = normalize_rankish(rs_rank_raw)
    ps = normalize_rankish(ps_rank_raw)

    # ---------------------------
    # Build sub-score dataframe (all 0..1)
    # ---------------------------
    subs = pd.DataFrame(
        {
            "growth": _pct_rank(growth),
            "margins": _pct_rank(margins),
            "roic": _pct_rank(roic),
            "stability": _pct_rank(stability),
            "reinvestment": _pct_rank(reinvestment),
            "dilution": _pct_rank(dilution),
            "rs": rs,
            "ps": ps,
        },
        index=out.index,
    )

    # ---------------------------
    # Weighted sum (weights optional; default 0 if missing)
    # ---------------------------
    def w(name: str, default: float = 0.0) -> float:
        try:
            return float(weights.get(name, default))
        except Exception:
            return default

    score01 = (
        w("growth") * subs["growth"]
        + w("margins") * subs["margins"]
        + w("roic") * subs["roic"]
        + w("stability") * subs["stability"]
        + w("reinvestment") * subs["reinvestment"]
        + w("dilution") * subs["dilution"]
        + w("rs") * subs["rs"]
        + w("ps") * subs["ps"]
    )

    out = out.join(subs.add_prefix("sub_"))
    out["fisherScore"] = (100.0 * score01).round(1)
    return out
