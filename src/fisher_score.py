from __future__ import annotations
import numpy as np
import pandas as pd

def _zscore(series: pd.Series) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def fisher_proxy_score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = df.copy()

    growth_raw = out.get("revenueGrowth", pd.Series(index=out.index, dtype=float))
    if growth_raw.isna().all() and "earningsGrowth" in out.columns:
        growth_raw = out["earningsGrowth"]
    growth = _zscore(growth_raw.fillna(0.0))

    gross = out.get("grossMargins", pd.Series(index=out.index, dtype=float)).fillna(0.0)
    opm   = out.get("operatingMargins", pd.Series(index=out.index, dtype=float)).fillna(0.0)
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

    def pct_rank(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    subs = pd.DataFrame({
        "growth": pct_rank(growth),
        "margins": pct_rank(margins),
        "roic": pct_rank(roic),
        "stability": pct_rank(stability),
        "reinvestment": pct_rank(reinvestment),
        "dilution": pct_rank(dilution),
    }, index=out.index)

    score01 = (
        weights["growth"] * subs["growth"] +
        weights["margins"] * subs["margins"] +
        weights["roic"] * subs["roic"] +
        weights["stability"] * subs["stability"] +
        weights["reinvestment"] * subs["reinvestment"] +
        weights["dilution"] * subs["dilution"]
    )

    out = out.join(subs.add_prefix("sub_"))
    out["fisherScore"] = (100.0 * score01).round(1)
    return out
