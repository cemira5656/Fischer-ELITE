from __future__ import annotations
import numpy as np
import pandas as pd

def compute_timing_signals(px: pd.Series, cfg: dict) -> dict:
    px = px.dropna().astype(float)
    close = px

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    last_close = float(close.iloc[-1])
    last_ma50 = float(ma50.iloc[-1])
    last_ma200 = float(ma200.iloc[-1])

    ath_window = close.tail(cfg["ath_lookback_days"])
    ath = float(ath_window.max())
    ath_dates = ath_window[ath_window == ath].index
    last_ath_date = ath_dates.max() if len(ath_dates) else None
    days_since_ath = int((close.index[-1] - last_ath_date).days) if last_ath_date is not None else None

    drawdown = (ath - last_close) / ath if ath > 0 else np.nan
    near_ma50 = abs(last_close - last_ma50) / last_ma50 if last_ma50 > 0 else np.nan

    ma50_above_ma200 = (last_ma50 > last_ma200) if (not np.isnan(last_ma50) and not np.isnan(last_ma200)) else False

    ma50_rising = False
    rd = cfg["ma50_rising_days"]
    if len(ma50.dropna()) > rd:
        ma50_rising = float(ma50.iloc[-1]) > float(ma50.iloc[-(rd + 1)])

    reclaim = False
    look = min(cfg["reclaim_lookback_days"] + 2, len(close))
    if look >= 3:
        recent_close = close.tail(look)
        recent_ma50 = ma50.tail(look)
        cross = (recent_close.shift(1) < recent_ma50.shift(1)) & (recent_close >= recent_ma50)
        reclaim = bool(cross.any())

    ok = True
    reasons = []

    if days_since_ath is None or days_since_ath > cfg["ath_within_days"]:
        ok = False
        reasons.append(f"ATH not within last {cfg['ath_within_days']} days (days_since_ath={days_since_ath})")

    if not (cfg["min_drawdown_from_ath"] <= drawdown <= cfg["max_drawdown_from_ath"]):
        ok = False
        reasons.append(f"Drawdown {drawdown:.1%} not in [{cfg['min_drawdown_from_ath']:.0%}, {cfg['max_drawdown_from_ath']:.0%}]")

    if near_ma50 > cfg["near_ma50_pct"]:
        ok = False
        reasons.append(f"Not near 50DMA (distance {near_ma50:.1%} > {cfg['near_ma50_pct']:.0%})")

    if cfg["require_ma50_above_ma200"] and not ma50_above_ma200:
        ok = False
        reasons.append("MA50 not above MA200")

    if cfg["require_ma50_rising"] and not ma50_rising:
        ok = False
        reasons.append(f"MA50 not rising (vs {rd} days ago)")

    if ok:
        signal = "SETUP"
        why = ["ATH→pullback→near 50DMA setup present"]
        if reclaim:
            signal = "TRIGGER"
            why.append(f"Reclaimed 50DMA within last {cfg['reclaim_lookback_days']} days")
    else:
        signal = "WATCH"
        why = ["Timing not ready"] + reasons

    return {
        "signal": signal,
        "last_close": last_close,
        "ma50": last_ma50,
        "ma200": last_ma200,
        "ath": ath,
        "days_since_ath": days_since_ath,
        "drawdown_from_ath": float(drawdown),
        "near_ma50_pct": float(near_ma50),
        "ma50_above_ma200": bool(ma50_above_ma200),
        "ma50_rising": bool(ma50_rising),
        "reclaim_ma50": bool(reclaim),
        "why": why,
    }
