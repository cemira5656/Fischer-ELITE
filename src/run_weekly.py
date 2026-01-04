from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from universe import fetch_us_common_stock_symbols
from fisher_score import fisher_proxy_score
from timing import compute_timing_signals

def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())

def dollar_vol(prices: pd.Series, volume: pd.Series, lookback: int = 20) -> float:
    df = pd.DataFrame({"p": prices, "v": volume}).dropna()
    if df.empty:
        return np.nan
    dv = (df["p"] * df["v"]).tail(lookback).mean()
    return float(dv)

def get_fast_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            fi = tk.fast_info
            info = getattr(tk, "info", {}) or {}

            market_cap = fi.get("market_cap") or info.get("marketCap")
            hist = tk.history(period="6mo", interval="1d", auto_adjust=False)
            if hist.empty:
                continue

            close = hist["Close"]
            vol = hist["Volume"]
            avg_dv = dollar_vol(close, vol, lookback=20)

            rets = close.pct_change().dropna()
            vol1y = float(rets.std() * np.sqrt(252)) if len(rets) > 20 else np.nan

            rows.append({
                "ticker": t,
                "marketCap": float(market_cap) if market_cap else np.nan,
                "avgDollarVol": avg_dv,
                "volatility1y": vol1y,
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "grossMargins": info.get("grossMargins"),
                "operatingMargins": info.get("operatingMargins"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                # placeholders; can be upgraded with a fundamentals API later
                "rndIntensity": np.nan,
                "sharesChange3y": np.nan,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows).set_index("ticker")
    return df

def load_prices_for_timing(tickers: list[str], history_days: int) -> dict[str, pd.Series]:
    px = {}
    data = yf.download(
        tickers=tickers,
        period=f"{history_days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True
    )
    for t in tickers:
        try:
            s = data[(t, "Adj Close")] if (t, "Adj Close") in data.columns else data[(t, "Close")]
            px[t] = s.dropna()
        except Exception:
            continue
    return px

def load_elite_state(path="state/elite.json"):
    p = Path(path)
    if not p.exists():
        return {"as_of": None, "elite": [], "history": {}}
    return json.loads(p.read_text())

def save_elite_state(state, path="state/elite.json"):
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2))

def update_elite(state, ranked, enter_weeks=2, exit_weeks=2, elite_cap=10):
    top10 = set(ranked[:10])
    top20 = set(ranked[:20])

    hist = state.get("history", {})
    elite = set(state.get("elite", []))

    for t in ranked[:300]:
        h = hist.get(t, {"in_top10_streak": 0, "out_top20_streak": 0})
        if t in top10:
            h["in_top10_streak"] += 1
            h["out_top20_streak"] = 0
        elif t not in top20:
            h["out_top20_streak"] += 1
            h["in_top10_streak"] = 0
        else:
            h["in_top10_streak"] = 0
            h["out_top20_streak"] = 0
        hist[t] = h

    for t, h in hist.items():
        if h["in_top10_streak"] >= enter_weeks:
            elite.add(t)

    for t, h in list(hist.items()):
        if t in elite and h["out_top20_streak"] >= exit_weeks:
            elite.remove(t)

    elite_sorted = [t for t in ranked if t in elite][:elite_cap]

    state["elite"] = elite_sorted
    state["history"] = hist
    state["as_of"] = str(date.today())
    return state

def main():
    settings = load_yaml("config/settings.yml")

    syms = fetch_us_common_stock_symbols(max_symbols=settings["universe"]["max_symbols"])

    base = get_fast_fundamentals(syms)

    u = settings["universe"]
    base = base.dropna(subset=["marketCap", "avgDollarVol"], how="any")
    base = base[
        (base["marketCap"] >= u["min_market_cap"]) &
        (base["avgDollarVol"] >= u["min_avg_dollar_vol"])
    ].copy()

    base = base.sort_values("marketCap", ascending=False).head(1200)

    scored = fisher_proxy_score(base, settings["scoring"]["weights"])
    scored = scored.sort_values("fisherScore", ascending=False)

    ranked = scored.index.tolist()

    # absolute top 10 (elite) updates automatically
    elite_cfg = settings.get("elite", {})
    state = load_elite_state()
    state = update_elite(
        state,
        ranked,
        enter_weeks=int(elite_cfg.get("enter_weeks", 2)),
        exit_weeks=int(elite_cfg.get("exit_weeks", 2)),
        elite_cap=int(elite_cfg.get("elite_cap", 10)),
    )
    save_elite_state(state)

    absolute_top10 = state["elite"]

    top20 = scored.head(20).copy()
    top20_tickers = top20.index.tolist()

    overlap = sorted(set(top20_tickers).intersection(set(absolute_top10)))
    overlap_pct_of_abs10 = round(100.0 * len(overlap) / 10.0, 1)

    # timing on top20
    timing_cfg = settings["timing"]
    px = load_prices_for_timing(top20_tickers, history_days=int(timing_cfg["ath_lookback_days"]) + 300)

    timing_rows = []
    for t in top20_tickers:
        if t not in px or len(px[t]) < 250:
            timing_rows.append({"ticker": t, "signal": "DATA_ERROR", "why": ["Missing price history"]})
            continue
        sig = compute_timing_signals(px[t], timing_cfg)
        timing_rows.append({"ticker": t, **sig})

    timing_df = pd.DataFrame(timing_rows).set_index("ticker")
    combined = top20.join(timing_df, how="left")

    order = {"TRIGGER": 0, "SETUP": 1, "WATCH": 2, "DATA_ERROR": 3}
    combined["signal_rank"] = combined["signal"].map(order).fillna(99)
    combined = combined.sort_values(["signal_rank", "fisherScore"], ascending=[True, False])

    outdir = Path("output")
    outdir.mkdir(exist_ok=True)

    payload = {
        "as_of": str(date.today()),
        "universe_size_after_filters": int(base.shape[0]),
        "absolute_top10": absolute_top10,
        "top20": combined.reset_index().to_dict(orient="records"),
        "overlap_with_absolute_top10": overlap,
        "overlap_pct_of_absolute_top10": overlap_pct_of_abs10,
    }

    (outdir / "signals.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("# Weekly Fisher (Proxy) Screener + 50DMA Timing")
    lines.append(f"As of **{date.today()}**")
    lines.append("")
    lines.append(f"Universe after filters: **{payload['universe_size_after_filters']}**")
    lines.append(f"Absolute top 10 (elite): **{', '.join(absolute_top10) if absolute_top10 else '(empty until streaks build)'}**")
    lines.append(f"Overlap with elite in Top 20: **{len(overlap)}/10 = {overlap_pct_of_abs10}%**")
    if overlap:
        lines.append(f"Overlap tickers: {', '.join(overlap)}")
    lines.append("")

    for t, row in combined.iterrows():
        lines.append(f"## {t} — {row.get('signal','?')} (FisherProxy {row.get('fisherScore','?')})")
        lines.append(f"- Market cap: {row.get('marketCap', float('nan')):,.0f}")
        lines.append(f"- Avg $ volume: {row.get('avgDollarVol', float('nan')):,.0f}")
        if pd.notna(row.get("last_close", np.nan)):
            lines.append(f"- Close: {row['last_close']:.2f} | 50DMA: {row['ma50']:.2f} | 200DMA: {row['ma200']:.2f}")
            lines.append(f"- ATH: {row['ath']:.2f} | Days since ATH: {row['days_since_ath']}")
            lines.append(f"- Drawdown from ATH: {row['drawdown_from_ath']:.1%} | Dist to 50DMA: {row['near_ma50_pct']:.1%}")
        why = row.get("why")
        if isinstance(why, list):
            for w in why:
                lines.append(f"- {w}")
        lines.append("")

    (outdir / "report.md").write_text("\n".join(lines))
    print("Wrote output/report.md, output/signals.json, state/elite.json")

if __name__ == "__main__":
    main()
