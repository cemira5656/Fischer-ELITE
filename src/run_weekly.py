from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from email_report import send_gmail
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
    rows: list[dict] = []
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

            rows.append(
                {
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
                }
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")


def load_prices_for_timing(tickers: list[str], history_days: int) -> dict[str, pd.Series]:
    """
    Robustly fetch Adj Close (or Close) for each ticker using yfinance.download.
    Handles both MultiIndex (multi-ticker) and single-index (single ticker) responses.
    """
    px: dict[str, pd.Series] = {}

    data = yf.download(
        tickers=tickers,
        period=f"{history_days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # MultiIndex columns: (Ticker, Field)
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Adj Close") in data.columns:
                s = data[(t, "Adj Close")].dropna()
            elif (t, "Close") in data.columns:
                s = data[(t, "Close")].dropna()
            else:
                continue
            if len(s) > 0:
                px[t] = s

    # Single ticker: normal columns
    else:
        if "Adj Close" in data.columns:
            s = data["Adj Close"].dropna()
        elif "Close" in data.columns:
            s = data["Close"].dropna()
        else:
            s = pd.Series(dtype=float)

        if len(s) > 0 and tickers:
            px[tickers[0]] = s

    return px


def load_elite_state(path: str = "state/elite.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {"as_of": None, "elite": [], "history": {}}
    return json.loads(p.read_text())


def save_elite_state(state: dict, path: str = "state/elite.json") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2))


def update_elite(
    state: dict,
    ranked: list[str],
    enter_weeks: int = 2,
    exit_weeks: int = 2,
    elite_cap: int = 10,
) -> dict:
    """
    Persistent 'absolute top 10' definition:
      - enters elite if top10 for enter_weeks consecutive runs
      - exits elite if outside top20 for exit_weeks consecutive runs
      - elite list capped at elite_cap by current ranking priority
    """
    top10 = set(ranked[:10])
    top20 = set(ranked[:20])

    hist: dict = state.get("history", {})
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


def to_html_email(as_of: str, elite: list[str], overlap_pct: float, combined: pd.DataFrame) -> str:
    triggers = combined[combined["signal"] == "TRIGGER"].copy()
    setups = combined[combined["signal"] == "SETUP"].copy()

    def fmt_df(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p><em>None</em></p>"

        cols = [
            "fisherScore",
            "signal",
            "last_close",
            "ma50",
            "drawdown_from_ath",
            "near_ma50_pct",
            "days_since_ath",
        ]
        df2 = df[cols].copy()
        df2.insert(0, "ticker", df.index)

        df2["fisherScore"] = df2["fisherScore"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        df2["last_close"] = df2["last_close"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        df2["ma50"] = df2["ma50"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        df2["drawdown_from_ath"] = df2["drawdown_from_ath"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        df2["near_ma50_pct"] = df2["near_ma50_pct"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        df2["days_since_ath"] = df2["days_since_ath"].map(lambda x: f"{int(x)}" if pd.notna(x) else "")

        return df2.to_html(index=False, escape=True, border=0)

    style = """
    <style>
      body { font-family: Arial, sans-serif; }
      h2 { margin: 0 0 6px 0; }
      p { margin: 6px 0; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 16px 0; }
      th, td { border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }
      th { background: #f6f6f6; }
      .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; background: #eee; font-size: 12px; }
      .muted { color: #666; font-size: 12px; }
    </style>
    """

    elite_str = ", ".join(elite) if elite else "(empty until streaks build)"

    html = f"""
    <html>
    <head>{style}</head>
    <body>
      <h2>Weekly Fisher ELITE + 50DMA Signals</h2>
      <p><strong>As of:</strong> {as_of}</p>
      <p><strong>ELITE (Absolute Top 10):</strong> <span class="pill">{elite_str}</span></p>
      <p><strong>Overlap (ELITE in Top 20):</strong> {overlap_pct:.1f}%</p>

      <h3>TRIGGER (bounce confirmed)</h3>
      {fmt_df(triggers)}

      <h3>SETUP (near 50DMA after ATH pullback)</h3>
      {fmt_df(setups)}

      <p class="muted">
        Full details are committed to your repo in <code>output/report.md</code>.
      </p>
    </body>
    </html>
    """
    return html


def main() -> None:
    settings = load_yaml("config/settings.yml")

    # 1) Universe
    syms = fetch_us_common_stock_symbols(max_symbols=settings["universe"]["max_symbols"])

    # 2) Fundamentals/liquidity
    base = get_fast_fundamentals(syms)
    if base.empty:
        raise RuntimeError("No fundamentals data returned. yfinance may be rate-limiting or failing.")

    # 3) Filters
    u = settings["universe"]
    base = base.dropna(subset=["marketCap", "avgDollarVol"], how="any")
    base = base[
        (base["marketCap"] >= u["min_market_cap"]) & (base["avgDollarVol"] >= u["min_avg_dollar_vol"])
    ].copy()

    # Keep workload sane
    base = base.sort_values("marketCap", ascending=False).head(1200)

    # 4) Score + rank
    scored = fisher_proxy_score(base, settings["scoring"]["weights"])
    scored = scored.sort_values("fisherScore", ascending=False)
    ranked = scored.index.tolist()

    # 5) Elite update
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

    # 6) Top 20 + overlap
    top20 = scored.head(20).copy()
    top20_tickers = top20.index.tolist()

    overlap = sorted(set(top20_tickers).intersection(set(absolute_top10)))
    overlap_pct_of_abs10 = round(100.0 * len(overlap) / 10.0, 1)

    # 7) Timing on top 20
    timing_cfg = settings["timing"]
    px = load_prices_for_timing(top20_tickers, history_days=int(timing_cfg["ath_lookback_days"]) + 300)

    timing_rows: list[dict] = []
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

    # 8) Write outputs
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

    lines: list[str] = []
    lines.append("# Weekly Fisher (Proxy) Screener + 50DMA Timing")
    lines.append(f"As of **{date.today()}**")
    lines.append("")
    lines.append(f"Universe after filters: **{payload['universe_size_after_filters']}**")
    lines.append(
        "Absolute top 10 (elite): **"
        + (", ".join(absolute_top10) if absolute_top10 else "(empty until streaks build)")
        + "**"
    )
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

    # 9) Email both plain text + HTML table version
    subject = f"Weekly Fisher ELITE + 50DMA Signals — {date.today()}"
    report_text = (outdir / "report.md").read_text()
    report_html = to_html_email(str(date.today()), absolute_top10, overlap_pct_of_abs10, combined)
    send_gmail(subject, report_text, report_html)

    print("Wrote output/report.md, output/signals.json, state/elite.json (and emailed report)")


if __name__ == "__main__":
    main()
