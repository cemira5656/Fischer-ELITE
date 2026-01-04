from __future__ import annotations
import io
import pandas as pd
import requests

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

def _load_txt(url: str, sep: str = "|") -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    text = r.text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    buf = io.StringIO("\n".join(lines))
    return pd.read_csv(buf, sep=sep)

def fetch_us_common_stock_symbols(max_symbols: int = 2500) -> list[str]:
    nas = _load_txt(NASDAQ_LISTED)
    oth = _load_txt(OTHER_LISTED)

    oth = oth.rename(columns={"ACT Symbol": "Symbol"})

    if "Test Issue" in nas.columns:
        nas = nas[nas["Test Issue"] == "N"]
    if "ETF" in nas.columns:
        nas = nas[nas["ETF"] == "N"]

    if "Test Issue" in oth.columns:
        oth = oth[oth["Test Issue"] == "N"]
    if "ETF" in oth.columns:
        oth = oth[oth["ETF"] == "N"]

    def is_probably_common(name: str) -> bool:
        n = str(name).lower()
        bad = ["preferred", "depositary", "warrant", "unit", "notes", "bond", "trust", "etf", "etn"]
        return not any(b in n for b in bad)

    if "Security Name" in oth.columns:
        oth = oth[oth["Security Name"].apply(is_probably_common)]

    symbols = pd.concat([nas[["Symbol"]], oth[["Symbol"]]], ignore_index=True)["Symbol"]
    symbols = symbols.dropna().astype(str).str.strip().str.upper().unique().tolist()
    return symbols[:max_symbols]
