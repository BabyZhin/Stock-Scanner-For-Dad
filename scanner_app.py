"""
US Stock Scanner (Streamlit)  |  美股扫描器

Setup:  pip install -U streamlit yfinance pandas numpy plotly
Run:    streamlit run scanner_app.py
"""
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

MIN_FLOAT = 450_000_000      # 4.5亿股 float
MIN_MCAP = 4_000_000_000     # 40亿美元 market cap
COLS = ["symbol", "name", "sig", "side", "price", "chg", "mcap", "flt", "date"]

# ───────────────────────── i18n ─────────────────────────
TXT = {
    "en": dict(
        title="📡 US Stock Scanner",
        sub="Float > 450M shares · Market cap > $4B · Daily & Weekly",
        mode="Mode", m_auto="Auto-scan US market", m_custom="My own tickers",
        tickers="Tickers (space / comma separated)",
        signals="Signals", side="Show", sell="Sell", buy="Buy", both="Both",
        run="🚀 Run scan", daily="Daily", weekly="Weekly",
        uni="Building universe & checking float / market cap (first run takes a few minutes)…",
        px="Downloading prices…", passed="{} stocks passed the filter",
        empty="No stocks to scan.", none="No signals right now.",
        hint="Pick a mode in the sidebar and hit Run scan.",
        symbol="Symbol", name="Name", signal="Signal", direction="Side", price="Price",
        chg="Chg %", mcap="Mkt cap", flt="Float", date="Bar date",
        dl="⬇️ Download CSV", chart="Chart a symbol",
        buy_lbl="🟢 Buy", sell_lbl="🔴 Sell",
        sig=dict(macd="MACD cross", kdj="KDJ cross", rsi="RSI exits 70/30", ma="Close vs MA20"),
    ),
    "zh": dict(
        title="📡 美股扫描器",
        sub="流通股 > 4.5亿股 · 市值 > 40亿美元 · 日线 & 周线",
        mode="模式", m_auto="自动扫描美股", m_custom="自选股票",
        tickers="股票代码（空格 / 逗号分隔）",
        signals="信号", side="显示", sell="卖出", buy="买入", both="全部",
        run="🚀 开始扫描", daily="日线", weekly="周线",
        uni="正在构建股票池并检查流通股 / 市值（首次运行需几分钟）…",
        px="正在下载行情…", passed="{} 只股票通过筛选",
        empty="没有可扫描的股票。", none="目前没有信号。",
        hint="在左侧选择模式，然后点击“开始扫描”。",
        symbol="代码", name="名称", signal="信号", direction="方向", price="现价",
        chg="涨跌幅 %", mcap="市值", flt="流通股", date="K线日期",
        dl="⬇️ 下载 CSV", chart="查看K线",
        buy_lbl="🟢 买入", sell_lbl="🔴 卖出",
        sig=dict(macd="MACD 金叉/死叉", kdj="KDJ 金叉/死叉", rsi="RSI 脱离超买/超卖", ma="收盘价 vs MA20"),
    ),
}


# ───────────────────────── data ─────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def us_universe() -> list:
    """US-listed symbols with market cap > $4B (Yahoo screener); falls back to Nasdaq Trader lists."""
    try:
        q = yf.EquityQuery("and", [
            yf.EquityQuery("eq", ["region", "us"]),
            yf.EquityQuery("is-in", ["exchange", "NMS", "NYQ", "NGM", "ASE"]),
            yf.EquityQuery("gt", ["intradaymarketcap", MIN_MCAP]),
        ])
        syms, offset = [], 0
        while True:
            r = yf.screen(q, offset=offset, size=250, sortField="intradaymarketcap", sortAsc=False)
            quotes = r.get("quotes", [])
            syms += [x["symbol"] for x in quotes]
            offset += 250
            if not quotes or offset >= r.get("total", 0):
                break
        if syms:
            return sorted(set(syms))
    except Exception:
        pass
    # fallback: every listed non-ETF symbol (slower: ~6000 symbols to check)
    base, out = "https://www.nasdaqtrader.com/dynamic/SymDir/", []
    for f, col in (("nasdaqlisted.txt", "Symbol"), ("otherlisted.txt", "ACT Symbol")):
        d = pd.read_csv(base + f, sep="|")
        d = d[(d["Test Issue"] == "N") & (d["ETF"] == "N")]
        out += d[col].dropna().astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted({s for s in out if s.replace("-", "").isalpha()})


def _info(sym):
    try:
        i = yf.Ticker(sym).info
        return sym, i.get("shortName") or sym, i.get("marketCap"), i.get("floatShares")
    except Exception:
        return sym, sym, None, None


@st.cache_data(ttl=86400, show_spinner=False)
def fundamentals(symbols: tuple) -> pd.DataFrame:
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(_info, symbols))
    df = pd.DataFrame(rows, columns=["symbol", "name", "mcap", "float_sh"])
    df["mcap"] = pd.to_numeric(df["mcap"], errors="coerce")
    df["float_sh"] = pd.to_numeric(df["float_sh"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def download_chunk(chunk: tuple) -> dict:
    raw = yf.download(list(chunk), period="2y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    for s in chunk:
        try:
            d = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if len(d) > 60:
                out[s] = d
        except KeyError:
            continue
    return out


# ───────────────────────── indicators & signals ─────────────────────────
def to_weekly(d):
    return d.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()


def add_indicators(d):
    c, h, l = d["Close"], d["High"], d["Low"]
    d["DIF"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + up / dn)
    lo, hi = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - lo) / (hi - lo) * 100).replace([np.inf, -np.inf], np.nan)
    d["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1 / 3, adjust=False).mean()
    d["MA20"] = c.rolling(20).mean()
    return d


def latest_signals(d):
    """Signals that fired on the newest bar: list of (key, +1 buy / -1 sell)."""
    p, n = d.iloc[-2], d.iloc[-1]

    def cross(a, b):
        if p[a] <= p[b] and n[a] > n[b]:
            return 1
        if p[a] >= p[b] and n[a] < n[b]:
            return -1
        return 0

    checks = {
        "macd": cross("DIF", "DEA"),
        "kdj": cross("K", "D"),
        "rsi": -1 if p["RSI"] >= 70 > n["RSI"] else (1 if p["RSI"] <= 30 < n["RSI"] else 0),
        "ma": cross("Close", "MA20"),
    }
    return [(k, s) for k, s in checks.items() if s]


def run_scan(frames, meta, tf):
    rows = []
    for sym, df in frames.items():
        d = to_weekly(df) if tf == "weekly" else df
        if len(d) < 35:
            continue
        d = add_indicators(d.copy())
        m = meta.loc[sym]
        for key, side in latest_signals(d):
            rows.append(dict(
                symbol=sym, name=m["name"], sig=key, side=side,
                price=d["Close"].iloc[-1], chg=d["Close"].pct_change().iloc[-1] * 100,
                mcap=m["mcap"] / 1e9, flt=m["float_sh"] / 1e6, date=d.index[-1].date(),
            ))
    out = pd.DataFrame(rows, columns=COLS)
    return out.sort_values("mcap", ascending=False) if len(out) else out


# ───────────────────────── UI helpers ─────────────────────────
def show(df, tf, t):
    if df.empty:
        st.info(t["none"])
        return
    v = df.assign(sig=df["sig"].map(t["sig"]),
                  side=df["side"].map({1: t["buy_lbl"], -1: t["sell_lbl"]}))
    cc = st.column_config
    st.dataframe(v, hide_index=True, column_config={
        "symbol": cc.TextColumn(t["symbol"]), "name": cc.TextColumn(t["name"]),
        "sig": cc.TextColumn(t["signal"]), "side": cc.TextColumn(t["direction"]),
        "price": cc.NumberColumn(t["price"], format="$%.2f"),
        "chg": cc.NumberColumn(t["chg"], format="%.2f"),
        "mcap": cc.NumberColumn(t["mcap"], format="$%.1fB"),
        "flt": cc.NumberColumn(t["flt"], format="%.0fM"),
        "date": cc.DateColumn(t["date"]),
    })
    st.download_button(t["dl"], v.to_csv(index=False).encode("utf-8-sig"),
                       f"scan_{tf}.csv", "text/csv", key=f"dl_{tf}")


def draw_chart(frames, sym, tf):
    d = frames[sym]
    d = to_weekly(d) if tf == "weekly" else d.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d = d.tail(120)
    fig = go.Figure([
        go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                       close=d["Close"], name=sym),
        go.Scatter(x=d.index, y=d["MA20"], name="MA20", line=dict(width=1.5)),
    ])
    fig.update_layout(xaxis_rangeslider_visible=False, height=450,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)


# ───────────────────────── app ─────────────────────────
st.set_page_config(page_title="Stock Scanner", page_icon="📡", layout="wide")

lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"], horizontal=True)
t = TXT["en" if lang == "English" else "zh"]

st.title(t["title"])
st.caption(t["sub"])

mode = st.sidebar.radio(t["mode"], ["auto", "custom"],
                        format_func=lambda m: t["m_" + m])
raw = st.sidebar.text_area(t["tickers"], "AAPL NVDA TSLA MSFT") if mode == "custom" else ""
wanted = st.sidebar.multiselect(t["signals"], list(t["sig"]), default=list(t["sig"]),
                                format_func=t["sig"].get)
side = st.sidebar.radio(t["side"], [-1, 1, 0], horizontal=True,
                        format_func=lambda s: {-1: t["sell"], 1: t["buy"], 0: t["both"]}[s])

if st.sidebar.button(t["run"], type="primary"):
    with st.spinner(t["uni"]):
        if mode == "auto":
            meta = fundamentals(tuple(us_universe()))
            meta = meta[(meta["mcap"] > MIN_MCAP) & (meta["float_sh"] > MIN_FLOAT)]
        else:
            syms = tuple(dict.fromkeys(
                s for s in re.split(r"[\s,;，；、]+", raw.upper()) if s))
            meta = fundamentals(syms) if syms else pd.DataFrame(
                columns=["symbol", "name", "mcap", "float_sh"])
    meta = meta.set_index("symbol")
    symbols = list(meta.index)

    if not symbols:
        st.warning(t["empty"])
    else:
        if mode == "auto":
            st.toast(t["passed"].format(len(symbols)))
        frames, bar = {}, st.progress(0.0, t["px"])
        for i in range(0, len(symbols), 100):
            frames.update(download_chunk(tuple(symbols[i:i + 100])))
            bar.progress(min((i + 100) / len(symbols), 1.0), t["px"])
        bar.empty()
        meta = meta.loc[[s for s in frames if s in meta.index]]
        st.session_state["res"] = dict(
            frames=frames,
            daily=run_scan(frames, meta, "daily"),
            weekly=run_scan(frames, meta, "weekly"),
        )

res = st.session_state.get("res")
if not res:
    st.info(t["hint"])
    st.stop()

tabs = st.tabs([t["daily"], t["weekly"]])
for tab, tf in zip(tabs, ("daily", "weekly")):
    with tab:
        df = res[tf]
        df = df[df["sig"].isin(wanted)]
        if side:
            df = df[df["side"] == side]
        show(df, tf, t)

pool = sorted(set(res["daily"]["symbol"]) | set(res["weekly"]["symbol"]))
if pool:
    st.divider()
    c1, c2 = st.columns([2, 1])
    sym = c1.selectbox(t["chart"], pool)
    tf = c2.radio("tf", ["daily", "weekly"], format_func=lambda x: t[x],
                  horizontal=True, label_visibility="collapsed")
    draw_chart(res["frames"], sym, tf)
