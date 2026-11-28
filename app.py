"""Streamlit front-end for the 7H Session Profile + MTF Structure + H4 CRT indicator
(TradingView-style dark layout: dashboard cards on top, chart below)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import indicator as ind
import ui
import vela_data as vd
from ui import BG, COL, GRID

st.set_page_config(page_title="7H Session Profile", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

BOX = {"ASIA": ("#787b86", "ASIA 7H"), "LONDON": ("#2962ff", "LONDON 7H"), "NEW YORK": ("#089981", "NEW YORK 7H")}
TFS = ["5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
TZS = ["Asia/Kolkata", "Asia/Dubai", "America/New_York", "UTC", "Europe/London", "Asia/Tokyo"]
MARKETS = ["twelve:EUR/USD", "twelve:GBP/USD", "twelve:NZD/USD", "PAXGUSDT", "BTCUSDT", "ETHUSDT",
           "oanda:XAU_USD", "Custom…"]


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# ---------------------------------------------------------------- top bar ----
st.markdown('<div class="brand">7H SESSION PROFILE<span>MTF Structure · H4 CRT · CISD</span></div>',
            unsafe_allow_html=True)
c_sym, c_tf, c_btn = st.columns([1.5, 3.6, 0.7], vertical_alignment="bottom")
with c_sym:
    preset = st.selectbox("Market", MARKETS, label_visibility="collapsed",
                          help="PAXGUSDT = gold-backed token on Binance (free, no key). twelve:* needs TWELVE_KEY.")
    symbol = (st.text_input("Custom symbol", "coinbase:BTC-USD", label_visibility="collapsed",
                            help="BTCUSDT · BTCUSDT.P · coinbase:BTC-USD · hyperliquid:BTC · oanda:USD_JPY")
              if preset == "Custom…" else preset)
with c_tf:
    interval = st.segmented_control("Timeframe", TFS, default="30m", label_visibility="collapsed") or "30m"
with c_btn:
    if st.button("↻ Refresh", width="stretch"):
        st.cache_data.clear()

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Chart")
    n_bars = st.slider("Bars to load", 500, 5000, 2000, 250)
    view_bars = st.slider("Bars shown on chart", 100, 1500, 400, 50)

    st.header("7-Hour Session Windows")
    tz = st.selectbox("Timezone", TZS, index=0)
    asia_sess = st.text_input("Asia 7H Window", "0430-1130")
    lon_sess = st.text_input("London 7H Window", "1130-1830")
    ny_sess = st.text_input("New York 7H Window", "1830-0130")

    st.header("Visual styling")
    show_boxes = st.checkbox("Live 7H session boxes", True)
    show_prev = st.checkbox("Prior session High/Low lines", True)
    show_inv = st.checkbox("Session 50% invalidation line", True)
    show_signals = st.checkbox("Trend-filtered CISD signals", True)
    show_dots = st.checkbox("Recent major pivot dots", True)
    show_lines = st.checkbox("Major BOS / sweep lines", True)
    show_eq = st.checkbox("Dealing range 50% EQ line", True)
    show_crt = st.checkbox("H4 PD-filtered CRT sweep lines", True)
    show_vol = st.checkbox("Volume panel (when data has volume)", True)

    st.header("Structure / CRT / CISD")
    m_len = st.number_input("Monthly structure strength", 1, 20, 1)
    w_len = st.number_input("Weekly structure strength", 1, 20, 1)
    d_len = st.number_input("Daily structure strength", 1, 20, 1)
    h4_len = st.number_input("H4 structure strength", 1, 20, 3)
    major_len = st.number_input("Major swing pivot length", 1, 30, 3)
    sweep_bars = st.number_input("CRT lookback bars", 1, 500, 5)
    max_bars_cisd = st.slider("Max bars after sweep for CISD", 1, 20, 6)
    req_disp = st.checkbox("Require strong body close (displacement)", True)
    strict_trend = st.checkbox("Strictly filter CISD by Daily trend bias", True)


# ------------------------------------------------------------------- data ----
@st.cache_data(ttl=120, show_spinner=False)
def load(sym: str, tf: str, n: int) -> pd.DataFrame:
    return vd.get_bars(sym, tf, n)


@st.cache_data(ttl=900, show_spinner=False)          # higher-timeframe bars change slowly; saves API credits
def load_htf(sym: str, tf: str, n: int) -> pd.DataFrame:
    return vd.get_bars(sym, tf, n)


try:                                   # secrets.toml may not exist locally
    vd.configure(oanda_token=st.secrets.get("OANDA_TOKEN"), oanda_env=st.secrets.get("OANDA_ENV"),
                 twelve_key=st.secrets.get("TWELVE_KEY"))
except Exception:
    pass

HTF_MAP = {"M": ("1M", 600), "1W": ("1w", 1000), "1D": ("1d", 1500), "H4": ("4h", 1500)}

try:
    with st.spinner("Loading candles…"):
        base = load(symbol, interval, n_bars)
        htf = {}
        for key, (tf, lim) in HTF_MAP.items():
            htf[key] = base if tf == interval else load_htf(symbol, tf, lim)
except vd.DataError as e:
    st.error(str(e))
    st.info("If Binance is geo-blocked from your host, try `coinbase:BTC-USD` or `hyperliquid:BTC`.")
    st.stop()

if len(base) < 100:
    st.warning("Not enough candles returned.")
    st.stop()

params = ind.Params(interval=interval, tz=tz, asia_sess=asia_sess, lon_sess=lon_sess, ny_sess=ny_sess,
                    m_len=int(m_len), w_len=int(w_len), d_len=int(d_len), h4_len=int(h4_len),
                    major_len=int(major_len), sweep_bars=int(sweep_bars), max_bars_cisd=int(max_bars_cisd),
                    req_disp=req_disp, strict_trend=strict_trend)
try:
    R = ind.run(base, htf, params)
except ValueError as e:                                   # e.g. malformed session string
    st.error(f"Check session inputs (format HHMM-HHMM): {e}")
    st.stop()

# ------------------------------------------------------------------ chart ----
T = R.times
N = len(T)
lo_i = max(0, N - view_bars)
step = (T[-1] - T[-2]) if N > 1 else pd.Timedelta(minutes=30)


def tx(i: int):
    """bar index -> timestamp (extrapolates beyond the last bar)."""
    return T[i] if i < N else T[-1] + step * (i - N + 1)


view = slice(lo_i, N)
vb = base.iloc[view]
has_vol = show_vol and float(vb["volume"].sum()) > 0
fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True, vertical_spacing=0.015,
                    row_heights=[0.8, 0.2] if has_vol else [1.0])
fig.add_trace(go.Candlestick(
    x=T[view], open=vb["open"], high=vb["high"], low=vb["low"], close=vb["close"],
    increasing_line_color=COL["bull"], increasing_fillcolor=COL["bull"],
    decreasing_line_color=COL["bear"], decreasing_fillcolor=COL["bear"], name=symbol, showlegend=False),
    row=1, col=1)
if has_vol:
    fig.add_trace(go.Bar(x=T[view], y=vb["volume"], showlegend=False, hoverinfo="skip",
                         marker_color=[rgba(COL["bull"] if c >= o else COL["bear"], 0.55)
                                       for o, c in zip(vb["open"], vb["close"])]), row=2, col=1)

shapes, notes = [], []


def hline(x0, x1, y, color, width=1, dash="solid"):
    shapes.append(dict(type="line", x0=tx(x0), x1=tx(x1), y0=y, y1=y, xref="x", yref="y",
                       line=dict(color=color, width=width, dash=dash)))


def label(i, y, text, color, ay, bg=None, size=10):
    notes.append(dict(x=tx(i), y=y, text=text, showarrow=False, yshift=ay, font=dict(color=color, size=size),
                      bgcolor=bg, borderpad=2))


if show_boxes:
    for b in R.boxes:
        if b["end"] < lo_i:
            continue
        c, txt = BOX[b["name"]]
        shapes.append(dict(type="rect", x0=tx(max(b["start"], lo_i)), x1=tx(b["end"]), y0=b["low"], y1=b["high"],
                           xref="x", yref="y", fillcolor=rgba(c, 0.12), line=dict(color=rgba(c, 0.6), width=1), layer="below"))
        label((max(b["start"], lo_i) + b["end"]) // 2, b["high"], txt, c, 10)

if show_prev and R.liq:
    hline(max(R.liq["x0"], lo_i), R.liq["x1"], R.liq["high"], rgba(COL["bear"], 0.6), dash="dot")
    hline(max(R.liq["x0"], lo_i), R.liq["x1"], R.liq["low"], rgba(COL["bull"], 0.6), dash="dot")

if show_inv:
    for v in R.invs:
        if v["x1"] < lo_i:
            continue
        c = COL["dead"] if v["broken"] else (COL["bull"] if v["dir"] == 1 else COL["bear"])
        hline(max(v["x0"], lo_i), v["x1"], v["y"], c, width=2, dash="dash" if v["broken"] else "solid")
        side = "Buy" if v["dir"] == 1 else "Sell"
        txt = f'{v["name"]} 50%<br>{side} Setup Invalidated' if v["broken"] else f'{v["name"]}<br>50% {side} Setup'
        label(v["x1"], v["y"], txt, c, 0, size=9)

if show_lines:
    for s in R.struct_lines:
        if s["x1"] >= lo_i:
            hline(max(s["x0"], lo_i), s["x1"], s["y"], COL[s["color"]], dash="dash" if s["dash"] else "solid")

if show_crt and show_lines:
    for s in R.crt_lines:
        if s["x1"] >= lo_i:
            hline(s["x0"], s["x1"], s["y"], COL["crt"])

if show_eq and R.eq:
    hline(R.eq["x0"], R.eq["x1"], R.eq["y"], COL["eq"], dash="dot")

if show_signals:
    for s in R.sweeps:
        if s["idx"] >= lo_i:
            c = COL["bull"] if s["dir"] == 1 else COL["bear"]
            label(s["idx"], s["y"], "Sweep", c, -14 if s["dir"] == 1 else 14, bg=rgba(c, 0.2), size=9)
    for s in R.cisd:
        if s["idx"] >= lo_i:
            c = COL["bull"] if s["dir"] == 1 else COL["bear"]
            hline(max(s["x0"], lo_i), s["x1"], s["lvl"], c, width=2)
            label(s["idx"], s["y"], s["text"], "#ffffff", -26 if s["dir"] == 1 else 26, bg=c, size=10)

if show_dots:
    for kind in ("bear", "bull"):
        pts = [d for d in R.dots if d["kind"] == kind and d["idx"] >= lo_i]
        if pts:
            fig.add_trace(go.Scatter(x=[tx(d["idx"]) for d in pts], y=[d["y"] for d in pts], mode="markers",
                                     showlegend=False, hoverinfo="skip",
                                     marker=dict(size=8, color=COL[kind], line=dict(width=1, color=BG))),
                          row=1, col=1)

# last-price line + tag on the price axis (like TradingView)
last, last_o = float(base["close"].iloc[-1]), float(base["open"].iloc[-1])
lc = COL["bull"] if last >= last_o else COL["bear"]
shapes.append(dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=last, y1=last,
                   line=dict(color=lc, width=1, dash="dot")))
notes.append(dict(xref="paper", x=1, y=last, yref="y", text=f"<b>{ui.pf(last)}</b>", showarrow=False,
                  xanchor="left", font=dict(color="#ffffff", size=11), bgcolor=lc, borderpad=3))

view_hi, view_lo = float(vb["high"].max()), float(vb["low"].min())
pad = (view_hi - view_lo) * 0.06
fig.update_layout(
    template="plotly_dark", height=780 if has_vol else 700, margin=dict(l=4, r=78, t=6, b=6),
    shapes=shapes, annotations=notes, showlegend=False, dragmode="pan", hovermode="x",
    paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#b2b5be", size=11),
    hoverlabel=dict(bgcolor="#1e222d", bordercolor="#2a2e39", font=dict(size=12)),
    xaxis_rangeslider_visible=False)
fig.update_xaxes(range=[tx(lo_i), tx(N + 6)], showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
                 showspikes=True, spikemode="across", spikecolor="#758696", spikethickness=1, spikedash="dot",
                 rangebreaks=ui.gap_breaks(T, step, lo_i))
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, side="right",
                 showspikes=True, spikemode="across", spikecolor="#758696", spikethickness=1, spikedash="dot")
fig.update_yaxes(range=[view_lo - pad, view_hi + pad], row=1, col=1)

# ----------------------------------------------------------------- layout ----
st.markdown(ui.cards_html(R.table), unsafe_allow_html=True)          # dashboard ABOVE the chart
st.markdown(ui.legend_html(symbol, interval, base), unsafe_allow_html=True)
st.plotly_chart(fig, width="stretch",
                config=dict(scrollZoom=True, displaylogo=False, displayModeBar="hover",
                            modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"]))
st.caption(f"{symbol} · {interval} · last bar {T[-1]:%d %b %Y %H:%M} ({tz}) · "
           f"HTF bias updates on closed bars; the last bar uses the forming HTF bar.")
