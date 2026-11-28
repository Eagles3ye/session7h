"""Presentation helpers for the Streamlit app (pure Python, no Streamlit/Plotly imports)."""
from __future__ import annotations

import html

import pandas as pd

BG, PANEL, BORDER, GRID = "#131722", "#1e222d", "#2a2e39", "rgba(42,46,57,0.55)"
COL = {"bull": "#089981", "bear": "#f23645", "dead": "#787b86", "eq": "#f2c94c", "gray": "#787b86",
       "orange": "#ff9800", "dim": "#b2b5be", "act": "#f2c94c", "white": "#ffffff", "crt": "#7a7a7a"}
SESS_COL = {"ASIA": "#787b86", "LONDON": "#2962ff", "NEW YORK": "#089981"}

CSS = f"""<style>
.stApp {{ background:{BG}; font-family:-apple-system,BlinkMacSystemFont,'Trebuchet MS',Roboto,Ubuntu,sans-serif; }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer, .stAppDeployButton {{ visibility:hidden; display:none; }}
.block-container {{ padding:0.8rem 1.2rem 1rem 1.2rem; max-width:100%; }}
.brand {{ font-weight:700; letter-spacing:.09em; font-size:13px; color:#d1d4dc; margin:0 0 6px 2px; }}
.brand span {{ color:#787b86; font-weight:500; letter-spacing:0; margin-left:8px; }}
.grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:10px; }}
.grid  {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:10px; margin-bottom:12px; }}
.card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:10px 12px; min-width:0; }}
.card.sess {{ border-left:3px solid #787b86; }}
.card.live {{ border-color:#f2c94c66; box-shadow:0 0 0 1px #f2c94c33; }}
.lbl {{ font-size:10.5px; text-transform:uppercase; letter-spacing:.07em; color:#787b86; margin-bottom:4px;
        display:flex; align-items:center; gap:6px; }}
.val {{ font-size:14px; font-weight:600; line-height:1.3; word-break:break-word; }}
.sub {{ font-size:11.5px; color:#787b86; margin-top:5px; }}
.pill {{ background:#f2c94c22; color:#f2c94c; border-radius:999px; font-size:9.5px; padding:1px 7px;
         font-weight:700; letter-spacing:.05em; }}
.chip {{ border-radius:4px; padding:0 6px; font-weight:600; font-size:11px; background:{BORDER}; }}
.legend {{ font-size:13px; color:#d1d4dc; margin:2px 0 4px 2px; display:flex; flex-wrap:wrap;
           gap:4px 14px; align-items:baseline; }}
.legend b {{ font-size:15px; }}
.legend .tf {{ color:#787b86; }}
.legend i {{ font-style:normal; font-weight:500; }}
@media (max-width:900px) {{ .grid3 {{ grid-template-columns:1fr; }} }}
</style>"""


def esc(x) -> str:
    return html.escape(str(x))


def pf(x: float) -> str:
    """Price formatting by magnitude (forex 5 dp, gold/crypto 2 dp)."""
    if x >= 1000:
        return f"{x:,.2f}"
    return f"{x:.3f}" if x >= 10 else f"{x:.5f}"


def cards_html(table: list) -> str:
    """Dashboard rows from indicator.Result.table -> responsive card grid (shown above the chart)."""
    sess, other = [], []
    for row in table:
        if row["kind"] == "row":
            (label, lcol), (target, _), (swept, scol), (result, rcol) = row["cells"]
            live = label.startswith(">")
            name = label.lstrip("> ").strip()
            pill = '<span class="pill">LIVE</span>' if live else ""
            sess.append(
                f'<div class="card sess{" live" if live else ""}" style="border-left-color:{SESS_COL.get(name, "#787b86")}">'
                f'<div class="lbl">{esc(name)}{pill}</div>'
                f'<div class="val" style="color:{COL[rcol]}">{esc(result)}</div>'
                f'<div class="sub">Sweeps {esc(target)} · Swept '
                f'<span class="chip" style="color:{COL[scol]}">{esc(swept)}</span></div></div>')
        elif row["kind"] == "merged":
            (label, _), (value, vcol) = row["cells"]
            other.append(f'<div class="card"><div class="lbl">{esc(label)}</div>'
                         f'<div class="val" style="color:{COL[vcol]}">{esc(value)}</div></div>')
    return f'<div class="grid3">{"".join(sess)}</div><div class="grid">{"".join(other)}</div>'


def legend_html(symbol: str, interval: str, base: pd.DataFrame) -> str:
    """TradingView-style OHLC legend for the last candle."""
    o, h, l, c = (float(base[k].iloc[-1]) for k in ("open", "high", "low", "close"))
    prev = float(base["close"].iloc[-2]) if len(base) > 1 else o
    chg, pct = c - prev, (c - prev) / prev * 100 if prev else 0.0
    col = COL["bull"] if c >= o else COL["bear"]
    cc = COL["bull"] if chg >= 0 else COL["bear"]
    return (f'<div class="legend"><b>{esc(symbol)}</b><span class="tf">{esc(interval)}</span>'
            f'<span>O <i style="color:{col}">{pf(o)}</i></span><span>H <i style="color:{col}">{pf(h)}</i></span>'
            f'<span>L <i style="color:{col}">{pf(l)}</i></span><span>C <i style="color:{col}">{pf(c)}</i></span>'
            f'<span><i style="color:{cc}">{chg:+.5g} ({pct:+.2f}%)</i></span></div>')


def gap_breaks(T: pd.DatetimeIndex, step: pd.Timedelta, lo_i: int) -> list:
    """Plotly rangebreaks that hide market-closed gaps (forex weekends) so candles sit side by side like
    TradingView. Returns [] when the bars are not on a regular grid or there are no gaps."""
    tv = T[lo_i:]
    if step >= pd.Timedelta(days=1) or len(tv) < 20:
        return []
    grid = pd.date_range(tv[0], tv[-1], freq=step)
    if tv.isin(grid).mean() < 0.98:
        return []
    missing = grid.difference(tv)
    if not 0 < len(missing) < len(grid) * 0.6:
        return []
    return [dict(values=[m.strftime("%Y-%m-%d %H:%M:%S") for m in missing], dvalue=int(step.total_seconds() * 1000))]
