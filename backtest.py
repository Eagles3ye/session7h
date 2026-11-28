"""Backtest of the alert strategy (daily trend + H4/H2 sweep + impulsive 30m CISD) on historical candles.

    python backtest.py                      # all symbols in alert_job.SYMBOLS, 5000 x 30m bars
    python backtest.py --bars 20000         # more history (crypto/gold on Binance has plenty)
    python backtest.py --symbols BTCUSDT,ETHUSDT

How trades are simulated (each signal is tested on its own, trades may overlap):
  * entry  = OPEN of the 30m bar after the CISD candle (alerts arrive after the candle closes)
  * exit A = the alert's own SL/TP (forex/gold: fixed pips from alert_job; crypto: sweep stop, RR x risk)
  * exit B = structure exit for every symbol: stop at the sweep extreme, target RR x risk
  * if SL and TP are both inside one bar, the SL is assumed to hit first (pessimistic)
  * a trade still open after 96 bars (2 days) is closed at that bar's close ("TIMEOUT")
  * costs (spread) are subtracted: --cost-pips for forex/gold, --cost-pct for crypto
  * R = result in units of the risk taken (SL hit = -1R)
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import alert_job as aj
import strategy as st
import vela_data as vd

MAX_HOLD = 96      # bars
MFE_BARS = 48      # bars used to measure how far price ran in favour (in R of the sweep stop)


def load(sym: str, bars: int, now: pd.Timestamp):
    base = st.drop_forming(vd.get_bars(sym, "30m", bars), "30m", now)
    lim = {"2h": bars // 4 + 200, "4h": bars // 8 + 100, "1d": bars // 48 + 100, "1w": bars // 336 + 60}
    return base, {tf: st.drop_forming(vd.get_bars(sym, tf, n), tf, now) for tf, n in lim.items()}


def exit_trade(buy, entry, sl, tp, h, l, c, start, cost):
    """Returns (R, outcome, bars_held) or None when the trade is invalid / still open at the end of data."""
    risk = abs(entry - sl)
    if risk <= 0 or (buy and sl >= entry) or (not buy and sl <= entry):
        return None
    last = len(h) - 1
    end = min(start + MAX_HOLD, last)
    for j in range(start, end + 1):
        if (l[j] <= sl) if buy else (h[j] >= sl):
            return -1.0 - cost / risk, "SL", j - start
        if (h[j] >= tp) if buy else (l[j] <= tp):
            return abs(tp - entry) / risk - cost / risk, "TP", j - start
    if end == last and end - start < MAX_HOLD:
        return None                                        # data ended before the trade resolved
    move = (c[end] - entry) if buy else (entry - c[end])
    return move / risk - cost / risk, "TIMEOUT", end - start


def trades_for(sym: str, base: pd.DataFrame, sigs: list, args) -> list[dict]:
    o, h, l, c = (base[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    N = len(base)
    rows = []
    for s in sigs:
        e = s["i"] + 1
        if e >= N:
            continue                                        # newest signal, no next bar yet
        buy = s["dir"] == "BUY"
        sgn = 1 if buy else -1
        entry = o[e]
        pip = aj.pip_size(sym, entry)
        cost = entry * args.cost_pct / 100 if sym in aj.CRYPTO else args.cost_pips * pip
        sl_sw = s["sl_sweep"]
        risk_sw = abs(entry - sl_sw)
        if aj.SL_PIPS and sym not in aj.CRYPTO:
            sl_a, tp_a = entry - sgn * aj.SL_PIPS * pip, entry + sgn * aj.TP_PIPS * pip
        else:
            sl_a, tp_a = sl_sw, entry + sgn * aj.RR * risk_sw
        tp_b = entry + sgn * aj.RR * risk_sw
        a = exit_trade(buy, entry, sl_a, tp_a, h, l, c, e, cost)
        b = exit_trade(buy, entry, sl_sw, tp_b, h, l, c, e, cost)
        mfe = None
        if risk_sw > 0 and e + MFE_BARS < N:
            seg_h, seg_l = h[e:e + MFE_BARS + 1], l[e:e + MFE_BARS + 1]
            mfe = ((seg_h.max() - entry) if buy else (entry - seg_l.min())) / risk_sw
        if a is None and b is None:
            continue
        rows.append(dict(
            time=s["time"], symbol=sym, dir=s["dir"], grade=s["grade"], swept="/".join(s["swept"]),
            fvg=s["fvg"], vol_hi=(None if math.isnan(s["vol_x"]) else s["vol_x"] >= st.Cfg().vol_mult),
            atr_x=s["atr_x"], body=s["body"], entry=entry, risk_pips=risk_sw / pip,
            r_alert=a[0] if a else np.nan, out_alert=a[1] if a else "",
            r_struct=b[0] if b else np.nan, out_struct=b[1] if b else "", mfe_r=mfe))
    return rows


def table(df: pd.DataFrame, rcol: str, ocol: str, by) -> str:
    d = df.dropna(subset=[rcol])
    if d.empty:
        return "  (no trades)"
    g = d.groupby(by, dropna=False)
    t = pd.DataFrame({
        "trades": g.size(),
        "TP_hit_%": g[ocol].apply(lambda s: (s == "TP").mean() * 100).round(1),
        "SL_hit_%": g[ocol].apply(lambda s: (s == "SL").mean() * 100).round(1),
        "avg_R": g[rcol].mean().round(2),
        "total_R": g[rcol].sum().round(1),
    })
    return t.to_string()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=5000)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--cost-pips", type=float, default=1.0)
    ap.add_argument("--cost-pct", type=float, default=0.05)
    ap.add_argument("--out", default="backtest_trades.csv")
    args = ap.parse_args()

    import os
    vd.configure(twelve_key=os.environ.get("TWELVE_KEY"))
    vd.CONFIG["twelve_min_gap"] = 7.8
    now = pd.Timestamp.now(tz="UTC")
    syms = [s for s in aj.SYMBOLS if not args.symbols or s in args.symbols.split(",") or aj.name_of(s) in args.symbols]
    rows = []
    for sym in syms:
        try:
            base, frames = load(sym, args.bars, now)
        except (vd.DataError, Exception) as e:                   # keep going if one symbol fails
            print(f"{sym}: DATA ERROR {e}")
            continue
        sigs = st.find_signals(base, frames["2h"], frames["4h"], frames["1d"], frames["1w"], aj.CFG)
        tr = trades_for(sym, base, sigs, args)
        rows += tr
        span = f"{base.index[0]:%Y-%m-%d} -> {base.index[-1]:%Y-%m-%d}"
        print(f"{aj.name_of(sym):16} {len(base):6} bars  {span}  signals {len(sigs):4}  simulated {len(tr):4}")
    if not rows:
        print("\nNo trades to analyse.")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    df["pair"] = df["symbol"].map(aj.name_of)
    be = 1 / (1 + aj.RR) * 100
    print(f"\nBreak-even TP-hit rate at {aj.RR:.0f}R target = {be:.0f}% (before costs). Costs already included in R.")
    for title, rcol, ocol in (("EXIT A: the alert's own SL/TP (what you would trade)", "r_alert", "out_alert"),
                              ("EXIT B: sweep stop, target = RR x risk (all pairs)", "r_struct", "out_struct")):
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print("\nALL\n" + table(df.assign(all="all"), rcol, ocol, "all"))
        print("\nBY GRADE (A+ vs B)\n" + table(df, rcol, ocol, "grade"))
        print("\nBY PAIR\n" + table(df, rcol, ocol, "pair"))
        print("\nBY PAIR + GRADE\n" + table(df, rcol, ocol, ["pair", "grade"]))
        print("\nBY SWEPT TIMEFRAME\n" + table(df, rcol, ocol, "swept"))
        print("\nFVG on CISD candle?\n" + table(df, rcol, ocol, "fvg"))
        print("\nVolume spike (>=1.5x avg; blank = no volume data)\n" + table(df, rcol, ocol, "vol_hi"))
    m = df.dropna(subset=["mfe_r"]).groupby("grade")["mfe_r"].agg(["count", "median", "mean"]).round(2)
    print(f"\n{'=' * 78}\nHOW FAR PRICE RAN IN FAVOUR in the next {MFE_BARS} bars, in R of the sweep stop\n{'=' * 78}\n{m.to_string()}")
    print(f"\nTrades saved to {args.out}. Samples under ~100 trades are NOT reliable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
