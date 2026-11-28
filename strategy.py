"""Alert strategy.

BUY  = daily trend bullish + H4 and/or H2 CRT *low* sweep (in DISCOUNT of the swept timeframe)
       + bullish 30m CISD that is impulsive.
SELL = the mirror image (daily bearish, H4/H2 *high* sweep in PREMIUM, bearish CISD).

Works on CLOSED bars only and never looks ahead: a higher-timeframe candle only becomes
visible on the first 30m bar that closes at/after the candle's close.

FVG and volume are NOT required; they are measured and returned so they can be logged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicator as ind
from vela_data import MIN

TF_NAME = {"2h": "H2", "4h": "H4", "1d": "Daily", "1w": "Weekly"}


@dataclass
class Cfg:
    w_len: int = 1            # weekly structure strength
    d_len: int = 1            # daily structure strength
    h4_len: int = 3           # H4 structure strength
    h2_len: int = 3           # H2 structure strength
    crt_lookback: int = 5     # a sweep stays valid for this many H4/H2 candles ...
    cisd_bars: int = 6        # ... and the CISD must come within this many 30m bars after it
    use_pd: bool = False      # False: send A+ and B setups (grade by zone). True: A+ only
    atr_len: int = 14
    atr_mult: float = 1.2     # impulsive: CISD candle range >= atr_mult * ATR (previous bar)
    body_min: float = 0.6     # impulsive: body >= body_min * range
    vol_len: int = 20
    vol_mult: float = 1.5     # bonus only (logged)


# ----------------------------------------------------------------------------
def period_end(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    """Close time of the bar that opens at `ts`."""
    if tf == "1M":
        return ts + pd.offsets.MonthBegin(1)
    if tf == "1w":
        return ts + pd.Timedelta(days=7)
    if tf == "1d":
        return ts + pd.Timedelta(days=1)
    return ts + pd.Timedelta(minutes=MIN[tf])


def drop_forming(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Remove the last bar if it has not closed yet."""
    if len(df) and period_end(df.index[-1], tf) > now:
        return df.iloc[:-1]
    return df


def grade(direction: str, zones: dict):
    """A+ when the swept timeframe is in the right zone (low sweep in DISCOUNT, high sweep in PREMIUM),
    else B. If H4 and H2 both swept, H4 decides. Returns (grade, tf_code, zone)."""
    tf = "4h" if "4h" in zones else "2h"
    want = "DISCOUNT" if direction == "BUY" else "PREMIUM"
    return ("A+" if zones[tf] == want else "B"), tf, zones[tf]


def analyze(df: pd.DataFrame, n: int, tf: str):
    """One pass over a higher-timeframe frame.

    Returns (ctx, sweeps):
      ctx    DataFrame per bar: b (bias), zone, label (latest event), new (event on this bar), close_time
      sweeps list of dicts: CRT high/low sweeps (prior-candle rule from the Pine script)
    Bias/zone logic is identical to indicator.f_bias."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    N = len(df)
    ph, pl = ind.pivots(h, l, n)
    hi = lo = float("nan")
    hib = lob = False
    b, label = 0, "—"
    rows, sweeps = [], []
    for k in range(N):
        if not math.isnan(ph[k]):
            hi, hib = ph[k], False
        if not math.isnan(pl[k]):
            lo, lob = pl[k], False
        bos = None
        if not math.isnan(hi) and not hib and c[k] > hi:
            hib, b, bos = True, 1, "BOS Bullish"
        elif not math.isnan(lo) and not lob and c[k] < lo:
            lob, b, bos = True, -1, "BOS Bearish"
        zone = "N/A" if (math.isnan(hi) or math.isnan(lo)) else ("PREMIUM" if c[k] > (hi + lo) * 0.5 else "DISCOUNT")

        hs = ls = False
        if k >= 1:
            crh, crl = h[k - 1], l[k - 1]
            bull = c[k] > o[k]
            hs = h[k] > crh and ((c[k] < crh) if bull else (o[k] < crh))
            ls = l[k] < crl and ((o[k] > crl) if bull else (c[k] > crl))
            if hs:
                sweeps.append(dict(k=k, kind="high", extreme=h[k], zone=zone, tf=tf))
            if ls:
                sweeps.append(dict(k=k, kind="low", extreme=l[k], zone=zone, tf=tf))
        sw = None
        if hs and not ls:
            sw = "Sweep High"
        elif ls and not hs:
            sw = "Sweep Low"
        elif hs and ls:
            sw = "Sweep Low" if c[k] > o[k] else "Sweep High"
        new = bos is not None or sw is not None
        if bos:
            label = bos
        elif sw:
            label = sw
        rows.append((b, zone, label, new))
    ctx = pd.DataFrame(rows, columns=["b", "zone", "label", "new"], index=df.index)
    ctx["close_time"] = [period_end(t, tf) for t in df.index]
    return ctx, sweeps


def map_ctx(bclose: pd.DatetimeIndex, ctx: pd.DataFrame | None):
    """Value of the latest CLOSED higher-timeframe bar for every base bar (no look-ahead)."""
    N = len(bclose)
    b = np.zeros(N, dtype=int)
    zone = np.array(["N/A"] * N, dtype=object)
    label = np.array(["—"] * N, dtype=object)
    new = np.zeros(N, dtype=bool)
    if ctx is None or not len(ctx):
        return b, zone, label, new
    left = pd.DataFrame({"t": bclose.values})
    right = ctx.reset_index(drop=True)[["close_time", "b", "zone", "label", "new"]].copy()
    right["close_time"] = pd.DatetimeIndex(right["close_time"]).values
    m = pd.merge_asof(left, right.sort_values("close_time"), left_on="t", right_on="close_time", direction="backward")
    ok = m["b"].notna().to_numpy()
    b[ok] = m.loc[ok, "b"].astype(int).to_numpy()
    zone[ok] = m.loc[ok, "zone"].to_numpy()
    label[ok] = m.loc[ok, "label"].to_numpy()
    new[ok] = m.loc[ok, "new"].astype(bool).to_numpy()
    return b, zone, label, new


def _ns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.as_unit("ns")
    return df


# ----------------------------------------------------------------------------
def find_signals(base: pd.DataFrame, h2, h4, daily, weekly, cfg: Cfg | None = None, return_arms: bool = False):
    """All signals found in `base` (closed 30m bars). Frames must contain closed bars only.
    With return_arms=True returns (signals, sweeps) where sweeps are the armed H4/H2 sweeps (CISD pending)."""
    cfg = cfg or Cfg()
    N = len(base)
    if N < cfg.atr_len + 3 or daily is None or len(daily) < 4:
        return []
    base = _ns(base)
    idx = base.index
    o, h, l, c = (base[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = base["volume"].to_numpy(float) if "volume" in base else np.zeros(N)
    step = idx.to_series().diff().median()
    bclose = idx[1:].append(pd.DatetimeIndex([idx[-1] + step]))

    # ---- context of higher timeframes at every base bar -------------------------
    ctx_d, _ = analyze(_ns(daily), cfg.d_len, "1d")
    dB, dZ, dL, dN = map_ctx(bclose, ctx_d)
    if weekly is not None and len(weekly) >= 4:
        ctx_w, _ = analyze(_ns(weekly), cfg.w_len, "1w")
    else:
        ctx_w = None
    wB, wZ, wL, wN = map_ctx(bclose, ctx_w)
    if h4 is not None and len(h4) >= 2 * cfg.h4_len + 3:
        ctx_h4, sw_h4 = analyze(_ns(h4), cfg.h4_len, "4h")
    else:
        ctx_h4, sw_h4 = None, []
    hB, hZ, hL, hN = map_ctx(bclose, ctx_h4)
    if h2 is not None and len(h2) >= 2 * cfg.h2_len + 3:
        _, sw_h2 = analyze(_ns(h2), cfg.h2_len, "2h")
    else:
        sw_h2 = []

    # ---- ATR / volume ----------------------------------------------------------------
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(alpha=1 / cfg.atr_len, adjust=False, min_periods=cfg.atr_len).mean().to_numpy()
    vsma = pd.Series(v).rolling(cfg.vol_len).mean().to_numpy()

    # ---- arm points: base bar at which each H4/H2 sweep candle closes --------------------
    arms: dict[int, list] = {}
    for tf, df, sweeps in (("2h", h2, sw_h2), ("4h", h4, sw_h4)):
        if df is None or not sweeps:
            continue
        di = df.index.as_unit("ns")
        per = period_end(di[0], tf) - di[0]
        for ev in sweeps:
            if cfg.use_pd and ((ev["kind"] == "low" and ev["zone"] != "DISCOUNT") or
                               (ev["kind"] == "high" and ev["zone"] != "PREMIUM")):
                continue
            close_t = period_end(di[ev["k"]], tf)
            j = int(bclose.searchsorted(close_t))
            if j >= N or bclose[j] - close_t > step:
                continue                                    # candle closed outside our base data
            arms.setdefault(j, []).append({**ev, "per": per})

    # ---- main loop (same order as the Pine CISD logic) -----------------------------------
    last_bear_open = last_bull_open = float("nan")
    bull = bear = None
    out: list[dict] = []
    arms_out: list[dict] = []
    for i in range(N):
        po, pcl = (o[i - 1], c[i - 1]) if i else (float("nan"), float("nan"))
        if c[i] < o[i] and (pcl >= po or math.isnan(last_bear_open)):
            last_bear_open = o[i]
        if c[i] > o[i] and (pcl <= po or math.isnan(last_bull_open)):
            last_bull_open = o[i]

        for ev in arms.get(i, []):
            if ev["kind"] == "low" and dB[i] == 1:
                lvl = o[i] if c[i] < o[i] else (h[i] if math.isnan(last_bear_open) else last_bear_open)
                if bull is None or bull["arm"] != i:
                    bull = dict(arm=i, lvl=lvl, px=ev["extreme"], ext=ev["extreme"], tfs={ev["tf"]}, per=ev["per"], zones={ev["tf"]: ev["zone"]})
                else:
                    bull["tfs"].add(ev["tf"]); bull["zones"][ev["tf"]] = ev["zone"]; bull["px"] = min(bull["px"], ev["extreme"]); bull["ext"] = min(bull["ext"], ev["extreme"])
                    bull["per"] = max(bull["per"], ev["per"])
            elif ev["kind"] == "high" and dB[i] == -1:
                lvl = o[i] if c[i] > o[i] else (l[i] if math.isnan(last_bull_open) else last_bull_open)
                if bear is None or bear["arm"] != i:
                    bear = dict(arm=i, lvl=lvl, px=ev["extreme"], ext=ev["extreme"], tfs={ev["tf"]}, per=ev["per"], zones={ev["tf"]: ev["zone"]})
                else:
                    bear["tfs"].add(ev["tf"]); bear["zones"][ev["tf"]] = ev["zone"]; bear["px"] = max(bear["px"], ev["extreme"]); bear["ext"] = max(bear["ext"], ev["extreme"])
                    bear["per"] = max(bear["per"], ev["per"])

        for stt, d in ((bull, "BUY"), (bear, "SELL")):
            if stt is not None and stt["arm"] == i:
                g, gtf, gz = grade(d, stt["zones"])
                arms_out.append(dict(i=i, time=idx[i], dir=d, grade=g, sweep_tf=TF_NAME[gtf], sweep_zone=gz,
                                     swept=sorted((TF_NAME[t] for t in stt["tfs"]), reverse=True)))

        for st in (bull, bear):
            if st is not None:
                st["ext"] = min(st["ext"], l[i]) if st is bull else max(st["ext"], h[i])
        if bull and (i - bull["arm"] > cfg.cisd_bars or idx[i] - idx[bull["arm"]] > cfg.crt_lookback * bull["per"]):
            bull = None
        if bear and (i - bear["arm"] > cfg.cisd_bars or idx[i] - idx[bear["arm"]] > cfg.crt_lookback * bear["per"]):
            bear = None

        rng = h[i] - l[i]
        atr_prev = atr[i - 1] if i > 0 else float("nan")
        if rng <= 0 or math.isnan(atr_prev) or atr_prev <= 0:
            continue
        atr_x, body = rng / atr_prev, abs(c[i] - o[i]) / rng
        impulsive = atr_x >= cfg.atr_mult and body >= cfg.body_min
        vol_x = v[i] / vsma[i - 1] if i > 0 and not math.isnan(vsma[i - 1]) and vsma[i - 1] > 0 else float("nan")

        for st, direction in ((bull, "BUY"), (bear, "SELL")):
            if st is None or i <= st["arm"] or not impulsive:
                continue
            if direction == "BUY":
                hit = dB[i] == 1 and c[i] > o[i] and c[i] > st["lvl"]
                fvg = i >= 2 and l[i] > h[i - 2]
            else:
                hit = dB[i] == -1 and c[i] < o[i] and c[i] < st["lvl"]
                fvg = i >= 2 and h[i] < l[i - 2]
            if not hit:
                continue
            g, gtf, gz = grade(direction, st["zones"])
            out.append(dict(
                i=i, time=idx[i], dir=direction, grade=g, sweep_tf=TF_NAME[gtf], sweep_zone=gz,
                arm_time=idx[st["arm"]], sweep_px=float(st["px"]), entry=float(c[i]), sl_sweep=float(st["ext"]),
                swept=sorted((TF_NAME[t] for t in st["tfs"]), reverse=True),   # H4 before H2
                atr_x=float(atr_x), body=float(body), fvg=bool(fvg), vol_x=float(vol_x),
                ctx={"Weekly": (int(wB[i]), wZ[i], wL[i], bool(wN[i])),
                     "Daily": (int(dB[i]), dZ[i], dL[i], bool(dN[i])),
                     "H4": (int(hB[i]), hZ[i], hL[i], bool(hN[i]))}))
            if direction == "BUY":
                bull = None
            else:
                bear = None
    return (out, arms_out) if return_arms else out
