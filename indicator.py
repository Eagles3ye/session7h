"""Python port of the Pine v6 indicator
"7H Session Profile + MTF Structure + H4 CRT [30m]".

`run(df, htf, params)` walks the bars once, in the same order as the Pine script
(structure -> sessions -> 50% invalidation -> sweep/CISD) and records everything
that Pine would have drawn (boxes, lines, labels) plus the dashboard table.
Rendering is left to the app; this module has no Streamlit/Plotly dependency.

Bar indices in the output are positions in `df` (0 = oldest bar).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NAN = float("nan")


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ----------------------------------------------------------------------------
# Parameters (the logic-affecting Pine inputs; show_* toggles live in the app)
# ----------------------------------------------------------------------------
@dataclass
class Params:
    interval: str = "30m"                 # chart timeframe of `df`
    tz: str = "Asia/Kolkata"
    asia_sess: str = "0430-1130"
    lon_sess: str = "1130-1830"
    ny_sess: str = "1830-0130"
    m_len: int = 1
    w_len: int = 1
    d_len: int = 1
    h4_len: int = 3
    major_len: int = 3
    sweep_bars: int = 5
    max_bars_cisd: int = 6
    req_disp: bool = True
    strict_trend: bool = True


@dataclass
class Result:
    times: pd.DatetimeIndex                # bar open time in `tz` (tz-naive, for plotting)
    boxes: list = field(default_factory=list)         # session boxes
    liq: dict | None = None                           # prior-session high/low lines
    invs: list = field(default_factory=list)          # 50% invalidation lines
    struct_lines: list = field(default_factory=list)  # BOS / sweep lines
    dots: list = field(default_factory=list)          # recent major pivots
    eq: dict | None = None                            # dealing-range EQ line
    crt_lines: list = field(default_factory=list)
    sweeps: list = field(default_factory=list)        # "Sweep" labels
    cisd: list = field(default_factory=list)          # CISD lines + labels
    table: list = field(default_factory=list)         # dashboard rows
    state: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """ta.pivothigh / ta.pivotlow(n, n). Value is reported on the bar where the pivot
    is *confirmed* (n bars after the pivot bar). Left side strict, right side allows ties."""
    N = len(high)
    ph = np.full(N, np.nan)
    pl = np.full(N, np.nan)
    for i in range(2 * n, N):
        c = i - n
        hc, lc = high[c], low[c]
        if (hc > high[c - n:c]).all() and (hc >= high[c + 1:i + 1]).all():
            ph[i] = hc
        if (lc < low[c - n:c]).all() and (lc <= low[c + 1:i + 1]).all():
            pl[i] = lc
    return ph, pl


def f_bias(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pine f_bias(): BOS-based bias + premium/discount zone, one row per bar of `df`."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    ph, pl = pivots(h, l, n)
    hi = lo = NAN
    hi_broken = lo_broken = False
    b, ev = 0, "NEUTRAL"
    out_b, out_ev, out_z = [], [], []
    for i in range(len(df)):
        if not math.isnan(ph[i]):
            hi, hi_broken = ph[i], False
        if not math.isnan(pl[i]):
            lo, lo_broken = pl[i], False
        if not math.isnan(hi) and not hi_broken and c[i] > hi:
            hi_broken, b, ev = True, 1, "BOS BULL"
        elif not math.isnan(lo) and not lo_broken and c[i] < lo:
            lo_broken, b, ev = True, -1, "BOS BEAR"
        if math.isnan(hi) or math.isnan(lo):
            z = "N/A"
        else:
            z = "PREM" if c[i] > (hi + lo) * 0.5 else "DISC"
        out_b.append(b); out_ev.append(ev); out_z.append(z)
    return pd.DataFrame({"b": out_b, "ev": out_ev, "zone": out_z}, index=df.index)


def map_htf(base_index: pd.DatetimeIndex, htf: pd.DataFrame | None, n: int):
    """request.security(..., lookahead_off) equivalent.

    A higher-timeframe bar's result becomes visible on the first base bar whose CLOSE is at or
    after that HTF bar's close (so no look-ahead). The very last base bar gets the still-forming
    HTF bar, like Pine on a realtime bar."""
    N = len(base_index)
    b = np.zeros(N, dtype=int)
    ev = np.array(["NEUTRAL"] * N, dtype=object)
    zn = np.array(["N/A"] * N, dtype=object)
    if htf is None or len(htf) < 2 * n + 2:
        return b, ev, zn
    res = f_bias(htf, n)
    htf_close = res.index.to_series().shift(-1)
    res = res.assign(close_time=htf_close.values).dropna(subset=["close_time"])
    bopen = base_index.to_series()
    step = bopen.diff().median()
    bclose = bopen.shift(-1)
    bclose.iloc[-1] = bopen.iloc[-1] + step
    left = pd.DataFrame({"t": bclose.values.astype("datetime64[ns]")})
    right = res[["close_time", "b", "ev", "zone"]].copy()
    right["close_time"] = right["close_time"].astype("datetime64[ns]")
    right = right.sort_values("close_time")
    m = pd.merge_asof(left, right, left_on="t", right_on="close_time", direction="backward")
    ok = m["b"].notna().to_numpy()
    b[ok] = m.loc[ok, "b"].astype(int).to_numpy()
    ev[ok] = m.loc[ok, "ev"].to_numpy()
    zn[ok] = m.loc[ok, "zone"].to_numpy()
    live = f_bias(htf, n).iloc[-1]
    b[-1], ev[-1], zn[-1] = int(live["b"]), live["ev"], live["zone"]
    return b, ev, zn


def _sess(s: str) -> tuple[int, int]:
    a, b = s.split("-")
    return int(a[:2]) * 60 + int(a[2:]), int(b[:2]) * 60 + int(b[2:])


def _in_sess(mins: np.ndarray, s: int, e: int) -> np.ndarray:
    return (mins >= s) & (mins < e) if s < e else (mins >= s) | (mins < e)


def fmt(x: float) -> str:
    if isnan(x):
        return "n/a"
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:.4f}"


# ----------------------------------------------------------------------------
# Main engine
# ----------------------------------------------------------------------------
def run(df: pd.DataFrame, htf: dict, p: Params) -> Result:
    """df: base OHLC (UTC index). htf: {'M','1W','1D','H4'} -> OHLC DataFrame or None."""
    N = len(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    local = df.index.tz_convert(p.tz)
    times = local.tz_localize(None)
    mins = (local.hour * 60 + local.minute).to_numpy()
    res = Result(times=times)

    # ---- MTF bias ----------------------------------------------------------
    mB, mEv, mZ = map_htf(df.index, htf.get("M"), p.m_len)
    wB, wEv, wZ = map_htf(df.index, htf.get("1W"), p.w_len)
    dB, dEv, dZ = map_htf(df.index, htf.get("1D"), p.d_len)
    hB, hEv, hZ = map_htf(df.index, htf.get("H4"), p.h4_len)
    is_m, is_w, is_h4 = p.interval == "1M", p.interval == "1w", p.interval == "4h"
    tf_tag = "M" if is_m else "1W" if is_w else "H4" if is_h4 else "1D"
    tf_b, tf_ev, tf_z = ((mB, mEv, mZ) if is_m else (wB, wEv, wZ) if is_w
                         else (hB, hEv, hZ) if is_h4 else (dB, dEv, dZ))

    # ---- sessions masks ----------------------------------------------------
    in_asia = _in_sess(mins, *_sess(p.asia_sess))
    in_lon = _in_sess(mins, *_sess(p.lon_sess))
    in_ny = _in_sess(mins, *_sess(p.ny_sess))

    # ---- major pivots ------------------------------------------------------
    SH, SL = pivots(h, l, p.major_len)
    ML = p.major_len

    last_sh = last_sl = NAN
    last_sh_idx = last_sl_idx = 0
    sh_break = sl_break = False
    last_event, event_col, next_target = "Initializing Structure...", "gray", "Waiting for Event..."
    dealing_h = dealing_l = NAN

    crt_status, crt_col = ("H4 Only", "gray")

    # ---- session state -----------------------------------------------------
    arr_swept = ["-"] * 3
    arr_res = ["-"] * 3
    arr_res_col = ["gray"] * 3
    arr_has = [False] * 3
    last_rev_txt, last_rev_col = "none yet", "gray"
    valid_prof, valid_col = "none yet", "gray"

    sess_h = sess_l = sess_o = sess_c = NAN
    sess_start, sess_name = None, ""
    prev_h = prev_l = prev_o = prev_c = NAN
    prev_start = prev_end = None
    prev_name = ""
    curr_box = None
    inv_lvl, inv_dir, inv_alive, inv_name, cur_inv = NAN, 0, False, "", None

    # ---- CISD state ----------------------------------------------------------
    last_bear_leg_open = last_bull_leg_open = NAN
    bull_armed = bear_armed = False
    bull_cisd_lvl = bull_sweep_low = NAN
    bear_cisd_lvl = bear_sweep_high = NAN
    bull_sweep_bar = bear_sweep_bar = -10**9

    recent_from = N - 1 - 100
    crt_from = N - 1 - p.sweep_bars

    for i in range(N):
        # ================= 3. structure ======================================
        if not math.isnan(SH[i]):
            last_sh, last_sh_idx, sh_break = SH[i], i - ML, False
            dealing_h = SH[i]
            if i >= recent_from:
                res.dots.append(dict(idx=i - ML, y=SH[i], kind="bear"))
        if not math.isnan(SL[i]):
            last_sl, last_sl_idx, sl_break = SL[i], i - ML, False
            dealing_l = SL[i]
            if i >= recent_from:
                res.dots.append(dict(idx=i - ML, y=SL[i], kind="bull"))

        if not math.isnan(last_sh) and not sh_break and h[i] > last_sh:
            if c[i] > last_sh:
                sh_break = True
                last_event, event_col = "BOS Bullish (Body Close)", "bull"
                next_target = "Target: Next External High / FVG"
                res.struct_lines.append(dict(x0=last_sh_idx, x1=i, y=last_sh, color="bull", dash=False))
            else:
                last_event, event_col = "LIQUIDITY SWEEP HIGH (Wick)", "bear"
                next_target = "Target: EQ (50%) -> Range Low"
                res.struct_lines.append(dict(x0=last_sh_idx, x1=i, y=last_sh, color="bear", dash=True))

        if not math.isnan(last_sl) and not sl_break and l[i] < last_sl:
            if c[i] < last_sl:
                sl_break = True
                last_event, event_col = "BOS Bearish (Body Close)", "bear"
                next_target = "Target: Next External Low / FVG"
                res.struct_lines.append(dict(x0=last_sl_idx, x1=i, y=last_sl, color="bear", dash=False))
            else:
                last_event, event_col = "LIQUIDITY SWEEP LOW (Wick)", "bull"
                next_target = "Target: EQ (50%) -> Range High"
                res.struct_lines.append(dict(x0=last_sl_idx, x1=i, y=last_sl, color="bull", dash=True))

        # H4 CRT sweeps filtered by H4 premium / discount
        if is_h4:
            crt_status = f"H4 ({hZ[i]}) - Idle"
            crt_col = "bear" if hZ[i] == "PREM" else "bull" if hZ[i] == "DISC" else "gray"
            if i >= crt_from and i >= 1:
                crh, crl = h[i - 1], l[i - 1]
                bull_bar = c[i] > o[i]
                bear_sweep = (c[i] < crh) if bull_bar else (o[i] < crh)
                bull_sweep = (o[i] > crl) if bull_bar else (c[i] > crl)
                if hZ[i] == "PREM" and h[i] > crh and bear_sweep:
                    crt_status, crt_col = "HIGH SWEEP (H4 PREM)", "bear"
                    res.crt_lines.append(dict(x0=i - 1, x1=i, y=crh))
                if hZ[i] == "DISC" and l[i] < crl and bull_sweep:
                    crt_status, crt_col = "LOW SWEEP (H4 DISC)", "bull"
                    res.crt_lines.append(dict(x0=i - 1, x1=i, y=crl))

        # ================= 4. sessions =======================================
        new_asia = in_asia[i] and not (i > 0 and in_asia[i - 1])
        new_lon = in_lon[i] and not (i > 0 and in_lon[i - 1])
        new_ny = in_ny[i] and not (i > 0 and in_ny[i - 1])
        active = 0 if in_asia[i] else 1 if in_lon[i] else 2 if in_ny[i] else -1

        if new_asia:
            arr_swept = ["-"] * 3
            arr_res = ["-"] * 3
            arr_res_col = ["gray"] * 3
            arr_has = [False] * 3
            valid_prof, valid_col = "none yet", "gray"

        if new_asia or new_lon or new_ny:
            prev_h, prev_l, prev_o, prev_c = sess_h, sess_l, sess_o, sess_c
            prev_name, prev_start, prev_end = sess_name, sess_start, i - 1
            name = "ASIA" if new_asia else "LONDON" if new_lon else "NEW YORK"
            curr_box = dict(name=name, start=i, end=i, high=h[i], low=l[i])
            res.boxes.append(curr_box)
            sess_h, sess_l, sess_o, sess_c = h[i], l[i], o[i], c[i]
            sess_start, sess_name = i, name

            # NOTE: the Pine source draws these from the NEW session's first bar; the label says
            # "Prior Session High/Low", so here they come from the prior session.
            if not isnan(prev_h) and prev_start is not None:
                res.liq = dict(x0=prev_start, x1=i, high=prev_h, low=prev_l)

            if not isnan(prev_h) and not isnan(prev_l) and not isnan(prev_c):
                inv_name = prev_name
                if prev_c > prev_o:
                    inv_dir, inv_lvl, inv_alive = 1, (prev_l + prev_c) * 0.5, True
                elif prev_c < prev_o:
                    inv_dir, inv_lvl, inv_alive = -1, (prev_h + prev_c) * 0.5, True
                else:
                    inv_dir, inv_alive = 0, False
                if inv_dir != 0:
                    cur_inv = dict(x0=prev_start, x1=prev_end, y=inv_lvl, dir=inv_dir,
                                   name=inv_name, broken=False)
                    res.invs.append(cur_inv)

        if active != -1:
            sess_h = max(h[i] if isnan(sess_h) else sess_h, h[i])
            sess_l = min(l[i] if isnan(sess_l) else sess_l, l[i])
            sess_c = c[i]
            if curr_box is not None:
                curr_box.update(end=i, high=sess_h, low=sess_l)

        if res.liq is not None:
            res.liq["x1"] = i

        # ================= 5. 50% invalidation ===============================
        if inv_alive and not isnan(inv_lvl):
            broken = (inv_dir == 1 and c[i] < inv_lvl) or (inv_dir == -1 and c[i] > inv_lvl)
            if broken:
                inv_alive = False
                if cur_inv is not None:
                    cur_inv["broken"] = True
                if active != -1:
                    prefix = "Swept - " if arr_has[active] else "50% "
                    arr_res[active] = prefix + ("Buy Setup Invalidated" if inv_dir == 1 else "Sell Setup Invalidated")
                    arr_res_col[active] = "bear"

        # ================= 6. sweep + CISD ===================================
        po, pc = (o[i - 1], c[i - 1]) if i > 0 else (NAN, NAN)
        if c[i] < o[i] and (pc >= po or isnan(last_bear_leg_open)):
            last_bear_leg_open = o[i]
        if c[i] > o[i] and (pc <= po or isnan(last_bull_leg_open)):
            last_bull_leg_open = o[i]

        allow_bull = (not p.strict_trend) or dB[i] == 1
        allow_bear = (not p.strict_trend) or dB[i] == -1
        t_str = times[i].strftime("%d %b %H:%M")
        act_name = "ASIA" if in_asia[i] else "LONDON" if in_lon[i] else "NEW YORK" if in_ny[i] else "OFF-SESSION"

        if not isnan(prev_l) and l[i] < prev_l and c[i] > prev_l and allow_bull:
            bull_armed, bull_sweep_low, bull_sweep_bar = True, l[i], i
            bull_cisd_lvl = o[i] if c[i] < o[i] else (h[i] if isnan(last_bear_leg_open) else last_bear_leg_open)
            res.sweeps.append(dict(idx=i, y=l[i], dir=1))
            if active != -1:
                cur = arr_swept[active]
                arr_has[active] = True
                arr_swept[active] = "HIGH + LOW" if cur in ("HIGH", "HIGH + LOW") else "LOW"
                arr_res[active] = "Swept - Buy Setup Invalidated" if not inv_alive else "Swept - Waiting CISD"
                arr_res_col[active] = "bear" if not inv_alive else "orange"
        if bull_armed:
            bull_sweep_low = min(bull_sweep_low, l[i])
            if i - bull_sweep_bar > p.max_bars_cisd:
                bull_armed = False

        if not isnan(prev_h) and h[i] > prev_h and c[i] < prev_h and allow_bear:
            bear_armed, bear_sweep_high, bear_sweep_bar = True, h[i], i
            bear_cisd_lvl = o[i] if c[i] > o[i] else (l[i] if isnan(last_bull_leg_open) else last_bull_leg_open)
            res.sweeps.append(dict(idx=i, y=h[i], dir=-1))
            if active != -1:
                cur = arr_swept[active]
                arr_has[active] = True
                arr_swept[active] = "HIGH + LOW" if cur in ("LOW", "HIGH + LOW") else "HIGH"
                arr_res[active] = "Swept - Sell Setup Invalidated" if not inv_alive else "Swept - Waiting CISD"
                arr_res_col[active] = "bear" if not inv_alive else "orange"
        if bear_armed:
            bear_sweep_high = max(bear_sweep_high, h[i])
            if i - bear_sweep_bar > p.max_bars_cisd:
                bear_armed = False

        rng = h[i] - l[i]
        solid = (not p.req_disp) or (rng > 0 and abs(c[i] - o[i]) / rng >= 0.4)

        if bull_armed and c[i] > bull_cisd_lvl and c[i] > o[i] and solid and i > bull_sweep_bar:
            bull_armed = False
            res.cisd.append(dict(dir=1, x0=bull_sweep_bar, x1=i, lvl=bull_cisd_lvl, idx=i, y=l[i],
                                 text=f"CISD BUY (Trend OK)<br>SL: {fmt(bull_sweep_low)}"))
            last_rev_txt, last_rev_col = f"{act_name} - BUY @ {t_str}", "bull"
            valid_prof, valid_col = f"{act_name} Reversal (Bullish)", "bull"
            if active != -1:
                arr_res[active], arr_res_col[active] = "CISD BUY Confirmed", "bull"

        if bear_armed and c[i] < bear_cisd_lvl and c[i] < o[i] and solid and i > bear_sweep_bar:
            bear_armed = False
            res.cisd.append(dict(dir=-1, x0=bear_sweep_bar, x1=i, lvl=bear_cisd_lvl, idx=i, y=h[i],
                                 text=f"CISD SELL (Trend OK)<br>SL: {fmt(bear_sweep_high)}"))
            last_rev_txt, last_rev_col = f"{act_name} - SELL @ {t_str}", "bear"
            valid_prof, valid_col = f"{act_name} Reversal (Bearish)", "bear"
            if active != -1:
                arr_res[active], arr_res_col[active] = "CISD SELL Confirmed", "bear"

    # ---- dealing range EQ --------------------------------------------------
    eq = (dealing_h + dealing_l) * 0.5 if not (isnan(dealing_h) or isnan(dealing_l)) else NAN
    if not isnan(eq):
        res.eq = dict(x0=N - 1 - 15, x1=N - 1 + 5, y=eq)
    active_zone = "N/A" if isnan(eq) else ("PREMIUM" if c[-1] > eq else "DISCOUNT")
    zone_col = "gray" if isnan(eq) else ("bear" if c[-1] > eq else "bull")

    # ---- dashboard ---------------------------------------------------------
    last_active = 0 if in_asia[-1] else 1 if in_lon[-1] else 2 if in_ny[-1] else -1
    names, targets = ["ASIA", "LONDON", "NEW YORK"], ["Prev NY H/L", "Asia H/L", "Asia + London H/L"]
    rows = [dict(kind="hdr", cells=[("Session", "white"), ("Sweeps", "white"), ("Swept", "white"), ("Result", "white")])]
    for k in range(3):
        sw = arr_swept[k]
        sw_c = "bull" if "LOW" in sw else "bear" if "HIGH" in sw else "dim"
        rows.append(dict(kind="row", cells=[
            (("> " if last_active == k else "") + names[k], "act" if last_active == k else "white"),
            (targets[k], "dim"), (sw, sw_c), (arr_res[k], arr_res_col[k])]))

    tb = int(tf_b[-1])
    tf_txt = f"{'BULLISH' if tb == 1 else 'BEARISH' if tb == -1 else 'NEUTRAL'} ({tf_tag}) | {tf_ev[-1]} | {tf_z[-1]}"
    rows.append(dict(kind="merged", cells=[("MTF Bias", "dim"), (tf_txt, "bull" if tb == 1 else "bear" if tb == -1 else "dim")]))
    rows.append(dict(kind="merged", cells=[("CRT (H4)", "dim"), (crt_status, crt_col)]))
    eq_txt = f"EQ: {fmt(eq)} ({active_zone})" if not isnan(eq) else "EQ: N/A"
    rows.append(dict(kind="merged", cells=[("Range / EQ", "dim"), (eq_txt, zone_col)]))
    rows.append(dict(kind="merged", cells=[("Structure", "dim"), (last_event, event_col)]))
    rows.append(dict(kind="merged", cells=[("Last reversal", "dim"), (last_rev_txt, last_rev_col)]))
    if not isnan(inv_lvl):
        l50 = f"{inv_name} 50% = {fmt(inv_lvl)}" + ("" if inv_alive else (" (Buy Setup Invalidated)" if inv_dir == 1 else " (Sell Setup Invalidated)"))
        l50_c = "act" if inv_alive else "bear"
    else:
        l50, l50_c = "none yet", "dim"
    rows.append(dict(kind="merged", cells=[("50% line", "dim"), (l50, l50_c)]))
    rows.append(dict(kind="merged", cells=[("Valid profile", "dim"), (valid_prof, valid_col)]))
    res.table = rows

    res.state = dict(
        d_bias=int(dB[-1]), d_event=dEv[-1], d_zone=dZ[-1], next_target=next_target,
        w_bias=int(wB[-1]), m_bias=int(mB[-1]), h4_bias=int(hB[-1]), h4_zone=hZ[-1],
    )
    return res
