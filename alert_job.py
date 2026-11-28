"""Scheduled alert job: scans the markets, sends setups to Telegram.

    python alert_job.py                 normal run (GitHub Actions calls this every 30 min)
    python alert_job.py --history       print every signal in the loaded data, send nothing
    python alert_job.py --dry-run       like a normal run but prints the messages instead of sending
    python alert_job.py --test-telegram send one test message

Environment: TWELVE_KEY (forex), TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import requests

import strategy as st
import vela_data as vd

# ============================== SETTINGS =====================================
SYMBOLS = ["twelve:EUR/USD", "twelve:GBP/USD", "twelve:NZD/USD", "PAXGUSDT", "BTCUSDT", "ETHUSDT"]
DISPLAY = {"PAXGUSDT": "XAU/USD (PAXG)", "BTCUSDT": "BTC/USD", "ETHUSDT": "ETH/USD"}   # names in Telegram
CRYPTO = {"BTCUSDT", "ETHUSDT"}      # no pips: stop = sweep extreme, target = RR x risk, shown in %

SL_PIPS = 30        # fixed stop for forex/gold, pips from entry. Set None to use the sweep extreme instead
TP_PIPS = 60        # fixed target for forex/gold, pips from entry
RR = 2.0            # target multiple when the stop is the sweep extreme (crypto, or SL_PIPS=None)
PIP = {"XAU": 0.1, "PAXG": 0.1, "BTC": 0.1, "ETH": 0.1}   # gold: 1 pip = 0.10. Forex 0.0001 (JPY 0.01)

SWEEP_UPDATES = False        # True = also send "sweep detected, waiting for CISD" heads-up messages
SWEEP_UPDATE_GRADES = {"A+"}  # which sweeps get a heads-up if SWEEP_UPDATES is True
BASE_TF, BASE_LIMIT = "30m", 600
ALERT_WITHIN_BARS = 2        # only alert for events from the last N closed 30m bars
CFG = st.Cfg()               # all strategy settings live in strategy.Cfg
# ==============================================================================

CACHE = Path(".cache")
LOG = Path("signals_log.csv")
LOG_COLS = ["kind", "time_utc", "symbol", "dir", "grade", "swept", "entry", "atr_x", "body", "fvg", "vol_x", "sent_utc"]
HTF_LIMITS = {"2h": 300, "4h": 300, "1d": 400, "1w": 200}
MIN_REFETCH = pd.Timedelta(minutes=55)


# ------------------------------------------------------------------ helpers -----
def is_forex(sym: str) -> bool:
    return sym.startswith(("twelve:", "oanda:"))


def fx_open(now: pd.Timestamp) -> bool:
    """Forex is closed from Friday 17:00 to Sunday 17:00 New York time."""
    ny = now.tz_convert("America/New_York")
    wd, hr = ny.weekday(), ny.hour
    return not (wd == 5 or (wd == 4 and hr >= 17) or (wd == 6 and hr < 17))


def pip_size(sym: str, price: float) -> float:
    s = sym.upper()
    for key, val in PIP.items():
        if key in s:
            return val
    return 0.01 if "JPY" in s else 0.0001


def fmt_price(x: float, pip: float) -> str:
    dec = max(0, int(-math.floor(math.log10(pip)))) + 1
    return f"{x:,.{dec}f}"


def name_of(sym: str) -> str:
    return DISPLAY.get(sym, sym.split(":", 1)[-1])


# ---------------------------------------------------------------- data + cache ---
def _cache_path(sym: str, tf: str) -> Path:
    return CACHE / (sym.replace(":", "_").replace("/", "") + f"_{tf}.csv")


def _meta() -> dict:
    p = CACHE / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def get_htf(sym: str, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Closed higher-timeframe bars. Re-fetched only when a new bar can have closed."""
    CACHE.mkdir(exist_ok=True)
    path, key, meta = _cache_path(sym, tf), f"{sym}|{tf}", _meta()
    df = None
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").as_unit("ns")
    if df is not None and len(df):
        nxt = st.period_end(st.period_end(df.index[-1], tf), tf)      # when the next bar will have closed
        fetched = pd.Timestamp(meta[key]) if key in meta else None
        if now < nxt or (fetched is not None and now - fetched < MIN_REFETCH):
            return df
    df = st.drop_forming(vd.get_bars(sym, tf, HTF_LIMITS[tf]), tf, now)
    df.to_csv(path)
    meta[key] = now.isoformat()
    (CACHE / "meta.json").write_text(json.dumps(meta))
    return df


def scan(sym: str, now: pd.Timestamp):
    base = st.drop_forming(vd.get_bars(sym, BASE_TF, BASE_LIMIT), BASE_TF, now)
    h2, h4 = get_htf(sym, "2h", now), get_htf(sym, "4h", now)
    daily, weekly = get_htf(sym, "1d", now), get_htf(sym, "1w", now)
    sigs, arms = st.find_signals(base, h2, h4, daily, weekly, CFG, return_arms=True)
    return base, sigs, arms


# -------------------------------------------------------------------- messages ----
def build_message(sym: str, sig: dict) -> str:
    buy = sig["dir"] == "BUY"
    pip = pip_size(sym, sig["entry"])
    entry = sig["entry"]
    fixed = bool(SL_PIPS) and sym not in CRYPTO
    if fixed:
        sl = entry - SL_PIPS * pip if buy else entry + SL_PIPS * pip
        tp = entry + TP_PIPS * pip if buy else entry - TP_PIPS * pip
    else:
        sl = sig["sl_sweep"]
        risk = abs(entry - sl)
        tp = entry + RR * risk if buy else entry - RR * risk
    if sym in CRYPTO:
        sl_txt, tp_txt = f"-{abs(entry - sl) / entry * 100:.2f}%", f"+{abs(tp - entry) / entry * 100:.2f}%"
    else:
        sl_txt, tp_txt = f"-{abs(entry - sl) / pip:.0f} pips", f"+{abs(tp - entry) / pip:.0f} pips"

    want_zone = "DISCOUNT" if buy else "PREMIUM"
    want_bias = 1 if buy else -1
    lines_ctx, agree, new = [], [], []
    for name in ("Weekly", "Daily", "H4"):
        b, zone, label, is_new = sig["ctx"][name]
        bias = "Bullish" if b == 1 else "Bearish" if b == -1 else "Neutral"
        lines_ctx.append(f"{name}: {bias} / {zone} ({label})")
        if b == want_bias and zone == want_zone:
            agree.append(name)
        if is_new:
            new.append(name)

    side = "Bullish" if buy else "Bearish"
    return "\n".join([
        ("🟢 " if buy else "🔴 ") + f"{side.upper()} setup — {name_of(sym)}",
        f"{side} bias + {want_zone} on: {', '.join(agree) if agree else '—'}",
        f"New: {', '.join(new) if new else '—'}",
        "",
        *lines_ctx,
        f"{'/'.join(sig['swept'])} : Sweep {fmt_price(sig['sweep_px'], pip)} & {sig['grade']}",
        f"M30 : CISD entry : {fmt_price(entry, pip)}",
        f"SL : {fmt_price(sl, pip)} ({sl_txt})",
        f"Take Profit : {fmt_price(tp, pip)} ({tp_txt})",
    ])


def build_sweep_message(sym: str, a: dict) -> str:
    buy = a["dir"] == "BUY"
    return "\n".join([
        f"👀 SWEEP — {name_of(sym)}",
        f"{'Bullish' if buy else 'Bearish'}: {'/'.join(a['swept'])} swept {'low' if buy else 'high'} "
        f"({a['sweep_tf']} {a['sweep_zone']}) — grade {a['grade']} if CISD confirms",
        f"Waiting for 30m CISD (within {CFG.cisd_bars} bars)",
    ])


def send_telegram(text: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=20)
    if r.status_code != 200 or not r.json().get("ok"):
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:200]}")


# ------------------------------------------------------------------------ log ----
def sent_keys() -> set:
    if not LOG.exists():
        return set()
    with LOG.open(newline="") as f:
        return {f'{r.get("kind", "ENTRY")}|{r["symbol"]}|{r["dir"]}|{r["time_utc"]}' for r in csv.DictReader(f)}


def append_log(kind: str, sym: str, item: dict, now: pd.Timestamp) -> None:
    new = not LOG.exists()
    nan = float("nan")
    with LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        vol = item.get("vol_x", nan)
        w.writerow([kind, item["time"].isoformat(), sym, item["dir"], item["grade"], "/".join(item["swept"]),
                    f'{item["entry"]:.5f}' if "entry" in item else "",
                    f'{item["atr_x"]:.2f}' if "atr_x" in item else "",
                    f'{item["body"]:.2f}' if "body" in item else "",
                    int(item["fvg"]) if "fvg" in item else "",
                    "" if (vol is None or math.isnan(vol)) else f"{vol:.2f}", now.isoformat()])


def deliver(kind: str, sym: str, item: dict, msg: str, key: str, done: set, now, dry: bool) -> bool:
    """Send (or print) one message and log it. Returns False if sending failed (retried next run)."""
    if dry:
        print("\n" + msg + "\n")
        return True
    try:
        send_telegram(msg)
    except RuntimeError as e:
        print(f"{sym}: {e}")
        return False
    append_log(kind, sym, item, now)
    done.add(key)
    print(f"{sym}: sent {kind} {item['dir']} {item['grade']}")
    return True


# ----------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-telegram", action="store_true")
    a = ap.parse_args()

    if a.test_telegram:
        send_telegram("✅ Alert bot connected")
        print("test message sent")
        return 0

    vd.configure(twelve_key=os.environ.get("TWELVE_KEY"))
    vd.CONFIG["twelve_min_gap"] = 7.8                      # stay under 8 requests/min on the free plan
    now = pd.Timestamp.now(tz="UTC")
    done, failed = sent_keys(), 0

    for sym in SYMBOLS:
        if is_forex(sym) and not fx_open(now) and not a.history:
            print(f"{sym}: forex market closed, skipped")
            continue
        try:
            base, sigs, arms = scan(sym, now)
        except (vd.DataError, requests.RequestException) as e:
            print(f"{sym}: DATA ERROR {e}")
            failed += 1
            continue
        if not len(base):
            print(f"{sym}: no data")
            continue
        if a.history:
            print(f"\n{sym}: {len(base)} bars, {len(sigs)} signals "
                  f"(A+ {sum(s['grade'] == 'A+' for s in sigs)}, B {sum(s['grade'] == 'B' for s in sigs)})")
            for s in sigs:
                print(f'  {s["time"]:%Y-%m-%d %H:%M} UTC  {s["dir"]:4} {s["grade"]:2}  {"/".join(s["swept"]):6}  '
                      f'entry {s["entry"]:.5g}  ATRx {s["atr_x"]:.2f}  body {s["body"]:.2f}  '
                      f'FVG {"Y" if s["fvg"] else "-"}')
            continue
        age = now - base.index[-1]
        if age > pd.Timedelta(hours=3) and (not is_forex(sym) or fx_open(now)):
            print(f"{sym}: WARNING newest closed bar is {age} old")
        cut = len(base) - ALERT_WITHIN_BARS
        recent = [s for s in sigs if s["i"] >= cut]
        print(f"{sym}: {len(sigs)} signals in data, {len(recent)} recent")

        if SWEEP_UPDATES:                                   # heads-up for sweeps still waiting for a CISD
            has_entry = {(s["dir"], s["arm_time"]) for s in sigs}
            for arm in (x for x in arms if x["i"] >= cut):
                key = f'SWEEP|{sym}|{arm["dir"]}|{arm["time"].isoformat()}'
                if key in done or (arm["dir"], arm["time"]) in has_entry or arm["grade"] not in SWEEP_UPDATE_GRADES:
                    continue
                if not deliver("SWEEP", sym, arm, build_sweep_message(sym, arm), key, done, now, a.dry_run):
                    failed += 1

        for s in recent:
            key = f'ENTRY|{sym}|{s["dir"]}|{s["time"].isoformat()}'
            if key in done:
                continue
            if not deliver("ENTRY", sym, s, build_message(sym, s), key, done, now, a.dry_run):
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
