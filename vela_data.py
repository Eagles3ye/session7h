"""Candlestick data layer, modelled on LuxAlgo Vela's bundled providers.

Vela is a JavaScript charting library and has no Python/REST API of its own.
Its bundled providers (Binance, Coinbase, Hyperliquid) just call public exchange
endpoints, so this module calls the same endpoints and uses the same symbol
conventions:

    BTCUSDT            Binance spot            (default when no prefix)
    BTCUSDT.P          Binance USDT perpetual
    coinbase:BTC-USD   Coinbase Exchange spot
    hyperliquid:BTC    Hyperliquid perp  (spot pairs like PURR/USDC also work)
    oanda:XAU_USD      OANDA forex / metals (needs OANDA_TOKEN)  e.g. oanda:EUR_USD, oanda:NZD_USD

Like Vela, timeframes an exchange does not serve natively (45m, 3h, ...) are
aggregated from a smaller native timeframe; weekly/monthly are built from daily
where the exchange has no native bars.

All frames returned have a tz-aware UTC DatetimeIndex (bar OPEN time, ns unit)
and columns: open, high, low, close, volume.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

UA = {"User-Agent": "session-profile-streamlit/1.0"}
TIMEOUT = 15
MAX_PAGES = 25

# minutes per interval (1w / 1M handled separately)
MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
       "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440, "3d": 4320}
PERIOD_MS = {**{k: v * 60_000 for k, v in MIN.items()},
             "1w": 7 * 86_400_000, "1M": 31 * 86_400_000}


class DataError(RuntimeError):
    pass


def _frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (open_time_ms, open, high, low, close, volume)"""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True).dt.as_unit("ns")
    df = df.drop(columns="t").set_index("time").astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


# ----------------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------------
class Provider:
    name = ""
    native: set[str] = set()

    def page(self, symbol: str, interval: str, end_ms: int | None) -> pd.DataFrame:
        raise NotImplementedError


class Binance(Provider):
    """Spot + USDT-M perpetuals. Falls back across mirrors like Vela does
    (api.binance.com is geo-blocked from US hosts such as Streamlit Cloud)."""
    name = "binance"
    native = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
    SPOT = ["https://api.binance.com", "https://data-api.binance.vision", "https://api.binance.us"]
    PERP = ["https://fapi.binance.com"]

    def page(self, symbol, interval, end_ms):
        perp = symbol.endswith(".P")
        sym = symbol[:-2] if perp else symbol
        hosts, path = (self.PERP, "/fapi/v1/klines") if perp else (self.SPOT, "/api/v3/klines")
        params = {"symbol": sym, "interval": interval, "limit": 1000}
        if end_ms:
            params["endTime"] = int(end_ms)
        errs = []
        for host in hosts:
            try:
                r = requests.get(host + path, params=params, headers=UA, timeout=TIMEOUT)
            except requests.RequestException as e:
                errs.append(f"{host}: {type(e).__name__}")
                continue
            if r.status_code == 200:
                return _frame([(k[0], k[1], k[2], k[3], k[4], k[5]) for k in r.json()])
            errs.append(f"{host}: HTTP {r.status_code}")
        raise DataError(f"Binance {symbol} {interval} failed ({'; '.join(errs)})")


class Coinbase(Provider):
    name = "coinbase"
    GRAN = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
    native = set(GRAN)
    HOST = "https://api.exchange.coinbase.com"

    def page(self, symbol, interval, end_ms):
        g = self.GRAN[interval]
        end = datetime.fromtimestamp((end_ms or int(datetime.now(timezone.utc).timestamp() * 1000)) / 1000, tz=timezone.utc)
        start = end - timedelta(seconds=g * 300)
        try:
            r = requests.get(f"{self.HOST}/products/{symbol}/candles",
                             params={"granularity": g, "start": start.isoformat(), "end": end.isoformat()},
                             headers=UA, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise DataError(f"Coinbase {symbol}: {type(e).__name__}") from e
        if r.status_code != 200:
            raise DataError(f"Coinbase {symbol} {interval}: HTTP {r.status_code} {r.text[:120]}")
        # [time(s), low, high, open, close, volume], newest first
        return _frame([(k[0] * 1000, k[3], k[2], k[1], k[4], k[5]) for k in r.json()])


class Hyperliquid(Provider):
    name = "hyperliquid"
    native = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"}
    HOST = "https://api.hyperliquid.xyz/info"

    def page(self, symbol, interval, end_ms):
        end = int(end_ms or datetime.now(timezone.utc).timestamp() * 1000)
        start = end - 5000 * PERIOD_MS[interval]
        body = {"type": "candleSnapshot",
                "req": {"coin": symbol, "interval": interval, "startTime": start, "endTime": end}}
        try:
            r = requests.post(self.HOST, json=body, headers=UA, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise DataError(f"Hyperliquid {symbol}: {type(e).__name__}") from e
        if r.status_code != 200:
            raise DataError(f"Hyperliquid {symbol} {interval}: HTTP {r.status_code}")
        return _frame([(k["t"], k["o"], k["h"], k["l"], k["c"], k["v"]) for k in r.json()])


class Oanda(Provider):
    """OANDA v20 REST (forex + metals, e.g. oanda:EUR_USD, oanda:XAU_USD). Needs an API token.

    Daily/weekly/monthly and H4+ candles are aligned to 17:00 America/New_York, the same
    convention TradingView uses for forex."""
    name = "oanda"
    GRAN = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "2h": "H2", "4h": "H4",
            "6h": "H6", "8h": "H8", "12h": "H12", "1d": "D", "1w": "W", "1M": "M"}
    native = set(GRAN)
    HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}

    @staticmethod
    def instrument(symbol: str) -> str:
        s = symbol.upper().replace("/", "_").replace("-", "_")
        return f"{s[:3]}_{s[3:]}" if "_" not in s and len(s) == 6 else s     # XAUUSD -> XAU_USD

    def page(self, symbol, interval, end_ms):
        token = CONFIG.get("oanda_token")
        if not token:
            raise DataError("OANDA token missing. Set OANDA_TOKEN in Streamlit secrets (see README).")
        host = self.HOSTS.get(CONFIG.get("oanda_env", "practice"), self.HOSTS["practice"])
        params = {"granularity": self.GRAN[interval], "count": 5000, "price": "M",
                  "dailyAlignment": 17, "alignmentTimezone": "America/New_York"}
        if end_ms:
            params["to"] = f"{end_ms / 1000:.3f}"
        try:
            r = requests.get(f"{host}/v3/instruments/{self.instrument(symbol)}/candles", params=params,
                             headers={**UA, "Authorization": f"Bearer {token}"}, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise DataError(f"OANDA {symbol}: {type(e).__name__}") from e
        if r.status_code == 400 and end_ms:        # paged past the start of history
            return _frame([])
        if r.status_code in (401, 403):
            raise DataError("OANDA rejected the token (HTTP %d). Check OANDA_TOKEN and that OANDA_ENV "
                            "('practice' or 'live') matches the account the token was created on." % r.status_code)
        if r.status_code != 200:
            raise DataError(f"OANDA {symbol} {interval}: HTTP {r.status_code} {r.text[:150]}")
        rows = []
        for k in r.json().get("candles", []):
            m = k.get("mid")
            if m:
                rows.append((int(round(pd.Timestamp(k["time"]).timestamp() * 1000)),
                             m["o"], m["h"], m["l"], m["c"], k.get("volume", 0)))
        return _frame(rows)


# Runtime config (token etc.). The app fills this from Streamlit secrets; env vars also work.
CONFIG: dict = {"oanda_token": os.environ.get("OANDA_TOKEN"),
                "oanda_env": os.environ.get("OANDA_ENV", "practice")}


def configure(**kw) -> None:
    CONFIG.update({k: v for k, v in kw.items() if v})


PROVIDERS: dict[str, Provider] = {p.name: p for p in (Binance(), Coinbase(), Hyperliquid(), Oanda())}


def resolve(symbol: str) -> tuple[Provider, str]:
    """'coinbase:BTC-USD' -> (Coinbase, 'BTC-USD'); bare symbols go to Binance."""
    s = symbol.strip()
    if ":" in s:
        venue, sym = s.split(":", 1)
        venue = venue.lower()
        if venue not in PROVIDERS:
            raise DataError(f"Unknown venue '{venue}'. Use one of: {', '.join(PROVIDERS)}")
        return PROVIDERS[venue], sym.strip()
    return PROVIDERS["binance"], s.upper()


# ----------------------------------------------------------------------------
# Resampling + paging
# ----------------------------------------------------------------------------
def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if interval == "1w":
        r = df.resample("W-MON", label="left", closed="left")      # weeks start Monday 00:00 UTC
    elif interval == "1M":
        r = df.resample("MS")
    else:
        r = df.resample(f"{MIN[interval]}min", origin="epoch")
    out = r.agg(agg).dropna(subset=["open"])
    out.index = out.index.as_unit("ns")
    return out


def _source_interval(prov: Provider, interval: str) -> tuple[str, int]:
    """Pick the native interval to aggregate from and the aggregation factor."""
    if interval in ("1w", "1M"):
        return "1d", (7 if interval == "1w" else 31)
    target = MIN[interval]
    cands = [(MIN[n], n) for n in prov.native if n in MIN and MIN[n] < target and target % MIN[n] == 0]
    if not cands:
        raise DataError(f"{prov.name} cannot provide {interval} bars")
    m, n = max(cands)
    return n, target // m


def _paginate(prov: Provider, sym: str, interval: str, limit: int) -> pd.DataFrame:
    frames, got, end = [], 0, None
    for _ in range(MAX_PAGES):
        df = prov.page(sym, interval, end)
        if df.empty:
            break
        frames.append(df)
        got += len(df)
        if got >= limit:
            break
        new_end = int(df.index[0].timestamp() * 1000) - 1
        if end is not None and new_end >= end:
            break
        end = new_end
    if not frames:
        raise DataError(f"No candles returned for {prov.name}:{sym} {interval}")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.tail(limit)


def get_bars(symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """Fetch `limit` most-recent bars for a Vela-style symbol and interval."""
    prov, sym = resolve(symbol)
    if interval in prov.native:
        return _paginate(prov, sym, interval, limit)
    src, factor = _source_interval(prov, interval)
    raw = _paginate(prov, sym, src, limit * factor + factor)
    return resample(raw, interval).tail(limit)
