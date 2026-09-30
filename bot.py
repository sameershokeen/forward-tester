TG_BOT_TOKEN = "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo"
TG_CHAT_ID   = "1950462171"

"""
Heikin Ashi x HMA(50) Crossover - FORWARD TEST (paper trading) + Telegram

* Separate $100 paper account for 3m and 5m
* 100% of equity per trade, compounding, 1 open trade per timeframe
* Round-trip cost 0.04% charged on every trade
* Signal: HA close crosses above HMA(50) | SL: HA low of signal candle | TP: 5R
* Every 24h a report goes to Telegram: P&L, trades, wins, losses, win rate, EV
* State is saved to forward_state.json, so restarting the script is safe

Install:  pip install ccxt pandas numpy requests
Run:      python ha_hma_forward_test.py
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

# ============================ SETTINGS ============================
TG_BOT_TOKEN = "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo"
TG_CHAT_ID   = "1950462171"

EXCHANGE_ID   = "binance"
SYMBOL        = "SOL/USDT"
TIMEFRAMES    = ["3m", "5m"]
HMA_LEN       = 25
TP_MULT       = 10
CANDLES       = 300
POLL_SECS     = 10

START_CAPITAL = 100.0
ROUND_TRIP    = 0.0004          # 0.04% per round trip
REPORT_EVERY  = 24 * 60 * 60    # seconds
NOTIFY_SIGNAL = True            # send entry alert
NOTIFY_CLOSE  = True            # send a message when each trade closes
SINGLE_TRADE_ONLY = False       # True = only ONE trade at a time across 3m AND 5m
                                # (next signal is taken only after the open trade completes)
STATE_FILE    = "forward_state.json"
# ==================================================================


# ---------------------------- telegram ----------------------------
def tg(text: str):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ---------------------------- indicators ----------------------------
def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    return wma(2 * wma(s, int(n / 2)) - wma(s, n), int(round(math.sqrt(n))))


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["ts"] = df["ts"]
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    o = np.zeros(len(df))
    o[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        o[i] = (o[i - 1] + ha["close"].iloc[i - 1]) / 2
    ha["open"] = o
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def fetch_ohlc(ex, tf: str) -> pd.DataFrame:
    raw = ex.fetch_ohlcv(SYMBOL, timeframe=tf, limit=CANDLES)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("time", inplace=True)
    return df.iloc[:-1]  # closed candles only


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    ha = heikin_ashi(df)
    ha["hma"] = hma(ha["close"], HMA_LEN)
    ha["signal"] = (ha["close"] > ha["hma"]) & (ha["close"].shift(1) <= ha["hma"].shift(1))
    return ha


# ---------------------------- state ----------------------------
def new_state() -> dict:
    now = time.time()
    return {
        "start_ts": now,
        "next_report": now + REPORT_EVERY,
        "last_report_ts": now,
        "accounts": {
            tf: {"equity": START_CAPITAL, "open": None, "trades": [], "last_seen": None}
            for tf in TIMEFRAMES
        },
    }


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            st = json.load(f)
        for tf in TIMEFRAMES:
            st["accounts"].setdefault(
                tf, {"equity": START_CAPITAL, "open": None, "trades": [], "last_seen": None}
            )
        return st
    return new_state()


def save_state(st: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


# ---------------------------- paper trading ----------------------------
def close_trade(acct: dict, tr: dict, exit_price: float, reason: str, ts_ms: int, tf: str):
    pnl_pct = exit_price / tr["entry"] - 1 - ROUND_TRIP
    pnl_usd = tr["equity_before"] * pnl_pct
    risk_pct = (tr["entry"] - tr["sl"]) / tr["entry"]
    acct["equity"] = tr["equity_before"] + pnl_usd
    acct["trades"].append({
        "tf": tf,
        "entry": tr["entry"], "exit": exit_price, "sl": tr["sl"], "tp": tr["tp"],
        "reason": reason,
        "pnl_usd": pnl_usd, "pnl_pct": pnl_pct * 100,
        "r": pnl_pct / risk_pct if risk_pct > 0 else 0.0,
        "open_ts": tr["open_ts"], "close_ts": ts_ms,
    })
    acct["open"] = None
    if NOTIFY_CLOSE:
        icon = "✅" if pnl_usd > 0 else "❌"
        tg(f"{icon} <b>{tf} trade closed ({reason})</b>\n"
           f"Entry {tr['entry']:.4f} → Exit {exit_price:.4f}\n"
           f"P&L: {pnl_usd:+.2f} USD ({pnl_pct*100:+.2f}%)\n"
           f"Equity: {acct['equity']:.2f} USD")


def check_open_trade(acct: dict, df: pd.DataFrame, tf: str):
    """Walk through closed real candles after entry. SL wins if both hit in one candle."""
    tr = acct["open"]
    if not tr:
        return
    for _, c in df[df["ts"] > tr["checked"]].iterrows():
        tr["checked"] = int(c["ts"])
        if c["low"] <= tr["sl"]:
            close_trade(acct, tr, tr["sl"], "SL", int(c["ts"]), tf)
            return
        if c["high"] >= tr["tp"]:
            close_trade(acct, tr, tr["tp"], "TP", int(c["ts"]), tf)
            return


def open_trade(acct: dict, ex, ha: pd.DataFrame, tf: str, st: dict):
    if acct["open"]:
        return  # pyramiding = 0
    if SINGLE_TRADE_ONLY and any(a["open"] for a in st["accounts"].values()):
        return  # wait until the running trade completes
    row = ha.iloc[-1]
    try:
        entry = float(ex.fetch_ticker(SYMBOL)["last"])
    except Exception:
        return
    sl = float(row["low"])
    if entry <= sl:
        return  # price already below stop, skip
    tp = entry + TP_MULT * (entry - sl)
    acct["open"] = {
        "entry": entry, "sl": sl, "tp": tp,
        "equity_before": acct["equity"],
        "open_ts": int(time.time() * 1000),
        "checked": int(row["ts"]),
    }
    if NOTIFY_SIGNAL:
        tg(f"🟢 <b>BUY SIGNAL</b> | {SYMBOL} | {tf} | Heikin Ashi\n"
           f"Entry: <code>{entry:.4f}</code>\n"
           f"SL: <code>{sl:.4f}</code>\n"
           f"TP ({TP_MULT}R): <code>{tp:.4f}</code>")


# ---------------------------- reporting ----------------------------
def stats(trades: list) -> dict:
    n = len(trades)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    gross_win = sum(t["pnl_usd"] for t in wins)
    gross_loss = -sum(t["pnl_usd"] for t in losses)
    return {
        "n": n, "wins": len(wins), "losses": len(losses),
        "winrate": len(wins) / n * 100 if n else 0.0,
        "pnl": sum(t["pnl_usd"] for t in trades),
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
        "ev_usd": sum(t["pnl_usd"] for t in trades) / n if n else 0.0,
        "ev_r": sum(t["r"] for t in trades) / n if n else 0.0,
        "pf": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
    }


def block(title: str, s: dict, equity: float, cap: float) -> str:
    pf = "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (
        f"<b>{title}</b>\n"
        f"Trades: {s['n']} | Wins: {s['wins']} | Losses: {s['losses']}\n"
        f"Win rate: {s['winrate']:.1f}%\n"
        f"P&L: {s['pnl']:+.2f} USD\n"
        f"Avg win: {s['avg_win']:+.2f} | Avg loss: {s['avg_loss']:+.2f}\n"
        f"EV/trade: {s['ev_usd']:+.3f} USD ({s['ev_r']:+.2f}R)\n"
        f"Profit factor: {pf}\n"
        f"Equity: {equity:.2f} USD ({(equity / cap - 1) * 100:+.2f}%)\n"
    )


def send_report(st: dict):
    since_ms = st["last_report_ts"] * 1000
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"📊 <b>24H FORWARD TEST REPORT</b>\n{SYMBOL} | HA HMA{HMA_LEN} | {now}\n"
    msg += f"Capital: {START_CAPITAL:.0f} USD per timeframe | Cost: {ROUND_TRIP*100:.2f}% round trip\n\n"

    all24, all_time, total_equity = [], [], 0.0
    for tf in TIMEFRAMES:
        a = st["accounts"][tf]
        t24 = [t for t in a["trades"] if t["close_ts"] >= since_ms]
        all24 += t24
        all_time += a["trades"]
        total_equity += a["equity"]
        msg += block(f"{tf} (last 24h)", stats(t24), a["equity"], START_CAPITAL)
        msg += f"Open trade: {'yes' if a['open'] else 'no'}\n\n"

    msg += block("COMBINED (last 24h)", stats(all24),
                 total_equity, START_CAPITAL * len(TIMEFRAMES))
    msg += "\n"
    msg += block("COMBINED (all time)", stats(all_time),
                 total_equity, START_CAPITAL * len(TIMEFRAMES))
    tg(msg)

    st["last_report_ts"] = time.time()
    st["next_report"] = time.time() + REPORT_EVERY


# ---------------------------- main ----------------------------
def main():
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    st = load_state()
    save_state(st)
    tg(f"🚀 Forward test started: {SYMBOL} {TIMEFRAMES} | {START_CAPITAL:.0f} USD each | "
       f"{ROUND_TRIP*100:.2f}% round trip. First report in 24h.")

    while True:
        for tf in TIMEFRAMES:
            try:
                acct = st["accounts"][tf]
                df = fetch_ohlc(ex, tf)
                ha = analyse(df)

                check_open_trade(acct, df, tf)

                candle_ts = int(ha["ts"].iloc[-1])
                if acct["last_seen"] != candle_ts:
                    first_run = acct["last_seen"] is None
                    acct["last_seen"] = candle_ts
                    if not first_run and bool(ha.iloc[-1]["signal"]):
                        open_trade(acct, ex, ha, tf, st)
                save_state(st)
            except Exception:
                pass

        if time.time() >= st["next_report"]:
            send_report(st)
            save_state(st)
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
  
