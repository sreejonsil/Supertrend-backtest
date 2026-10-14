# pip install yfinance pandas numpy
import numpy as np, pandas as pd

PERIOD, MULT = 10, 3.0   # supertrend settings
FEE = 0.0003             # 0.03% per side
TP = 0.30                # take profit at +30% from entry (None = off)
SL = 0.10                # stop loss at -10% from entry (None = off)
MAX_DAYS = 20            # max holding period in trading days (None = off). Try 15 or 20

STOCKS = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE
BAJAJFINSV BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH
HDFCBANK HDFCLIFE HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JSWSTEEL JIOFIN
KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN
SHRIRAMFIN SUNPHARMA TATACONSUM TMPV TATASTEEL TCS TECHM TITAN TRENT
ULTRACEMCO WIPRO""".split()

def supertrend(df, n, m):
    h, l, c = df.High.values, df.Low.values, df.Close.values
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    hl2 = (h + l) / 2
    ub, lb = hl2 + m * atr, hl2 - m * atr
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(c))  # 1 = green, -1 = red
    for i in range(1, len(c)):
        fub[i] = ub[i] if ub[i] < fub[i-1] or c[i-1] > fub[i-1] else fub[i-1]
        flb[i] = lb[i] if lb[i] > flb[i-1] or c[i-1] < flb[i-1] else flb[i-1]
        if d[i-1] == 1:
            d[i] = -1 if c[i] < flb[i] else 1
        else:
            d[i] = 1 if c[i] > fub[i] else -1
    return d

def backtest(df):
    d = supertrend(df, PERIOD, MULT)
    o, h, l, c = df.Open.values, df.High.values, df.Low.values, df.Close.values
    cap, pos, entry, blocked = 1.0, False, 0.0, False
    eq, trades, tp_hits, sl_hits = [1.0], [], 0, 0
    cap_in = 1.0
    entry_day, time_exits = 0, 0
    try:
        day_no = pd.factorize(df.index.date)[0]      # trading-day number of each candle
    except Exception:
        day_no = np.arange(len(c)) // 7
    for i in range(1, len(c)):
        sig = d[i-1]                      # signal from previous closed candle
        if sig == -1:
            blocked = False               # after a take-profit, wait for a fresh green
        if not pos and sig == 1 and not blocked:
            pos, entry = True, o[i]       # enter at next candle open
            cap_in = cap
            entry_day = day_no[i]
            cap *= (1 - FEE)
        if pos:
            exit_px = None
            if SL and l[i] <= entry * (1 - SL):   # stop checked first (conservative)
                exit_px = min(entry * (1 - SL), o[i]); sl_hits += 1; blocked = True
            elif TP and h[i] >= entry * (1 + TP):
                exit_px = max(entry * (1 + TP), o[i]); tp_hits += 1; blocked = True
            elif sig == -1:
                exit_px = o[i]
            elif MAX_DAYS and day_no[i] - entry_day >= MAX_DAYS:
                exit_px = o[i]; time_exits += 1; blocked = True
            if exit_px is not None:
                cap = cap * (exit_px / entry) * (1 - FEE)
                trades.append(cap / cap_in - 1)
                pos = False
                eq.append(cap); continue
            eq.append(cap * c[i] / entry)
        else:
            eq.append(cap)
    if pos:                               # close open trade at last price
        cap = cap * (c[-1] / entry) * (1 - FEE)
        trades.append(cap / cap_in - 1)
    eq = pd.Series(eq)
    t = np.array(trades)
    wins, losses = t[t > 0], t[t <= 0]
    return dict(Trades=len(t),
                Win=len(wins), Loss=len(losses),
                WinPct=round(len(wins) / len(t) * 100, 1) if len(t) else 0,
                LossPct=round(len(losses) / len(t) * 100, 1) if len(t) else 0,
                AvgWin=round(wins.mean() * 100, 1) if len(wins) else 0,
                AvgLoss=round(losses.mean() * 100, 1) if len(losses) else 0,
                TPHits=tp_hits, SLHits=sl_hits, TimeExits=time_exits,
                StrategyPct=round((cap - 1) * 100, 1),
                BuyHoldPct=round((c[-1] / c[0] - 1) * 100, 1),
                MaxDD=round((eq / eq.cummax() - 1).min() * 100, 1))

import sys, io, time, urllib.request

URLS = {
    "nifty200": ["https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                 "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"],
    "midcap100": ["https://niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
                  "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap100list.csv"],
}

def get_universe(name):
    if name == "nifty50":
        return STOCKS
    for url in URLS[name]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read()
            syms = pd.read_csv(io.BytesIO(raw))["Symbol"].str.strip().tolist()
            print("Loaded", len(syms), "symbols for", name)
            return syms
        except Exception as e:
            print("list download failed:", url, e)
    sys.exit("Could not download the stock list")

if __name__ == "__main__":
    import yfinance as yf
    name = sys.argv[1] if len(sys.argv) > 1 else "nifty50"
    rows = []
    for s in get_universe(name):
        try:
            df = yf.download(s + ".NS", period="730d", interval="1h",
                             auto_adjust=True, progress=False)
            df.columns = [x[0] if isinstance(x, tuple) else x for x in df.columns]
            df = df.dropna()
            if len(df) < 100: raise ValueError("not enough data")
            rows.append({"Stock": s, **backtest(df)})
        except Exception as e:
            print("skip", s, e)
        time.sleep(0.3)
    res = pd.DataFrame(rows).sort_values("StrategyPct", ascending=False)
    print(res.to_string(index=False))
    tot_w, tot_l = res.Win.sum(), res.Loss.sum()
    print("\nOVERALL: winning trades", round(tot_w / (tot_w + tot_l) * 100, 1),
          "% | losing trades", round(tot_l / (tot_w + tot_l) * 100, 1), "%")
    print("\nStocks tested:", len(res),
          "| Average strategy %:", round(res.StrategyPct.mean(), 1),
          "| Average buy&hold %:", round(res.BuyHoldPct.mean(), 1),
          "| Profitable stocks:", (res.StrategyPct > 0).sum(), "/", len(res))
    res.to_csv(f"{name}_supertrend_results.csv", index=False)
