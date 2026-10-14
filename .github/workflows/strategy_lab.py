# Tests many entry strategies + filters with a fixed target/stoploss (e.g. 10% target, 3.33% stop = 1:3).
# Usage: python strategy_lab.py nifty200 1h
import sys, time, itertools
import numpy as np, pandas as pd
import nifty50_supertrend_backtest as b

FEE = 0.0003
RR_SETS = [(0.10, 0.0333), (0.10, 0.05)]     # (target, stoploss): 1:3 and 1:2
MAX_DAYS = [10, 20]                           # trading days
FILTER_SETS = [(), ("trend",), ("trend", "adx"), ("trend", "vol"),
               ("st", "trend"), ("adx", "vol"), ("trend", "adx", "vol")]

def cross_up(a, b_):
    return ((a > b_) & (a.shift(1) <= b_.shift(1))).fillna(False).values

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def adx(df, n=14):
    up, dn = df.High.diff(), -df.Low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def build(df):
    c = df.Close
    e20, e50, e200 = (c.ewm(span=x, adjust=False).mean() for x in (20, 50, 200))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    r = rsi(c)
    hh = df.High.rolling(20).max().shift(1)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    st = b.supertrend(df, 10, 3.0)
    vol = df.Volume if "Volume" in df.columns else pd.Series(1.0, index=df.index)
    events = {
        "ST_flip":  np.r_[False, (st[1:] == 1) & (st[:-1] == -1)],
        "EMA20x50": cross_up(e20, e50),
        "MACD_x":   cross_up(macd, sig),
        "RSI_bounce": cross_up(r, pd.Series(40.0, index=df.index)),
        "Donchian20": ((c > hh) & (c.shift(1) <= hh.shift(1))).fillna(False).values,
        "BB_break": cross_up(c, mid + 2 * sd),
    }
    filt = {
        "trend": (c > e200).fillna(False).values,
        "adx": (adx(df) > 20).fillna(False).values,
        "vol": (vol > 1.5 * vol.rolling(20).mean()).fillna(False).values,
        "st": st == 1,
    }
    try:    day = pd.factorize(df.index.date)[0]
    except Exception: day = np.arange(len(df)) // 7
    return dict(o=df.Open.values, h=df.High.values, l=df.Low.values, c=c.values,
                day=day, events=events, filt=filt, n=len(df))

def sim(sig_idx, D, start, end, tp, sl, maxd):
    o, h, l, c, day = D["o"], D["h"], D["l"], D["c"], D["day"]
    rets, free = [], start
    for i in sig_idx:
        if i < free: continue
        j = i + 1
        if j >= end: break
        e = o[j]; tpx, slx, d0 = e * (1 + tp), e * (1 - sl), day[j]
        k = j
        while k < end:
            if k > j and day[k] - d0 >= maxd: px = o[k]; break
            if l[k] <= slx: px = min(slx, o[k]); break      # stop checked first (cautious)
            if h[k] >= tpx: px = max(tpx, o[k]); break
            k += 1
        else:
            px = c[end - 1]; k = end - 1
        rets.append(px / e * (1 - FEE) ** 2 - 1)
        free = k
    return rets

if __name__ == "__main__":
    import yfinance as yf
    name = sys.argv[1] if len(sys.argv) > 1 else "nifty50"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1h"
    data = {}
    for s in b.get_universe(name):
        if s == "BAJFINANCE": continue            # split not adjusted in free data
        try:
            df = yf.download(s + ".NS", period="730d", interval="1h", auto_adjust=True, progress=False)
            df.columns = [x[0] if isinstance(x, tuple) else x for x in df.columns]
            df = df.dropna()
            if tf == "4h": df = b.to_4h(df)
            if len(df) > 250: data[s] = build(df)
        except Exception as e:
            print("skip", s, e)
        time.sleep(0.3)
    print("Stocks loaded:", len(data))

    rows = []
    for ev, fs, (tp, sl), md in itertools.product(list(next(iter(data.values()))["events"]),
                                                  FILTER_SETS, RR_SETS, MAX_DAYS):
        out = {}
        for half in (1, 2):
            wins = losses = 0; tr, stock_rets = [], []
            for D in data.values():
                n = D["n"]; start, end = (0, n // 2) if half == 1 else (n // 2, n)
                mask = D["events"][ev].copy()
                for f in fs: mask &= D["filt"][f]
                idx = np.nonzero(mask[start:end])[0] + start
                r = sim(idx, D, start, end, tp, sl, md)
                if r:
                    tr += r; wins += sum(x > 0 for x in r); losses += sum(x <= 0 for x in r)
                    stock_rets.append(np.prod(1 + np.array(r)) - 1)
            nt = wins + losses
            out[half] = (round(wins / nt * 100, 1) if nt else 0,
                         round(np.mean(tr) * 100, 2) if nt else 0,
                         round(np.mean(stock_rets) * 100, 1) if stock_rets else 0, nt)
        rows.append(dict(Entry=ev, Filters="+".join(fs) or "none", TP=tp, SL=sl, MaxDays=md,
                         Win_Y1=out[1][0], Win_Y2=out[2][0],
                         PerTrade_Y1=out[1][1], PerTrade_Y2=out[2][1],
                         StockRet_Y1=out[1][2], StockRet_Y2=out[2][2],
                         Trades=out[1][3] + out[2][3]))
    res = pd.DataFrame(rows)
    res["MinWin"] = res[["Win_Y1", "Win_Y2"]].min(axis=1)
    res["MinPerTrade"] = res[["PerTrade_Y1", "PerTrade_Y2"]].min(axis=1)
    pd.set_option("display.width", 250)
    cols = [c for c in res.columns if c not in ("MinWin", "MinPerTrade")]
    ok = res[(res.MinPerTrade > 0) & (res.Trades >= 150)].sort_values("MinPerTrade", ascending=False)
    print("\nTOP 15: PROFITABLE IN BOTH YEARS (ranked by profit per trade, min 150 trades)")
    print(ok[cols].head(15).to_string(index=False) if len(ok) else "None found")
    hi = res[(res.MinWin >= 60) & (res.Trades >= 150)].sort_values("MinPerTrade", ascending=False)
    print("\nWIN RATE >= 60% IN BOTH YEARS (min 150 trades)")
    print(hi[cols].head(15).to_string(index=False) if len(hi) else "None found")
    print("\nHIGHEST WIN RATES (any profit)")
    print(res.sort_values("MinWin", ascending=False)[cols].head(10).to_string(index=False))
    print("\nBreak-even win rate: 25% for 1:3 (SL 3.33%), 33% for 1:2 (SL 5%)")
    res.drop(columns=["MinWin", "MinPerTrade"]).to_csv(f"{name}_{tf}_lab_results.csv", index=False)
