# pip install yfinance pandas numpy
import numpy as np, pandas as pd, yfinance as yf

PERIOD, MULT = 10, 3.0      # supertrend settings
FEE = 0.0003                # 0.03% per side
SHORT = False               # True = also go short on red

# Edit this list if Nifty 50 constituents change
STOCKS = """ADANIENT ADANIPORTS APOLLOHOSP ASIANPAINT AXISBANK BAJAJ-AUTO BAJFINANCE
BAJAJFINSV BEL BHARTIARTL CIPLA COALINDIA DRREDDY EICHERMOT ETERNAL GRASIM HCLTECH
HDFCBANK HDFCLIFE HINDALCO HINDUNILVR ICICIBANK INDUSINDBK INFY ITC JSWSTEEL JIOFIN
KOTAKBANK LT M&M MARUTI NESTLEIND NTPC ONGC POWERGRID RELIANCE SBILIFE SBIN
SHRIRAMFIN SUNPHARMA TATACONSUMER TATAMOTORS TATASTEEL TCS TECHM TITAN TRENT
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
    pos = pd.Series(d).shift(1).fillna(0).values   # trade on next bar
    if not SHORT:
        pos = np.where(pos > 0, 1, 0)
    ret = df.Close.pct_change().fillna(0).values
    # use open-to-close style approximation on close returns
    strat = pos * ret
    trades = np.abs(np.diff(pos, prepend=0))
    strat = strat - trades * FEE
    eq = (1 + pd.Series(strat, index=df.index)).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    # trade-level stats
    tid = (trades > 0).cumsum()
    tr_ret = pd.Series(strat).groupby(tid).apply(lambda x: (1 + x).prod() - 1)
    in_trade = pd.Series(pos != 0).groupby(tid).first()
    tr_ret = tr_ret[in_trade.reindex(tr_ret.index).values]
    return dict(Trades=len(tr_ret),
                WinRate=round((tr_ret > 0).mean() * 100, 1) if len(tr_ret) else 0,
                StrategyPct=round((eq.iloc[-1] - 1) * 100, 1),
                BuyHoldPct=round((df.Close.iloc[-1] / df.Close.iloc[0] - 1) * 100, 1),
                MaxDD=round(dd * 100, 1))

rows = []
for s in STOCKS:
    try:
        df = yf.download(s + ".NS", period="730d", interval="1h",
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
        if len(df) < 100: raise ValueError("not enough data")
        rows.append({"Stock": s, **backtest(df)})
    except Exception as e:
        print("skip", s, e)

res = pd.DataFrame(rows).sort_values("StrategyPct", ascending=False)
print(res.to_string(index=False))
print("\nAverage strategy %:", round(res.StrategyPct.mean(), 1),
      "| Average buy&hold %:", round(res.BuyHoldPct.mean(), 1),
      "| Profitable stocks:", (res.StrategyPct > 0).sum(), "/", len(res))
res.to_csv("nifty50_supertrend_results.csv", index=False)
