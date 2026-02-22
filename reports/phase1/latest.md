# Quadrick Phase 1 Paper Assessment

## Run Window
- Start (UTC): 2026-02-21T23:47:50.999385+00:00
- End (UTC): 2026-02-22T01:49:46.712653+00:00
- Target Duration (min): 120
- Actual Duration (min): 121.929
- Snapshot Poll (sec): 15

## Runtime Reliability
- Initialized: True
- Fatal Error: None
- Runtime Error Count: 0
- Forced Task Cancelled: False
- Snapshots Collected: 331

## Trading Metrics
- Closed Trades: 5
- Win Rate: 80.0%
- Expectancy: 2.125603% per trade
- Profit Factor: 3.700201
- Max Drawdown on $100 curve: 3.936008%

## $100 Projection (Observed Sample)
- End Equity: $110.773614
- Net PnL: $10.773614
- Total Return: 10.773614%
- Trades/Hour: 2.460457

## Actual Exchange-Equity Snapshot
- Start Equity: $67872.440496
- End Equity: $67995.393348
- Net PnL: $122.952852
- Return: 0.181153%

## Monte Carlo Projection (Same Trade Distribution)
- 7D trade count estimate: 413
- 7D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0
- 30D trade count estimate: 1771
- 30D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0

## Grades
- Engineering Grade: 100.0/100
- Quant Grade: 60.0/100

## Engineering Notes
- No major runtime reliability issues observed.

## Quant Notes
- Grade capped at 60 due to small sample (<10 closed trades).

## Bucket Attribution (Symbol+Side+Regime)
- Tracked buckets: 13
- Top+: 1000PEPEUSDT Sell volatile (exp=1.559764%, n=3.6738)
- Top+: DOTUSDT Sell volatile (exp=3.850932%, n=0.9673)
- Top+: LINKUSDT Buy volatile (exp=0.434175%, n=2.278)
- Top-: ADAUSDT Sell volatile (exp=-2.56748%, n=3.4119)
- Top-: LINKUSDT Sell volatile (exp=-1.077244%, n=4.7632)
- Top-: DOTUSDT Buy trend (exp=-2.76321%, n=1.7576)
- Tail: 1000PEPEUSDT Sell volatile (<=-5%=0.137537, <=-7%=0.137537)
- Tail: ADAUSDT Sell volatile (<=-5%=0.0, <=-7%=0.0)
- Tail: DOTUSDT Buy trend (<=-5%=0.0, <=-7%=0.0)

## Runtime Notes
- Forced cleanup closed 1 open position(s).

## Runtime Errors
- None

## Caveats
- This is testnet paper execution and does not include full live slippage/latency/psychology effects.
- Small sample windows can strongly overfit recent market regime.
- Projection assumes future trade distribution resembles observed sample.