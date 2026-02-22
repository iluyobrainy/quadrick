# Quadrick Phase 1 Paper Assessment

## Run Window
- Start (UTC): 2026-02-16T18:16:21.792487+00:00
- End (UTC): 2026-02-16T18:28:13.522860+00:00
- Target Duration (min): 10
- Actual Duration (min): 11.862
- Snapshot Poll (sec): 15

## Runtime Reliability
- Initialized: True
- Fatal Error: None
- Runtime Error Count: 0
- Forced Task Cancelled: False
- Snapshots Collected: 22

## Trading Metrics
- Closed Trades: 2
- Win Rate: 0.0%
- Expectancy: -4.122072% per trade
- Profit Factor: 0.0
- Max Drawdown on $100 curve: 8.074557%

## $100 Projection (Observed Sample)
- End Equity: $91.925443
- Net PnL: $-8.074557
- Total Return: -8.074557%
- Trades/Hour: 10.11619

## Monte Carlo Projection (Same Trade Distribution)
- 7D trade count estimate: 1699
- 7D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0
- 30D trade count estimate: 7283
- 30D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0

## Grades
- Engineering Grade: 100.0/100
- Quant Grade: 10.4/100

## Engineering Notes
- No major runtime reliability issues observed.

## Quant Notes
- Grade capped at 60 due to small sample (<10 closed trades).
- Non-positive expectancy observed.
- Profit factor below 1.0 indicates weak edge.

## Runtime Notes
- None

## Runtime Errors
- None

## Caveats
- This is testnet paper execution and does not include full live slippage/latency/psychology effects.
- Small sample windows can strongly overfit recent market regime.
- Projection assumes future trade distribution resembles observed sample.