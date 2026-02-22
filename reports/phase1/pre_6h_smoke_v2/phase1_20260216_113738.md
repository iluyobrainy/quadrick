# Quadrick Phase 1 Paper Assessment

## Run Window
- Start (UTC): 2026-02-16T11:26:39.485614+00:00
- End (UTC): 2026-02-16T11:37:38.126269+00:00
- Target Duration (min): 10
- Actual Duration (min): 10.977
- Snapshot Poll (sec): 15

## Runtime Reliability
- Initialized: True
- Fatal Error: None
- Runtime Error Count: 0
- Forced Task Cancelled: False
- Snapshots Collected: 21

## Trading Metrics
- Closed Trades: 1
- Win Rate: 0.0%
- Expectancy: -2.55732% per trade
- Profit Factor: 0.0
- Max Drawdown on $100 curve: 2.55732%

## $100 Projection (Observed Sample)
- End Equity: $97.44268
- Net PnL: $-2.55732
- Total Return: -2.55732%
- Trades/Hour: 5.465803

## Monte Carlo Projection (Same Trade Distribution)
- 7D trade count estimate: 918
- 7D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0
- 30D trade count estimate: 3935
- 30D P10/P50/P90 on $100: $100.0 / $100.0 / $100.0

## Grades
- Engineering Grade: 100.0/100
- Quant Grade: 17.4/100

## Engineering Notes
- No major runtime reliability issues observed.

## Quant Notes
- Grade capped at 60 due to small sample (<10 closed trades).
- Non-positive expectancy observed.
- Profit factor below 1.0 indicates weak edge.

## Runtime Notes
- Forced cleanup closed 1 open position(s).

## Runtime Errors
- None

## Caveats
- This is testnet paper execution and does not include full live slippage/latency/psychology effects.
- Small sample windows can strongly overfit recent market regime.
- Projection assumes future trade distribution resembles observed sample.