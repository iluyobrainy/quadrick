# Quadrick Phase 1 Paper Assessment

## Run Window
- Start (UTC): 2026-02-16T11:38:25.320771+00:00
- End (UTC): 2026-02-16T17:39:57.676853+00:00
- Target Duration (min): 360
- Actual Duration (min): 361.539
- Snapshot Poll (sec): 15

## Runtime Reliability
- Initialized: True
- Fatal Error: None
- Runtime Error Count: 0
- Forced Task Cancelled: False
- Snapshots Collected: 942

## Trading Metrics
- Closed Trades: 37
- Win Rate: 29.73%
- Expectancy: -0.57034% per trade
- Profit Factor: 0.611815
- Max Drawdown on $100 curve: 20.60208%

## $100 Projection (Observed Sample)
- End Equity: $79.39792
- Net PnL: $-20.60208
- Total Return: -20.60208%
- Trades/Hour: 6.140412

## Monte Carlo Projection (Same Trade Distribution)
- 7D trade count estimate: 1031
- 7D P10/P50/P90 on $100: $0.041288 / $0.159449 / $0.635202
- 30D trade count estimate: 4421
- 30D P10/P50/P90 on $100: $0.0 / $0.0 / $0.0

## Grades
- Engineering Grade: 100.0/100
- Quant Grade: 20.0/100

## Engineering Notes
- No major runtime reliability issues observed.

## Quant Notes
- Non-positive expectancy observed.
- Profit factor below 1.0 indicates weak edge.
- High drawdown profile on normalized $100 curve.

## Runtime Notes
- None

## Runtime Errors
- None

## Caveats
- This is testnet paper execution and does not include full live slippage/latency/psychology effects.
- Small sample windows can strongly overfit recent market regime.
- Projection assumes future trade distribution resembles observed sample.