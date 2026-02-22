# Quadrick Phase 1 Paper Assessment

## Run Window
- Start (UTC): 2026-02-16T18:39:02.037854+00:00
- End (UTC): 2026-02-17T00:40:04.400110+00:00
- Target Duration (min): 360
- Actual Duration (min): 361.039
- Snapshot Poll (sec): 15

## Runtime Reliability
- Initialized: True
- Fatal Error: None
- Runtime Error Count: 0
- Forced Task Cancelled: False
- Snapshots Collected: 911

## Trading Metrics
- Closed Trades: 10
- Win Rate: 50.0%
- Expectancy: 0.146437% per trade
- Profit Factor: 1.090977
- Max Drawdown on $100 curve: 12.371082%

## $100 Projection (Observed Sample)
- End Equity: $100.588299
- Net PnL: $0.588299
- Total Return: 0.588299%
- Trades/Hour: 1.661869

## Monte Carlo Projection (Same Trade Distribution)
- 7D trade count estimate: 279
- 7D P10/P50/P90 on $100: $45.625986 / $115.338937 / $280.296209
- 30D trade count estimate: 1196
- 30D P10/P50/P90 on $100: $32.278948 / $187.843984 / $1257.306099

## Grades
- Engineering Grade: 100.0/100
- Quant Grade: 32.5/100

## Engineering Notes
- No major runtime reliability issues observed.

## Quant Notes
- Grade capped at 75 due to modest sample (<20 closed trades).

## Runtime Notes
- None

## Runtime Errors
- None

## Caveats
- This is testnet paper execution and does not include full live slippage/latency/psychology effects.
- Small sample windows can strongly overfit recent market regime.
- Projection assumes future trade distribution resembles observed sample.