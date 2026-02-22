# Post-Run Fix Backlog

## Confirmed Issues To Fix Next

1. Margin cap float precision edge case blocks valid orders
- Symptom: order rejected with message like `Position margin exceeds 35% of account ($35.00 > $35.00)`.
- Likely cause: strict float `>` comparison without epsilon tolerance.
- Fix: apply small epsilon in margin-cap comparisons (`risk_manager`) and log raw values when near threshold.
- Expected impact: prevent false negatives on valid entries at exact cap boundary.

## What Worked In This Run (2026-02-17 2h sample)

1. Positive expectancy came from asymmetric winners, not high hit rate
- Closed trades: `20`
- Wins/Losses/Flats: `4 / 5 / 11`
- Expectancy: `+0.5486%` per trade
- Avg win vs avg loss: `+4.10%` vs `-1.08%`

2. Profitable cluster was mostly trend-aligned ADA/DOT
- `ADAUSDT`: 6 trades, net `+3.24%` (3 wins, 3 losses)
- `DOTUSDT`: 3 trades, net `+7.73%` (1 large win, 2 micro-losses)

3. Small-account risk caps prevented blowups
- Drawdown remained controlled (`~4.06%` on normalized `$100` curve).

## High-Impact Upgrades To Implement Next

1. Add epsilon margin tolerance in risk checks
- Allow exact-threshold trades by comparing with small tolerance (e.g., `1e-6`).

2. Treat flat outcomes as neutral in symbol quality scoring
- Do not count repeated `0.00%` as wins or losses for edge estimation.
- Add symbol-level flat streak cooldown with short duration.

3. Add anti-churn diversification guardrails
- Max consecutive entries per symbol.
- Repetition penalty in EV ranking with time decay and capped penalty.
- Keep override path when EV/confidence is materially stronger.

4. Tighten counter-trend soft override
- Keep override only for stronger confluence conditions.
- Reduce frequency of low-yield contrarian churn (mainly OPUSDT shorts).

5. Improve order placement resilience for Bybit 30208 rejects
- Reprice/retry once with fresh ticker before final reject.
- Keep post-reprice RR validation.

6. Preserve existing protections that are currently helping
- Keep small-account leverage/risk caps.
- Keep R:R auto-adjust and affordability checks.

## Implemented In Code (2026-02-18)

1. Margin-cap boundary tolerance applied end-to-end
- Added affordability epsilon handling in `main.py` so exact-cap cases are no longer rejected on float noise.

2. Flat outcomes treated as neutral with controlled cooldown
- Flat trades now update dedicated symbol flat streak state.
- Flat streak contributes a small EV penalty instead of being treated as directional win/loss signal.
- Short flat-trade cooldown remains configurable.

3. Anti-churn diversification and repetition control enabled
- EV selection now uses adjusted EV:
  - repetition penalty
  - flat-streak penalty
  - reject-streak penalty
- Added max consecutive symbol cap with EV-gap override logic for strong exceptions.

4. Counter-trend soft overrides tightened
- Raised score/confidence requirements and tightened ADX requirements.
- Added forecast-alignment requirement before allowing soft counter-trend overrides.

5. Bybit 30208 resilience improved
- Market orders now use IOC default and optional percent slippage tolerance.
- Reprice retries now run up to configurable attempts.
- On repeated reject, symbol reject streak + cooldown are applied to reduce churn.

6. Forecast intelligence integrated into EV scoring
- Added deterministic multi-horizon forecast engine (`5m/15m/30m`).
- Forecast confidence and directional probability now blend into expected edge when confidence threshold is met.

7. Quant primary warm-start enabled
- Added deterministic heuristic priors for `5m/15m/30m` horizons and blended them with model output by sample strength.
- This prevents cold-start `0.50` dead-zones where no proposals are generated.

8. Quant gating made sample-aware
- Uncertainty/confidence gates now relax modestly during low-sample warmup and auto-tighten as labels accumulate.
- Gate thresholds are logged into proposal metadata for auditability.

9. $100 affordability filter in quant selector
- Quant now skips non-executable symbols before final selection (e.g., ETH min-margin above cap under `$100` mode).
- Prevents repeated dead-loop picks on high-notional symbols.

10. Volatility-adaptive entry drift cap
- Entry drift ceiling now scales with ATR volatility and allows controlled high-volatility entries.
- Keeps stale-price protection while avoiding unnecessary skips during fast but tradable moves.

11. Quant observability endpoint path
- Added internal/public quant monitor endpoints for cycle metrics (`candidates`, `proposals`, `drift`, `reject-rate`, `latency`).
- Main bot now pushes quant monitor payloads through dashboard bridge.

## Validation Criteria For Next Run

1. Maintain or improve expectancy with lower flat-trade share.
2. Keep max drawdown at or below current level.
3. Reduce OPUSDT churn concentration without starving total trade count.
4. Reduce execution rejects (`30208` and margin boundary false rejects).
