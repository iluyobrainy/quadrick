# Quadrick 6H Run - Assessment

As of: 2026-02-16 18:40 (local machine time)
Run start: 2026-02-16 12:38:25
Run end: 2026-02-16 18:39:57 (UTC from assessment report)
Mode: Bybit testnet, `ALLOW_LIVE_TRADING=false`, sizing normalized to `$100` logic

## Executive Summary
- The system is operational, but the current strategy profile is **not profitable** on this sample.
- Losses are primarily from low win rate + repeated execution friction (frequent inline TP/SL rejects and fallback flow).
- The bot can trade, but it is not yet safe to scale live capital with current behavior.

## Observed Performance (Final 6H Report)
- Closed trades: `37`
- Wins / Losses: `11 / 26`
- Win rate: `29.73%`
- Expectancy: `-0.57034%` per trade
- Profit factor: `0.61`
- Normalized `$100` ending balance: `$79.39792`
- Normalized return: `-20.60208%`
- Max drawdown on normalized `$100`: `20.60208%`

Reference files:
- `reports/phase1/run_6h_20260216_123817/latest.md`
- `reports/phase1/run_6h_20260216_123817/latest.json`

## Reliability / Execution Observations
- Inline TP/SL reject events: `29`
- Order placement failures (handled/retried): `43`
- Order placement successes: `47`
- Post-entry protection applied after fallback entry: `13`
- Post-entry protection hard failure events: `1`
- Cooldown triggers: `2`
- Cooldown-blocked cycles: `203`
- BTC affordability skips worked as intended: `7`

## Key Findings (Priority Order)
1. **Edge is negative on this sample.**
   - Evidence: low win rate, negative expectancy, PF < 1.
2. **Execution flow still has heavy friction.**
   - Frequent exchange rejects around TP/SL base-price checks degrade fill quality and timing.
   - Relevant paths: `main.py` fallback order flow around `_open_position`, `src/exchange/bybit_client.py` `update_position_protection`.
3. **Position adds on same symbol increase risk concentration.**
   - There is no strict guard to block re-entry/scale-in on an already-open symbol unless explicitly intended.
   - Relevant paths: `main.py` `_open_position`; risk only checks concurrent count in `src/risk/risk_manager.py`.
4. **Cooldown observability is confusing under local-vs-UTC display.**
   - Cooldown state uses naive `datetime.utcnow()` persistence and prints naive timestamps.
   - Relevant path: `src/controls/emergency_controls.py`.

## What Is Working
- Live/testnet safety gate for mainnet is present.
- BTC min-order affordability filter prevents repeated unaffordable attempts.
- Fallback from inline TP/SL to post-entry protection exists and usually recovers.
- Bot loop remains stable under network/API noise (keeps running, retries, and logs).

## Why Losses Are Happening
- Entry quality and stop placement are not producing positive expectancy in this regime.
- Too many trades still get opened with fragile protection assumptions (later corrected/retried).
- Small-account logic still permits effective risk concentration through repeated same-symbol exposure.

## Improvement Plan (To Reduce Losses, Not Eliminate Them)
Note: eliminating losses completely is not realistic in live markets; the goal is tighter drawdown control and positive expectancy.

### P0 (Immediate)
1. Enforce **single-position-per-symbol** by default.
   - Block new `OPEN_POSITION` if same symbol already open, unless explicit `scale_in=true`.
2. Harden pre-order TP/SL validation against latest mark/base price.
   - Recompute side-valid TP/SL immediately before order submit.
3. Reduce small-account aggression.
   - For `$100` mode: cap risk to `2-4%` and leverage to `<=5x`.
4. Raise minimum quality threshold.
   - Increase minimum R:R floor to `>=1.8` in small-account mode.

### P1 (Next)
1. Add regime filter to reduce churn.
   - Skip entries when LTF/HTF disagree or market is chop unless score is exceptional.
2. Add execution slippage guard.
   - Reject entry if mark moved beyond threshold between decision and order.
3. Replace naive cooldown timestamps with timezone-aware UTC objects and explicit local rendering.

### P2 (Validation)
1. Re-run controlled test with isolated conditions:
   - fixed symbols, fixed hours, no external account contamination.
2. Require pass gates before live consideration:
   - sample >= 100 closed trades
   - expectancy > 0
   - profit factor >= 1.2
   - max drawdown on normalized `$100` <= 10%

## Grade Notes
- Script-generated grades:
  - Engineering: `100/100` (runtime stability-focused, does not heavily penalize execution friction/churn)
  - Quant: `20/100`
- Practical assessment for live readiness:
  - Engineering: `~62/100`
  - Quant: `20/100`

## Recommendation
- **No-go for live scaling** in current form.
- Implement P0 fully, then run a fresh controlled paper phase and re-grade.
