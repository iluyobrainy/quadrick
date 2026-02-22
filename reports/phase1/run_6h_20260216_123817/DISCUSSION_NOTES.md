# Phase 1 Discussion Notes (6H Run)

Date: 2026-02-16  
Scope: Paper/Testnet run review and improvement plan before next implementation cycle.

## 1) What I Observed
- The bot runtime is stable (no fatal loop crashes), but trade outcomes are currently negative.
- The strategy has negative edge on this sample:
  - Closed trades: 37
  - Win rate: 29.73%
  - Expectancy: -0.57034% per trade
  - Profit factor: 0.611815
  - $100 normalized end balance: $79.39792 (-20.60208%)
- Execution friction is still material:
  - Frequent inline TP/SL rejects and fallback protection flow usage
  - Order retries/failures occurred often even though loop survived
- Risk concentration risk remains:
  - Re-entry/stacking on the same symbol is still too permissive for a $100 account mode.

## 2) Key Conclusion
- System is operational but not yet live-scale ready.
- Objective should be loss reduction and positive expectancy, not zero-loss trading (zero loss is not realistic in live markets).

## 3) Why Losses Persist
- Entry quality filter is not strict enough for noisy/choppy periods.
- TP/SL placement and validation are still fragile close to order send time.
- Position management allows behavior that can over-concentrate risk in one symbol.
- Small-account mode is still too aggressive for current hit rate profile.

## 4) Improvement Plan

### P0 (Implement first)
1. Single-position-per-symbol default guard (no add unless explicit `scale_in=true`).
2. Re-validate TP/SL at submit time with latest mark/base price to avoid side-invalid rejects.
3. Reduce risk in $100 mode:
   - max risk per trade 2-4%
   - leverage cap <= 5x
4. Raise minimum quality floor:
   - enforce minimum R:R >= 1.8

### P1 (Next hardening)
1. Add regime filter: avoid low-quality chop when HTF/LTF disagree.
2. Add slippage/latency guard: cancel if price moved too far from planned entry.
3. Make cooldown timestamps timezone-aware UTC and clearer in logs.

### P2 (Validation gates)
1. Run fresh controlled paper test after P0/P1:
   - fixed symbol set, fixed window, no external manual contamination
2. Pass criteria before live consideration:
   - >= 100 closed trades
   - expectancy > 0
   - profit factor >= 1.2
   - normalized $100 max drawdown <= 10%

## 5) Current Grades (Practical)
- Engineering readiness: ~62/100
- Quant readiness: 20/100
- Live scaling decision: NO-GO until P0/P1 are complete and validation gates pass.

## 6) Next Discussion Focus
- Confirm exact P0 parameters (risk %, leverage cap, symbol cap, R:R floor).
- Confirm whether scale-in is disabled globally for now.
- Confirm the fixed symbol basket for the next controlled run.
