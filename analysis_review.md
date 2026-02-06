# Codebase Review & Improvement Plan (Bybit + DeepSeek Futures Bot)

This review focuses on trade decision flow, risk management, and SL/TP mechanics, based on current implementation in `main.py`, `src/llm/deepseek_client.py`, `src/risk/risk_manager.py`, `src/execution/smart_execution.py`, and `src/analysis/market_analyzer.py`.

## Key Findings

### 1) Forced-trade mode always on (overtrading risk)
* `DeepSeekClient.max_consecutive_waits` is set to `0`, which makes `force_trade` true every cycle. This means the LLM is **never allowed to WAIT**, even if market conditions are poor, and will be escalated until it outputs a trade. This is a strong driver of low-quality entries and drawdowns in choppy or low-liquidity regimes. (`src/llm/deepseek_client.py`).
* **Status: Fixed** — the client now allows multiple WAITs before escalation.

### 2) Conflicting risk guidance in prompt vs enforcement
* System prompt says “< $20 risk 6–10%”, but later says “< $20 use 15–25% risk” and other aggressive rules. This inconsistency likely confuses the model. (`src/llm/deepseek_client.py`).
* The `RiskManager` defaults to **min 10% / max 30% risk** and allows up to 15% risk for < $20 accounts. That is aggressive for scalping and makes SL/TP mistakes far more punishing. (`src/risk/risk_manager.py`).
* **Status: Fixed** — prompt and LLM context now enforce a 10–30% risk band in `system_state`.

### 3) SL/TP validation is incomplete and can be directionally wrong
* When opening a position, the code only enforces a minimum **buffer on stop-loss** (0.5%), but does not validate TP or enforce risk/reward ratios. This can let the LLM submit a TP on the wrong side of entry or a very low R:R. (`main.py`).
* In modify-position flow, there is direction checking for TP vs entry, but **not in open-position** flow. (`main.py`).
* **Status: Fixed** — open-position flow now applies direction-aware SL/TP guardrails and enforces minimum R:R using ATR fallbacks.

### 4) Partial profit logic is long-only
* Partial take profit checks only `current_price >= target`. For shorts, the target price is below entry, so this condition never triggers. That means **partial profits won’t execute for shorts**. (`src/execution/smart_execution.py`).
* **Status: Fixed** — partial-profit checks now honor trade direction for long/short.

### 5) Fixed trailing stop distance (0.5%) is too rigid
* Trailing stop uses a fixed 0.5% distance for all symbols and volatility regimes. This is likely too tight in high volatility and too loose in low volatility. It causes premature stops or fails to protect profits. (`main.py`, `src/execution/smart_execution.py`).
* **Status: Fixed** — trailing distance is now ATR-based and adapts to volatility.

### 6) Market/strategy data is rich, but the LLM is doing too much
* The LLM is asked to create the strategy, decide direction, size, and SL/TP with a huge prompt. There is little deterministic “guardrail” logic to enforce SL/TP quality, R:R, or volatility-adjusted sizing. (`src/llm/deepseek_client.py`, `main.py`).
* **Status: In progress** — SL/TP guardrails are now deterministic, but strategy selection and sizing still rely on the LLM.

## Priority Improvements

### A) Allow WAITs and reduce forced trades
* **Raise `max_consecutive_waits`** to at least 2–4 and only force trade when the portfolio is under-utilized and the model has a strong signal. This should reduce low-quality trades. (`src/llm/deepseek_client.py`).
* **Status: Implemented** — set to allow multiple WAITs before escalation.

### B) Harmonize risk rules and make them data-driven
* Align prompt guidance with `RiskManager` rules (e.g., <= 3–6% for small accounts, 1–2% for higher balances) and **make risk contingent on volatility and signal strength**. (`src/llm/deepseek_client.py`, `src/risk/risk_manager.py`).
* Implement a risk “band” per symbol (e.g., 0.5–1.5% per trade) and override LLM risk when it’s inconsistent with that band. (`src/risk/risk_manager.py`).
* **Status: Partially implemented** — prompt + system_state now use a single 10–30% band; volatility-based risk scaling is still pending.

### C) Enforce SL/TP directional correctness + minimum R:R in code
* Add deterministic validation for:
  - Long: `SL < entry < TP`
  - Short: `TP < entry < SL`
  - Minimum R:R (e.g., >= 1.3 or 1.5)
* If invalid, auto-correct using ATR or key-levels from analysis. (`main.py`, `src/analysis/market_analyzer.py`).
* **Status: Implemented** — open-position flow now applies ATR-based guardrails and minimum R:R enforcement.

### D) Fix partial take profit for shorts
* Update partial profit trigger to check direction (e.g., for short: `current_price <= target`). (`src/execution/smart_execution.py`).
* **Status: Implemented** — partial-profit triggers now respect trade direction.

### E) Replace fixed trailing stop with ATR-based
* Compute trailing distance using ATR or BB width per symbol/timeframe. This should be part of `risk_management` in the decision payload, with defaults from analysis. (`main.py`, `src/analysis/market_analyzer.py`, `src/execution/smart_execution.py`).
* **Status: Implemented** — trailing stops now use ATR-based sizing.

### F) Split the decision flow into deterministic + LLM
* Use LLM primarily for **direction and setup selection**, while SL/TP, risk sizing, and validation are computed deterministically from market analysis and risk policy. This reduces LLM-induced error and bias. (`main.py`, `src/llm/deepseek_client.py`).
* **Status: In progress** — SL/TP guardrails and scalping filters are deterministic; next step is moving sizing and entry selection into deterministic logic.

## Suggested Next Discussion Topics

1. **Desired trading profile**: pure scalping vs hybrid (scalp + short swing). This will drive SL/TP distance and trailing behavior.
2. **Risk tolerance**: target max daily drawdown and per-trade risk. This should be reflected in `RiskManager` and prompt.
3. **Signal sources**: which indicators you trust most (RSI/MACD/BB/EMA/volume). We can weigh them and enforce confluence requirements in code.
4. **Direction bias**: we can reshape the prompt and guardrails to always compare long vs short edges and choose the higher expected value.

## Proposed Next Steps (implementation order)

1. Deterministic SL/TP validation + ATR sizing. ✅
2. Regime-driven scalping filters. ✅
3. Orderbook/microstructure signals. ✅
4. Better memory embeddings + retrieval integration. ✅

**Next Upgrade Targets**
1. Deterministic entry selection (reduce LLM dependence on symbol choice). ✅
2. Volatility-aware risk scaling within the 10–30% band. ✅
3. Backtest + walk-forward validation harness. ✅

---

If you'd like, I can proceed in this order and deliver each upgrade as a discrete, testable change set.
