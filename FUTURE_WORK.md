# Future Work & Known Issues

This document tracks potential improvements and known issues for the Quadrick trading bot's hybrid LLM+Algo system.

---

## 🔧 Future Improvements

### 1. Dynamic Symbol Config (Auto-Learn Volatility)
**Impact:** High | **Complexity:** Medium

Instead of static volatility classifications, learn from historical trades:

```python
# Proposed implementation
class DynamicSymbolConfig:
    async def calculate_volatility_class(self, symbol: str, days: int = 14) -> str:
        """Auto-classify based on historical ATR"""
        trades = await self.db.get_trades_by_symbol(symbol, days)
        avg_atr_pct = sum(t["atr_pct"] for t in trades) / len(trades)
        
        if avg_atr_pct > 1.5:
            return "high"
        elif avg_atr_pct < 0.8:
            return "low"
        return "medium"
    
    async def update_symbol_config(self, symbol: str):
        """Update config based on recent performance"""
        vol_class = await self.calculate_volatility_class(symbol)
        win_rate = await self.db.get_symbol_win_rate(symbol)
        
        # Adjust trail_mult based on win rate
        # If low win rate, widen stops
        if win_rate < 40:
            self.configs[symbol].trail_mult *= 1.1
```

**Files to modify:** `src/config/symbol_config.py`, `main.py`

---

### 2. Backtest Report on Startup
**Impact:** Medium | **Complexity:** Low

Auto-generate performance report when bot starts:

```python
# Add to main.py __init__ or startup
async def _show_startup_report(self):
    """Show performance summary on startup"""
    from src.analytics.backtest_analyzer import BacktestAnalyzer
    
    analyzer = BacktestAnalyzer(self.db)
    
    # Direction stats
    dir_stats = await analyzer.get_direction_stats(days=7)
    if dir_stats:
        logger.info(f"📊 7-Day Stats: Long {dir_stats.long_win_rate:.1f}% WR, Short {dir_stats.short_win_rate:.1f}% WR")
    
    # Confidence calibration
    calibration = await analyzer.get_confidence_calibration(days=30)
    for cal in calibration:
        if not cal.is_well_calibrated:
            logger.warning(f"⚠️ Calibration issue: {cal.confidence_bucket} predicted {cal.predicted_win_rate:.0f}%, actual {cal.actual_win_rate:.0f}%")
```

**Files to modify:** `main.py` (add call in `start()`)

---

### 3. Funding Rate Cache
**Impact:** Low | **Complexity:** Low

Avoid repeated API calls by caching funding rates:

```python
# Add to FundingAnalyzer
class FundingAnalyzer:
    def __init__(self, ...):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_cached_rate(self, symbol: str) -> Optional[float]:
        if symbol in self._cache:
            rate, timestamp = self._cache[symbol]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return rate
        return None
    
    def set_cache(self, symbol: str, rate: float):
        self._cache[symbol] = (rate, datetime.now())
```

**Files to modify:** `src/analysis/funding_analyzer.py`

---

### 4. Multi-Timeframe OpportunityScorer
**Impact:** Medium | **Complexity:** Medium

Score on both 1H and 15M for better filtering:

```python
# Add to OpportunityScorer
def score_opportunity(self, symbol: str, analysis: Dict) -> ScoredOpportunity:
    tf_15m = analysis.get("timeframe_analysis", {}).get("15m", {})
    tf_1h = analysis.get("timeframe_analysis", {}).get("1h", {})
    
    # 15M scores (60% weight for scalping)
    score_15m = self._score_timeframe(tf_15m)
    
    # 1H scores (40% weight for trend alignment)
    score_1h = self._score_timeframe(tf_1h)
    
    # Weighted combination
    total_score = (score_15m * 0.6) + (score_1h * 0.4)
    
    # Bonus if both agree on direction
    if tf_15m.get("trend") == tf_1h.get("trend"):
        total_score += 10
```

**Files to modify:** `src/analysis/opportunity_scorer.py`

---

### 5. Market Regime Detection
**Impact:** High | **Complexity:** High

Adjust all parameters based on market conditions:

```python
class MarketRegimeDetector:
    """Detect market regime and adjust strategy"""
    
    REGIMES = ["trending", "ranging", "volatile", "quiet"]
    
    def detect_regime(self, btc_analysis: Dict) -> str:
        adx = btc_analysis.get("adx", 25)
        bb_width = btc_analysis.get("bb_width", 2)
        
        if adx > 30 and bb_width > 3:
            return "trending"
        elif adx < 15:
            return "ranging"
        elif bb_width > 4:
            return "volatile"
        else:
            return "quiet"
    
    def get_regime_adjustments(self, regime: str) -> Dict:
        adjustments = {
            "trending": {"trail_mult": 0.7, "counter_trend_allowed": False},
            "ranging": {"trail_mult": 1.2, "counter_trend_allowed": True},
            "volatile": {"trail_mult": 1.5, "max_risk": 10},
            "quiet": {"trail_mult": 0.8, "max_risk": 20},
        }
        return adjustments.get(regime, {})
```

**Files to create:** `src/analysis/regime_detector.py`

---

## ⚠️ Known Issues & Mitigations

### 1. OpportunityScorer Format Dependency
**Issue:** `get_top_opportunities` expects specific analysis dict format

**Mitigation:** Fallback to volume ranking if no opportunities scored
```python
if opportunities:
    ranked_watchlist = [opp.symbol for opp in opportunities]
else:
    ranked_watchlist = sorted(self.watchlist, key=volume_key)[:5]
```

**Status:** ✅ Already implemented

---

### 2. Symbol Config Variable Scope
**Issue:** `symbol_config` referenced in trailing stop log before definition

**Check:** Review line ~1357 in main.py
```python
logger.info(f"ATR-based trailing ({symbol_config.vol_class} vol): ...")
```
`symbol_config` should be defined at line ~1151 before use.

**Status:** ✅ Should be fine, but verify during runtime

---

### 3. Backtest Requires Trade Data
**Issue:** BacktestAnalyzer returns None/empty if no trades in Supabase

**Mitigation:** Methods handle gracefully:
```python
if not response.data:
    return None
```

**Action needed:** Run bot for a few days to collect data before using backtest

---

### 4. Funding Rate API Availability
**Issue:** `market_data["funding_rates"]` may be empty if API fails

**Mitigation:** Default to 0 funding rate (neutral):
```python
funding_rate = market_data.get("funding_rates", {}).get(decision.symbol, 0)
```

**Status:** ✅ Already handled

---

## 📊 Monitoring Checklist

When deploying, watch for these log patterns:

| Pattern | Meaning | Action if Excessive |
|---------|---------|---------------------|
| `📊 OpportunityScorer: Top X setups` | Pre-filtering working | None |
| `📊 No high-scoring opportunities` | Fallback active | Check if analysis data valid |
| `📊 Risk capped for` | Symbol limits active | Review if caps too aggressive |
| `⚠️ Crowded long/short` | Funding spike detected | Normal, position reduced |
| `🚫 Counter-trend BLOCKED` | Algo gate working | Normal, risky trade prevented |
| `ATR not available` | Data issue | Check market_analyzer output |

---

## 🗓️ Recommended Review Schedule

- **Daily:** Check win rate by direction (long vs short)
- **Weekly:** Review symbol performance, adjust configs if needed
- **Monthly:** Run confidence calibration, tune OpportunityScorer thresholds

---

*Last updated: 2026-02-08*
