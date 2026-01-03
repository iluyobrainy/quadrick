"""
Market Analysis Module - Technical indicators and market regime detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
# import talib  # Commented out due to numpy compatibility issue - using pandas_ta instead
import pandas_ta as ta
from loguru import logger
from dataclasses import dataclass
from enum import Enum


class MarketRegime(str, Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE_EXPANSION = "volatile_expansion"
    LOW_VOLATILITY = "low_volatility"


class TrendDirection(str, Enum):
    """Trend direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TechnicalIndicators:
    """Technical indicators for a specific timeframe"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    volume_sma: float
    volume_ratio: float
    ema_9: float
    ema_21: float
    ema_50: float
    ema_200: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    
    # Advanced indicators
    vwap: Optional[float] = None
    pivot_point: Optional[float] = None
    pivot_r1: Optional[float] = None
    pivot_r2: Optional[float] = None
    pivot_s1: Optional[float] = None
    pivot_s2: Optional[float] = None
    fib_236: Optional[float] = None
    fib_382: Optional[float] = None
    fib_500: Optional[float] = None
    fib_618: Optional[float] = None
    fib_786: Optional[float] = None
    rsi_divergence: Optional[str] = None  # "bullish", "bearish", "none"
    macd_divergence: Optional[str] = None

    def to_dict(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Convert to flat dictionary for compatibility with AI and Dashboard"""
        data = {
            "rsi": round(float(self.rsi), 2),
            "macd": round(float(self.macd), 6),
            "macd_signal": round(float(self.macd_signal), 6),
            "macd_histogram": round(float(self.macd_histogram), 6),
            "atr": round(float(self.atr), 6),
            "bb_upper": round(float(self.bb_upper), 2),
            "bb_middle": round(float(self.bb_middle), 2),
            "bb_lower": round(float(self.bb_lower), 2),
            "bb_width": round(float(self.bb_width), 6),
            "volume_sma": float(self.volume_sma),
            "volume_ratio": round(float(self.volume_ratio), 2),
            "ema_9": round(float(self.ema_9), 2),
            "ema_21": round(float(self.ema_21), 2),
            "ema_50": round(float(self.ema_50), 2),
            "ema_200": round(float(self.ema_200), 2) if self.ema_200 else None,
            "adx": round(float(self.adx), 2) if self.adx else None,
        }
        
        # Categorical MACD
        data["macd_categorical"] = "bullish" if self.macd_histogram > 0 else "bearish"
        
        # Categorical BB Position
        if current_price:
            if current_price >= self.bb_upper:
                data["bb_position"] = "upper"
            elif current_price <= self.bb_lower:
                data["bb_position"] = "lower"
            else:
                data["bb_position"] = "middle"
        
        return data


@dataclass
class KeyLevels:
    """Key support and resistance levels"""
    immediate_resistance: float
    major_resistance: float
    immediate_support: float
    major_support: float
    pivot_point: float
    r1: float
    r2: float
    s1: float
    s2: float


@dataclass
class MarketStructure:
    """Market structure analysis"""
    trend: TrendDirection
    trend_strength: float  # 0-100
    market_regime: MarketRegime
    volatility_percentile: float  # 0-100
    momentum: float  # -100 to 100
    volume_trend: str  # "increasing", "decreasing", "stable"
    breakout_potential: float  # 0-100
    reversal_probability: float  # 0-100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "trend": self.trend.value if hasattr(self.trend, 'value') else str(self.trend),
            "trend_strength": round(float(self.trend_strength), 2),
            "market_regime": self.market_regime.value if hasattr(self.market_regime, 'value') else str(self.market_regime),
            "volatility_percentile": round(float(self.volatility_percentile), 2),
            "momentum": round(float(self.momentum), 2),
            "volume_trend": self.volume_trend,
            "breakout_potential": round(float(self.breakout_potential), 2),
            "reversal_probability": round(float(self.reversal_probability), 2),
        }


@dataclass
class TimeframeAnalysis:
    """Complete analysis for a single timeframe"""
    timeframe: str
    candles: pd.DataFrame
    indicators: TechnicalIndicators
    key_levels: KeyLevels
    structure: MarketStructure
    patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Flatten and convert to dictionary for strategy/AI compatibility"""
        current_price = self.candles['close'].iloc[-1] if not self.candles.empty else None
        data = self.indicators.to_dict(current_price=current_price)
        data.update(self.structure.to_dict())
        
        # Add timeframe name
        data["timeframe"] = self.timeframe
        
        # Include key levels if available
        if hasattr(self, 'key_levels') and self.key_levels:
            data["key_levels"] = {
                "immediate_support": self.key_levels.immediate_support,
                "immediate_resistance": self.key_levels.immediate_resistance,
                "major_support": self.key_levels.major_support,
                "major_resistance": self.key_levels.major_resistance,
            }
            
        data["timeframe"] = self.timeframe
        data["patterns"] = self.patterns
        return data


class MarketAnalyzer:
    """Market analysis engine"""
    
    def __init__(self):
        """Initialize market analyzer"""
        self.lookback_periods = {
            "1m": 100,
            "5m": 100,
            "15m": 100,
            "1h": 100,
            "4h": 50,
            "1d": 30,
        }
        logger.info("Market analyzer initialized")
    
    def detect_volume_spike(self, df: pd.DataFrame, multiplier: float = 3.0) -> Dict[str, Any]:
        """Detect volume spikes for scalping entries"""
        try:
            if len(df) < 20:
                return {'is_spike': False, 'spike_ratio': 0, 'direction': 'neutral'}
            
            # Calculate rolling average volume
            df = df.copy()
            df['volume_avg'] = df['volume'].rolling(window=20, min_periods=10).mean()
            
            current_volume = float(df['volume'].iloc[-1])
            avg_volume = float(df['volume_avg'].iloc[-1])
            spike_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Determine direction
            current_close = float(df['close'].iloc[-1])
            current_open = float(df['open'].iloc[-1])
            direction = 'bullish' if current_close > current_open else 'bearish' if current_close < current_open else 'neutral'
            
            # Check for quality spike
            is_spike = spike_ratio >= multiplier
            if is_spike:
                price_change = abs((current_close - current_open) / current_open) * 100
                if price_change < 0.05:
                    is_spike = False
            
            # 3-candle momentum
            momentum = 0
            if len(df) >= 3:
                momentum = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
            
            return {
                'is_spike': is_spike,
                'spike_ratio': round(spike_ratio, 2),
                'direction': direction,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'quality': 'high' if is_spike and spike_ratio > 5 else 'medium' if is_spike else 'low',
                'momentum': round(momentum, 3)
            }
        except Exception as e:
            logger.error(f"Error detecting volume spike: {e}")
            return {'is_spike': False, 'spike_ratio': 0, 'direction': 'neutral'}
    
    def analyze_symbol(
        self,
        symbol: str,
        klines_data: Dict[str, List[Dict[str, Any]]],
        current_price: float,
    ) -> Dict[str, TimeframeAnalysis]:
        """
        Comprehensive analysis of a symbol across multiple timeframes
        
        Args:
            symbol: Trading symbol
            klines_data: Dict of timeframe -> klines data
            current_price: Current market price
        
        Returns:
            Dict of timeframe -> TimeframeAnalysis
        """
        analyses = {}
        
        for timeframe, klines in klines_data.items():
            if not klines or len(klines) < 20:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                continue
            
            # Convert to DataFrame
            df = self._klines_to_dataframe(klines)
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Identify key levels
            key_levels = self._identify_key_levels(df, current_price)
            
            # Analyze market structure
            structure = self._analyze_market_structure(df, indicators)
            
            # Detect patterns
            patterns = self._detect_patterns(df, indicators)
            
            analyses[timeframe] = TimeframeAnalysis(
                timeframe=timeframe,
                candles=df,
                indicators=indicators,
                key_levels=key_levels,
                structure=structure,
                patterns=patterns,
            )
        
        return analyses
    
    def _klines_to_dataframe(self, klines: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert klines data to pandas DataFrame"""
        df = pd.DataFrame(klines)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        
        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators using pandas_ta"""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        
        # Ensure we have enough data
        min_period = 50
        if len(close) < min_period:
            # Return default values if not enough data
            return TechnicalIndicators(
                rsi=50.0,
                macd=0.0,
                macd_signal=0.0,
                macd_histogram=0.0,
                atr=0.0,
                bb_upper=close[-1] * 1.02,
                bb_middle=close[-1],
                bb_lower=close[-1] * 0.98,
                bb_width=0.04,
                volume_sma=volume[-1] if len(volume) > 0 else 0,
                volume_ratio=1.0,
                ema_9=close[-1],
                ema_21=close[-1],
                ema_50=close[-1],
            )
        
        # Use pandas_ta for indicators
        df_copy = df.copy()
        
        # RSI
        df_copy['rsi'] = ta.rsi(df_copy['close'], length=14)
        rsi = df_copy['rsi'].values
        
        # MACD
        try:
            macd_result = ta.macd(df_copy['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                macd_cols = macd_result.columns.tolist()
                macd_col = [c for c in macd_cols if 'MACD_' in c and 'h' not in c and 's' not in c.lower()][0] if any('MACD_' in c for c in macd_cols) else None
                signal_col = [c for c in macd_cols if 'MACDs' in c or 'signal' in c.lower()][0] if any('MACDs' in c or 'signal' in c.lower() for c in macd_cols) else None
                hist_col = [c for c in macd_cols if 'MACDh' in c or 'histogram' in c.lower()][0] if any('MACDh' in c or 'histogram' in c.lower() for c in macd_cols) else None
                
                macd = macd_result[macd_col].values if macd_col else np.zeros_like(close)
                macd_signal = macd_result[signal_col].values if signal_col else np.zeros_like(close)
                macd_hist = macd_result[hist_col].values if hist_col else np.zeros_like(close)
            else:
                macd = np.zeros_like(close)
                macd_signal = np.zeros_like(close)
                macd_hist = np.zeros_like(close)
        except Exception as e:
            logger.debug(f"MACD calculation failed: {e}")
            macd = np.zeros_like(close)
            macd_signal = np.zeros_like(close)
            macd_hist = np.zeros_like(close)
        
        # ATR
        df_copy['atr'] = ta.atr(df_copy['high'], df_copy['low'], df_copy['close'], length=14)
        atr = df_copy['atr'].values
        
        # Bollinger Bands
        try:
            bb_result = ta.bbands(df_copy['close'], length=20, std=2)
            if bb_result is not None and not bb_result.empty:
                # Find the actual column names (they vary)
                bb_cols = bb_result.columns.tolist()
                bb_upper_col = [c for c in bb_cols if 'BBU' in c][0] if any('BBU' in c for c in bb_cols) else None
                bb_middle_col = [c for c in bb_cols if 'BBM' in c][0] if any('BBM' in c for c in bb_cols) else None
                bb_lower_col = [c for c in bb_cols if 'BBL' in c][0] if any('BBL' in c for c in bb_cols) else None
                
                bb_upper = bb_result[bb_upper_col].values if bb_upper_col else close * 1.02
                bb_middle = bb_result[bb_middle_col].values if bb_middle_col else close
                bb_lower = bb_result[bb_lower_col].values if bb_lower_col else close * 0.98
            else:
                bb_upper = close * 1.02
                bb_middle = close
                bb_lower = close * 0.98
        except Exception as e:
            logger.debug(f"Bollinger Bands calculation failed: {e}")
            bb_upper = close * 1.02
            bb_middle = close
            bb_lower = close * 0.98
        
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Volume SMA
        df_copy['volume_sma'] = ta.sma(df_copy['volume'], length=20)
        volume_sma = df_copy['volume_sma'].values
        volume_ratio = volume / volume_sma if len(volume_sma) > 0 else np.ones_like(volume)
        
        # EMAs
        df_copy['ema_9'] = ta.ema(df_copy['close'], length=9)
        df_copy['ema_21'] = ta.ema(df_copy['close'], length=21)
        df_copy['ema_50'] = ta.ema(df_copy['close'], length=50)
        ema_9 = df_copy['ema_9'].values
        ema_21 = df_copy['ema_21'].values
        ema_50 = df_copy['ema_50'].values
        
        if len(close) >= 200:
            df_copy['ema_200'] = ta.ema(df_copy['close'], length=200)
            ema_200 = df_copy['ema_200'].values
        else:
            ema_200 = None
        
        # Stochastic
        try:
            stoch_result = ta.stoch(df_copy['high'], df_copy['low'], df_copy['close'], k=14, d=3)
            stoch_k = stoch_result[f'STOCHk_14_3_3'].values if stoch_result is not None else np.full_like(close, 50.0)
            stoch_d = stoch_result[f'STOCHd_14_3_3'].values if stoch_result is not None else np.full_like(close, 50.0)
        except Exception as e:
            logger.debug(f"Stochastic calculation failed: {e}")
            stoch_k = np.full_like(close, 50.0)
            stoch_d = np.full_like(close, 50.0)
        
        # ADX and DI
        try:
            adx_result = ta.adx(df_copy['high'], df_copy['low'], df_copy['close'], length=14)
            adx = adx_result[f'ADX_14'].values if adx_result is not None else np.full_like(close, 25.0)
            plus_di = adx_result[f'DMP_14'].values if adx_result is not None else np.zeros_like(close)
            minus_di = adx_result[f'DMN_14'].values if adx_result is not None else np.zeros_like(close)
        except Exception as e:
            logger.debug(f"ADX calculation failed: {e}")
            adx = np.full_like(close, 25.0)
            plus_di = np.zeros_like(close)
            minus_di = np.zeros_like(close)
        
        # VWAP (Volume Weighted Average Price)
        try:
            df_copy['vwap'] = ta.vwap(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
            vwap = df_copy['vwap'].values[-1] if 'vwap' in df_copy.columns else None
        except:
            vwap = None
        
        # Pivot Points (using last day's high/low/close)
        if len(df_copy) >= 2:
            pivot_high = df_copy['high'].iloc[-2]
            pivot_low = df_copy['low'].iloc[-2]
            pivot_close = df_copy['close'].iloc[-2]
            
            pivot_point = (pivot_high + pivot_low + pivot_close) / 3
            pivot_r1 = 2 * pivot_point - pivot_low
            pivot_r2 = pivot_point + (pivot_high - pivot_low)
            pivot_s1 = 2 * pivot_point - pivot_high
            pivot_s2 = pivot_point - (pivot_high - pivot_low)
        else:
            pivot_point = pivot_r1 = pivot_r2 = pivot_s1 = pivot_s2 = None
        
        # Fibonacci Retracements (from recent swing high/low)
        if len(close) >= 20:
            recent_high = np.max(close[-20:])
            recent_low = np.min(close[-20:])
            fib_range = recent_high - recent_low
            
            fib_236 = recent_high - (fib_range * 0.236)
            fib_382 = recent_high - (fib_range * 0.382)
            fib_500 = recent_high - (fib_range * 0.500)
            fib_618 = recent_high - (fib_range * 0.618)
            fib_786 = recent_high - (fib_range * 0.786)
        else:
            fib_236 = fib_382 = fib_500 = fib_618 = fib_786 = None
        
        # Divergence Detection
        rsi_divergence = self._detect_divergence(close, rsi, lookback=10)
        macd_divergence = self._detect_divergence(close, macd, lookback=10)
        
        return TechnicalIndicators(
            rsi=float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0,
            macd=float(macd[-1]) if not np.isnan(macd[-1]) else 0.0,
            macd_signal=float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0,
            macd_histogram=float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0,
            atr=float(atr[-1]) if not np.isnan(atr[-1]) else 0.0,
            bb_upper=float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else close[-1] * 1.02,
            bb_middle=float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else close[-1],
            bb_lower=float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else close[-1] * 0.98,
            bb_width=float(bb_width[-1]) if not np.isnan(bb_width[-1]) else 0.04,
            volume_sma=float(volume_sma[-1]) if not np.isnan(volume_sma[-1]) else volume[-1],
            volume_ratio=float(volume_ratio[-1]) if not np.isnan(volume_ratio[-1]) else 1.0,
            ema_9=float(ema_9[-1]) if not np.isnan(ema_9[-1]) else close[-1],
            ema_21=float(ema_21[-1]) if not np.isnan(ema_21[-1]) else close[-1],
            ema_50=float(ema_50[-1]) if not np.isnan(ema_50[-1]) else close[-1],
            ema_200=float(ema_200[-1]) if ema_200 is not None and not np.isnan(ema_200[-1]) else None,
            stoch_k=float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else 50.0,
            stoch_d=float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else 50.0,
            adx=float(adx[-1]) if not np.isnan(adx[-1]) else 25.0,
            plus_di=float(plus_di[-1]) if not np.isnan(plus_di[-1]) else 0.0,
            minus_di=float(minus_di[-1]) if not np.isnan(minus_di[-1]) else 0.0,
            # Advanced indicators
            vwap=float(vwap) if vwap and not np.isnan(vwap) else None,
            pivot_point=float(pivot_point) if pivot_point else None,
            pivot_r1=float(pivot_r1) if pivot_r1 else None,
            pivot_r2=float(pivot_r2) if pivot_r2 else None,
            pivot_s1=float(pivot_s1) if pivot_s1 else None,
            pivot_s2=float(pivot_s2) if pivot_s2 else None,
            fib_236=float(fib_236) if fib_236 else None,
            fib_382=float(fib_382) if fib_382 else None,
            fib_500=float(fib_500) if fib_500 else None,
            fib_618=float(fib_618) if fib_618 else None,
            fib_786=float(fib_786) if fib_786 else None,
            rsi_divergence=rsi_divergence,
            macd_divergence=macd_divergence,
        )
    
    def _identify_key_levels(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> KeyLevels:
        """Identify key support and resistance levels"""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        # Calculate pivot points
        last_high = high[-1]
        last_low = low[-1]
        last_close = close[-1]
        
        pivot = (last_high + last_low + last_close) / 3
        r1 = 2 * pivot - last_low
        r2 = pivot + (last_high - last_low)
        s1 = 2 * pivot - last_high
        s2 = pivot - (last_high - last_low)
        
        # Find recent swing highs and lows
        window = min(20, len(df))
        recent_highs = high[-window:]
        recent_lows = low[-window:]
        
        # Identify resistance levels (above current price)
        resistances = sorted([h for h in recent_highs if h > current_price])
        immediate_resistance = resistances[0] if resistances else current_price * 1.01
        major_resistance = resistances[min(2, len(resistances)-1)] if len(resistances) > 1 else current_price * 1.02
        
        # Identify support levels (below current price)
        supports = sorted([l for l in recent_lows if l < current_price], reverse=True)
        immediate_support = supports[0] if supports else current_price * 0.99
        major_support = supports[min(2, len(supports)-1)] if len(supports) > 1 else current_price * 0.98
        
        return KeyLevels(
            immediate_resistance=immediate_resistance,
            major_resistance=major_resistance,
            immediate_support=immediate_support,
            major_support=major_support,
            pivot_point=pivot,
            r1=r1,
            r2=r2,
            s1=s1,
            s2=s2,
        )
    
    def _analyze_market_structure(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> MarketStructure:
        """Analyze market structure and regime"""
        close = df["close"].values
        volume = df["volume"].values
        
        # Determine trend - prioritize recent price action
        trend = TrendDirection.NEUTRAL
        
        # Price action trend (more immediate)
        if len(close) >= 10:
            recent_close = close[-1]
            older_close_5 = close[-5]
            older_close_10 = close[-10]
            
            # Strong bearish if both recent periods are down
            if recent_close < older_close_5 < older_close_10:
                trend = TrendDirection.BEARISH
            # Strong bullish if both recent periods are up  
            elif recent_close > older_close_5 > older_close_10:
                trend = TrendDirection.BULLISH
            # Use EMA crossover as secondary signal
            elif indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
                trend = TrendDirection.BULLISH
            elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
                trend = TrendDirection.BEARISH
            # Check immediate direction if EMAs are mixed
            elif recent_close < older_close_5:
                trend = TrendDirection.BEARISH
            elif recent_close > older_close_5:
                trend = TrendDirection.BULLISH
        else:
            # Fallback to EMA for insufficient data
            if indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
                trend = TrendDirection.BULLISH
            elif indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
                trend = TrendDirection.BEARISH
        
        # Trend strength (ADX-based)
        trend_strength = min(indicators.adx * 2, 100) if indicators.adx else 50.0
        
        # Market regime
        if indicators.bb_width > 0.06:  # High volatility
            if abs(indicators.macd_histogram) > abs(indicators.atr * 0.1):
                market_regime = MarketRegime.VOLATILE_EXPANSION
            else:
                market_regime = MarketRegime.RANGING
        elif indicators.bb_width < 0.02:  # Low volatility
            market_regime = MarketRegime.LOW_VOLATILITY
        elif trend == TrendDirection.BULLISH and trend_strength > 60:
            market_regime = MarketRegime.TRENDING_UP
        elif trend == TrendDirection.BEARISH and trend_strength > 60:
            market_regime = MarketRegime.TRENDING_DOWN
        else:
            market_regime = MarketRegime.RANGING
        
        # Volatility percentile
        try:
            df_temp = df.copy()
            df_temp['atr_hist'] = ta.atr(df_temp['high'], df_temp['low'], df_temp['close'], length=14)
            atr_series = df_temp['atr_hist'].dropna().values
            if len(atr_series) > 0:
                volatility_percentile = (np.sum(atr_series <= indicators.atr) / len(atr_series)) * 100
            else:
                volatility_percentile = 50.0
        except Exception as e:
            logger.debug(f"Volatility percentile calculation failed: {e}")
            volatility_percentile = 50.0
        
        # Momentum
        momentum = indicators.rsi - 50  # Convert RSI to momentum scale
        
        # Volume trend
        if len(volume) >= 20:
            recent_volume = np.mean(volume[-5:])
            older_volume = np.mean(volume[-20:-5])
            if recent_volume > older_volume * 1.2:
                volume_trend = "increasing"
            elif recent_volume < older_volume * 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
        else:
            volume_trend = "stable"
        
        # Breakout potential
        price_position = (close[-1] - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower)
        if price_position > 0.8 or price_position < 0.2:
            if indicators.volume_ratio > 1.5:
                breakout_potential = 80.0
            else:
                breakout_potential = 60.0
        elif indicators.bb_width < 0.02:  # Squeeze
            breakout_potential = 70.0
        else:
            breakout_potential = 30.0
        
        # Reversal probability
        reversal_probability = 0.0
        if indicators.rsi > 70:
            reversal_probability += 30
        elif indicators.rsi < 30:
            reversal_probability += 30
        
        if indicators.stoch_k and indicators.stoch_d:
            if indicators.stoch_k > 80 and indicators.stoch_d > 80:
                reversal_probability += 20
            elif indicators.stoch_k < 20 and indicators.stoch_d < 20:
                reversal_probability += 20
        
        # MACD divergence check
        if len(close) >= 50:
            price_trend = 1 if close[-1] > close[-10] else -1
            macd_trend = 1 if indicators.macd > indicators.macd_signal else -1
            if price_trend != macd_trend:
                reversal_probability += 30
        
        reversal_probability = min(reversal_probability, 100)
        
        return MarketStructure(
            trend=trend,
            trend_strength=trend_strength,
            market_regime=market_regime,
            volatility_percentile=volatility_percentile,
            momentum=momentum,
            volume_trend=volume_trend,
            breakout_potential=breakout_potential,
            reversal_probability=reversal_probability,
        )
    
    def _detect_patterns(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> List[str]:
        """Detect chart patterns and formations"""
        patterns = []
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        
        if len(close) < 10:
            return patterns
        
        # Candlestick patterns (Manual detection to avoid TA-Lib log spam on Railway)
        try:
            if len(df) >= 3:
                last_candle = df.iloc[-1]
                
                # Simple pattern detection
                body = abs(last_candle['close'] - last_candle['open'])
                total_range = last_candle['high'] - last_candle['low']
                
                # Ensure we don't divide by zero
                if total_range > 0:
                    # Hammer-like pattern
                    if body / total_range < 0.3 and last_candle['close'] > last_candle['open']:
                        patterns.append("hammer_like")
    
                    # Shooting star-like
                    elif body / total_range < 0.3 and last_candle['close'] < last_candle['open']:
                        patterns.append("shooting_star_like")
                    
                    # Strong bullish candle
                    if last_candle['close'] > last_candle['open'] and body / total_range > 0.7:
                        patterns.append("strong_bullish")
    
                    # Strong bearish candle
                    elif total_range > 0 and last_candle['close'] < last_candle['open'] and body / total_range > 0.7:
                        patterns.append("strong_bearish")
        except Exception as e:
            logger.debug(f"Simplified pattern detection failed: {e}")
        
        # Technical patterns
        
        # Golden/Death cross
        if indicators.ema_50 and indicators.ema_200:
            if indicators.ema_50 > indicators.ema_200 and indicators.ema_21 > indicators.ema_50:
                patterns.append("golden_cross")
            elif indicators.ema_50 < indicators.ema_200 and indicators.ema_21 < indicators.ema_50:
                patterns.append("death_cross")
        
        # MACD crossover
        if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
            patterns.append("macd_bullish_crossover")
        elif indicators.macd < indicators.macd_signal and indicators.macd_histogram < 0:
            patterns.append("macd_bearish_crossover")
        
        # RSI patterns
        if indicators.rsi > 70:
            patterns.append("rsi_overbought")
        elif indicators.rsi < 30:
            patterns.append("rsi_oversold")
        
        # Bollinger Band patterns
        if close[-1] > indicators.bb_upper:
            patterns.append("bb_breakout_up")
        elif close[-1] < indicators.bb_lower:
            patterns.append("bb_breakout_down")
        
        if indicators.bb_width < 0.02:
            patterns.append("bb_squeeze")
        
        # Volume patterns
        if indicators.volume_ratio > 2.0:
            patterns.append("volume_spike")
        elif indicators.volume_ratio < 0.5:
            patterns.append("volume_dry_up")
        
        return patterns
    
    def get_market_sentiment(
        self,
        analyses: Dict[str, TimeframeAnalysis]
    ) -> Dict[str, Any]:
        """Get overall market sentiment from multiple timeframe analysis"""
        if not analyses:
            return {
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "signals": [],
            }
        
        bullish_signals = 0
        bearish_signals = 0
        signals = []
        
        # Weight by timeframe (higher timeframes have more weight)
        weights = {
            "1m": 1,
            "5m": 2,
            "15m": 3,
            "1h": 4,
            "4h": 5,
            "1d": 6,
        }
        
        total_weight = 0
        
        for tf, analysis in analyses.items():
            weight = weights.get(tf, 1)
            total_weight += weight
            
            # Check trend
            if analysis.structure.trend == TrendDirection.BULLISH:
                bullish_signals += weight
                signals.append(f"{tf}: Bullish trend")
            elif analysis.structure.trend == TrendDirection.BEARISH:
                bearish_signals += weight
                signals.append(f"{tf}: Bearish trend")
            
            # Check patterns
            for pattern in analysis.patterns:
                if "bullish" in pattern.lower() or pattern in ["hammer", "morning_star", "golden_cross"]:
                    bullish_signals += weight * 0.5
                    signals.append(f"{tf}: {pattern}")
                elif "bearish" in pattern.lower() or pattern in ["shooting_star", "evening_star", "death_cross"]:
                    bearish_signals += weight * 0.5
                    signals.append(f"{tf}: {pattern}")
        
        # Calculate overall sentiment
        if total_weight > 0:
            bullish_score = bullish_signals / total_weight
            bearish_score = bearish_signals / total_weight
            
            if bullish_score > bearish_score * 1.5:
                sentiment = "bullish"
                confidence = min((bullish_score / (bullish_score + bearish_score)) * 100, 100)
            elif bearish_score > bullish_score * 1.5:
                sentiment = "bearish"
                confidence = min((bearish_score / (bullish_score + bearish_score)) * 100, 100)
            else:
                sentiment = "neutral"
                confidence = 50.0
        else:
            sentiment = "neutral"
            confidence = 0.0
        
        return {
            "overall_sentiment": sentiment,
            "confidence": confidence,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "signals": signals[:10],  # Top 10 signals
        }
    
    def _detect_divergence(
        self,
        price: np.ndarray,
        indicator: np.ndarray,
        lookback: int = 10,
    ) -> str:
        """
        Detect bullish or bearish divergence
        
        Args:
            price: Price array
            indicator: Indicator array (RSI, MACD, etc.)
            lookback: Periods to look back
        
        Returns:
            "bullish", "bearish", or "none"
        """
        if len(price) < lookback or len(indicator) < lookback:
            return "none"
        
        try:
            # Get recent data
            recent_price = price[-lookback:]
            recent_indicator = indicator[-lookback:]
            
            # Remove NaN values
            mask = ~(np.isnan(recent_price) | np.isnan(recent_indicator))
            recent_price = recent_price[mask]
            recent_indicator = recent_indicator[mask]
            
            if len(recent_price) < 5:
                return "none"
            
            # Find highs and lows
            price_high_idx = np.argmax(recent_price)
            price_low_idx = np.argmin(recent_price)
            
            # Bullish Divergence: Price making lower lows, indicator making higher lows
            if price_low_idx > len(recent_price) // 2:  # Recent low
                earlier_lows = recent_price[:price_low_idx]
                if len(earlier_lows) > 0 and recent_price[price_low_idx] < np.min(earlier_lows):
                    # Price made lower low
                    if recent_indicator[price_low_idx] > np.min(recent_indicator[:price_low_idx]):
                        # Indicator made higher low
                        return "bullish"
            
            # Bearish Divergence: Price making higher highs, indicator making lower highs
            if price_high_idx > len(recent_price) // 2:  # Recent high
                earlier_highs = recent_price[:price_high_idx]
                if len(earlier_highs) > 0 and recent_price[price_high_idx] > np.max(earlier_highs):
                    # Price made higher high
                    if recent_indicator[price_high_idx] < np.max(recent_indicator[:price_high_idx]):
                        # Indicator made lower high
                        return "bearish"
            
            return "none"
            
        except Exception as e:
            logger.debug(f"Divergence detection failed: {e}")
            return "none"
