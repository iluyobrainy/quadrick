"""
Momentum Ranking System for Scalper Mode
Ranks all symbols by 1-minute momentum strength to pick the best
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
import pandas as pd


class MomentumRanker:
    """Ranks symbols by momentum strength for scalping"""
    
    def __init__(self):
        """Initialize momentum ranker"""
        self.last_rankings = {}
        self.ranking_history = []
        logger.info("Momentum ranker initialized")
    
    def rank_symbols(
        self,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        mode: str = "scalper"
    ) -> List[Dict[str, Any]]:
        """
        Rank all symbols by momentum strength
        
        Args:
            market_data: Market data for all symbols
            technical_analysis: Technical indicators for all symbols
            mode: Trading mode (scalper/normal)
        
        Returns:
            List of symbols ranked by momentum score
        """
        rankings = []
        
        for symbol in market_data:
            try:
                score = self._calculate_momentum_score(
                    symbol,
                    market_data.get(symbol, {}),
                    technical_analysis.get(symbol, {}),
                    mode
                )
                
                rankings.append({
                    'symbol': symbol,
                    'score': score['total'],
                    'direction': score['direction'],
                    'entry_reason': score['reason'],
                    'components': score['components'],
                    'confidence': score['confidence']
                })
                
            except Exception as e:
                logger.error(f"Error ranking {symbol}: {e}")
                continue
        
        # Sort by score (highest first)
        rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)
        
        # Store rankings
        self.last_rankings = {r['symbol']: r for r in rankings}
        self.ranking_history.append({
            'timestamp': datetime.utcnow(),
            'rankings': rankings[:5]  # Top 5 only
        })
        
        # Keep only last 10 ranking snapshots
        if len(self.ranking_history) > 10:
            self.ranking_history.pop(0)
        
        return rankings
    
    def _calculate_momentum_score(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """Calculate momentum score for a single symbol"""
        
        components = {}
        total_score = 0
        direction = 'neutral'
        reason = []
        
        # Get 1m data (most important for scalping)
        data_1m = technical_data.get('1', {})
        indicators_1m = data_1m.get('indicators', {})
        
        # 1. VOLUME SPIKE (0-40 points) - MOST IMPORTANT FOR SCALPING
        volume_spike = data_1m.get('volume_spike', {})
        if volume_spike.get('is_spike', False):
            spike_score = min(40, volume_spike.get('spike_ratio', 1) * 10)
            components['volume_spike'] = spike_score
            total_score += spike_score
            reason.append(f"Volume spike {volume_spike.get('spike_ratio', 0)}x")
            
            # Extra points for high-quality spikes
            if volume_spike.get('quality') == 'high':
                total_score += 10
                components['volume_quality'] = 10
        
        # 2. PRICE MOMENTUM (0-30 points)
        if '1m_change' in market_data:
            one_min_change = abs(market_data['1m_change'])
            momentum_score = min(30, one_min_change * 50)  # 0.6% = 30 points
            components['price_momentum'] = momentum_score
            total_score += momentum_score
            
            if one_min_change > 0.3:  # Strong 1m move
                reason.append(f"Strong 1m move {one_min_change:.2f}%")
            
            # Determine direction
            direction = 'long' if market_data['1m_change'] > 0 else 'short'
        
        # 3. RSI EXTREMES (0-20 points)
        rsi = indicators_1m.get('rsi', 50)
        if rsi > 80 or rsi < 20:  # Extreme overbought/oversold
            rsi_score = 20
            reason.append(f"RSI extreme {rsi:.0f}")
        elif rsi > 70 or rsi < 30:  # Strong overbought/oversold
            rsi_score = 15
            reason.append(f"RSI strong {rsi:.0f}")
        elif rsi > 60 or rsi < 40:  # Moderate
            rsi_score = 10
        else:
            rsi_score = 0
        
        components['rsi_extreme'] = rsi_score
        total_score += rsi_score
        
        # 4. BREAKOUT DETECTION (0-15 points)
        # Check if breaking recent highs/lows
        if data_1m.get('breaking_high', False):
            total_score += 15
            components['breakout'] = 15
            reason.append("Breaking 1m high")
            direction = 'long'
        elif data_1m.get('breaking_low', False):
            total_score += 15
            components['breakout'] = 15
            reason.append("Breaking 1m low")
            direction = 'short'
        
        # 5. CONSECUTIVE CANDLES (0-10 points)
        # Reward momentum continuation
        consecutive = data_1m.get('consecutive_candles', 0)
        if abs(consecutive) >= 3:  # 3+ candles in same direction
            consec_score = min(10, abs(consecutive) * 3)
            components['consecutive'] = consec_score
            total_score += consec_score
            reason.append(f"{abs(consecutive)} consecutive candles")
        
        # 6. MACD MOMENTUM (0-10 points)
        macd = indicators_1m.get('macd', 0)
        macd_signal = indicators_1m.get('macd_signal', 0)
        if macd > macd_signal and macd > 0:  # Bullish and positive
            total_score += 10
            components['macd'] = 10
        elif macd < macd_signal and macd < 0:  # Bearish and negative
            total_score += 10
            components['macd'] = 10
        
        # 7. VOLATILITY BONUS (0-5 points)
        # Scalping loves volatility
        atr = indicators_1m.get('atr', 0)
        if market_data.get('price', 1) > 0:
            atr_pct = (atr / market_data['price']) * 100
            if atr_pct > 0.5:  # High volatility
                total_score += 5
                components['volatility'] = 5
        
        # Calculate confidence based on score
        if total_score > 70:
            confidence = 0.9
        elif total_score > 50:
            confidence = 0.75
        elif total_score > 30:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Format reason
        if not reason:
            reason = ["No clear momentum"]
        
        return {
            'total': total_score,
            'direction': direction,
            'reason': ', '.join(reason),
            'components': components,
            'confidence': confidence
        }
    
    def get_best_scalping_opportunity(
        self,
        rankings: List[Dict[str, Any]],
        min_score: float = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best scalping opportunity from rankings
        
        Args:
            rankings: Momentum rankings
            min_score: Minimum score to consider
        
        Returns:
            Best opportunity or None
        """
        if not rankings:
            return None
        
        # Filter by minimum score
        valid_opportunities = [r for r in rankings if r['score'] >= min_score]
        
        if not valid_opportunities:
            logger.info(f"No symbols meet minimum score of {min_score}")
            return None
        
        # Return the best
        best = valid_opportunities[0]
        logger.info(
            f"🎯 Best scalping opportunity: {best['symbol']} "
            f"(Score: {best['score']:.1f}, Direction: {best['direction']}, "
            f"Reason: {best['entry_reason']})"
        )
        
        return best
    
    def get_multiple_opportunities(
        self,
        rankings: List[Dict[str, Any]],
        max_positions: int = 3,
        min_score: float = 25
    ) -> List[Dict[str, Any]]:
        """
        Get multiple scalping opportunities for multi-position trading
        
        Args:
            rankings: Momentum rankings
            max_positions: Maximum number of opportunities
            min_score: Minimum score to consider
        
        Returns:
            List of opportunities
        """
        valid_opportunities = [r for r in rankings if r['score'] >= min_score]
        
        # Diversify - don't take all longs or all shorts
        selected = []
        long_count = 0
        short_count = 0
        
        for opp in valid_opportunities:
            if len(selected) >= max_positions:
                break
            
            # Balance long/short
            if opp['direction'] == 'long' and long_count >= 2:
                continue
            if opp['direction'] == 'short' and short_count >= 2:
                continue
            
            selected.append(opp)
            
            if opp['direction'] == 'long':
                long_count += 1
            else:
                short_count += 1
        
        if selected:
            logger.info(f"📊 Found {len(selected)} scalping opportunities:")
            for opp in selected:
                logger.info(
                    f"  • {opp['symbol']}: Score {opp['score']:.1f}, "
                    f"{opp['direction'].upper()}, {opp['entry_reason']}"
                )
        
        return selected
    
    def get_momentum_summary(self) -> Dict[str, Any]:
        """Get summary of current momentum across all symbols"""
        if not self.last_rankings:
            return {'status': 'No data'}
        
        # Count bullish vs bearish
        bullish = sum(1 for r in self.last_rankings.values() if r['direction'] == 'long')
        bearish = sum(1 for r in self.last_rankings.values() if r['direction'] == 'short')
        
        # Average score
        avg_score = sum(r['score'] for r in self.last_rankings.values()) / len(self.last_rankings)
        
        # Top movers
        top_3 = sorted(self.last_rankings.values(), key=lambda x: x['score'], reverse=True)[:3]
        
        return {
            'bullish_count': bullish,
            'bearish_count': bearish,
            'average_score': round(avg_score, 1),
            'market_sentiment': 'bullish' if bullish > bearish else 'bearish' if bearish > bullish else 'neutral',
            'top_movers': [
                {'symbol': r['symbol'], 'score': r['score'], 'direction': r['direction']}
                for r in top_3
            ],
            'high_momentum_count': sum(1 for r in self.last_rankings.values() if r['score'] > 50)
        }
