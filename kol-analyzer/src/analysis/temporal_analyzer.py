"""
Temporal Analyzer - Analyze timing of posts relative to price action.

Detects:
- Front-running: Posts bullish before pumps consistently
- Lag calling: Always calls after the move happened
- Crisis behavior: How they act during market crashes
- Pump timing: Correlation between shills and price movements
"""

import re
import httpx
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TimingPattern(Enum):
    """Timing pattern classifications."""
    FRONT_RUNNER = "front_runner"  # Consistently posts before price moves
    LAG_CALLER = "lag_caller"  # Posts after moves, claims hindsight
    NEUTRAL_TIMING = "neutral_timing"  # Random/no pattern
    CRISIS_AVOIDER = "crisis_avoider"  # Goes silent during crashes
    CRISIS_EXPLOITER = "crisis_exploiter"  # Posts FUD during crashes
    CRISIS_SUPPORTER = "crisis_supporter"  # Supportive during crashes


@dataclass
class TimingEvent:
    """A detected timing event."""
    tweet_id: str
    tweet_text: str
    timestamp: str
    token: str
    sentiment: str  # "bullish" or "bearish"
    price_at_tweet: Optional[float]
    price_24h_later: Optional[float]
    price_change_pct: Optional[float]
    was_predictive: bool  # Did the price move in predicted direction?
    timing_type: str  # "before_move", "after_move", "during_crash"

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:100] + '...' if len(self.tweet_text) > 100 else self.tweet_text,
            'timestamp': self.timestamp,
            'token': self.token,
            'sentiment': self.sentiment,
            'price_at_tweet': self.price_at_tweet,
            'price_24h_later': self.price_24h_later,
            'price_change_pct': round(self.price_change_pct, 2) if self.price_change_pct else None,
            'was_predictive': self.was_predictive,
            'timing_type': self.timing_type
        }


@dataclass
class TemporalReport:
    """Report on timing patterns."""
    timing_score: float = 50.0  # 0-100, higher = more suspicious timing
    primary_pattern: TimingPattern = TimingPattern.NEUTRAL_TIMING
    events: List[TimingEvent] = field(default_factory=list)

    # Statistics
    predictive_calls: int = 0  # Calls where price moved in predicted direction
    reactive_calls: int = 0  # Calls after price already moved
    total_timed_calls: int = 0

    # Front-running indicators
    front_run_score: float = 0.0  # 0-100
    avg_lead_time_hours: float = 0.0  # Average hours before price move
    suspiciously_early_calls: int = 0  # Calls that were "too good"

    # Crisis behavior
    crash_tweet_count: int = 0
    crash_sentiment: str = ""  # "supportive", "exploitative", "silent"

    # Pattern details
    patterns_detected: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'timing_score': round(self.timing_score, 1),
            'primary_pattern': self.primary_pattern.value,
            'events': [e.to_dict() for e in self.events[:15]],
            'predictive_calls': self.predictive_calls,
            'reactive_calls': self.reactive_calls,
            'total_timed_calls': self.total_timed_calls,
            'front_run_score': round(self.front_run_score, 1),
            'avg_lead_time_hours': round(self.avg_lead_time_hours, 1),
            'suspiciously_early_calls': self.suspiciously_early_calls,
            'crash_tweet_count': self.crash_tweet_count,
            'crash_sentiment': self.crash_sentiment,
            'patterns_detected': self.patterns_detected,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class TemporalAnalyzer:
    """
    Analyzes timing of KOL posts relative to price movements.

    Key questions:
    1. Do they post BEFORE or AFTER price moves?
    2. Do they go silent during crashes or double down?
    3. Is their timing suspiciously good (potential insider info)?
    """

    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Token mappings
    TICKER_TO_ID = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'DOGE': 'dogecoin', 'PEPE': 'pepe', 'ARB': 'arbitrum',
        'OP': 'optimism', 'MATIC': 'matic-network', 'AVAX': 'avalanche-2',
        'LINK': 'chainlink', 'UNI': 'uniswap', 'AAVE': 'aave',
        'SUI': 'sui', 'APT': 'aptos', 'SEI': 'sei-network',
        'TIA': 'celestia', 'INJ': 'injective-protocol', 'NEAR': 'near',
        'ATOM': 'cosmos', 'DOT': 'polkadot', 'ADA': 'cardano',
        'BNB': 'binancecoin', 'WIF': 'dogwifcoin', 'BONK': 'bonk',
        'JUP': 'jupiter-exchange-solana', 'STRK': 'starknet',
    }

    BULLISH_KEYWORDS = [
        r'\b(bullish|moon|pump|buy|long|accumulate|ape|send)\b',
        r'\b(undervalued|gem|alpha|breakout|bottom)\b',
        r'\b(100x|10x|going up|ripping)\b',
    ]

    BEARISH_KEYWORDS = [
        r'\b(bearish|dump|sell|short|exit|avoid)\b',
        r'\b(overvalued|scam|rug|top|crash)\b',
        r'\b(going down|tanking|rekt)\b',
    ]

    CRISIS_KEYWORDS = [
        r'\b(crash|collapse|blood|capitulation|panic)\b',
        r'\b(black swan|liquidat|margin call|blow up)\b',
    ]

    # Thresholds
    SIGNIFICANT_MOVE_PCT = 10.0  # 10% move is significant
    FRONT_RUN_WINDOW_HOURS = 48  # Check price 48h after tweet
    SUSPICIOUSLY_EARLY_PCT = 20.0  # 20%+ move after call is suspicious

    def __init__(self):
        self.bullish_patterns = [re.compile(p, re.IGNORECASE) for p in self.BULLISH_KEYWORDS]
        self.bearish_patterns = [re.compile(p, re.IGNORECASE) for p in self.BEARISH_KEYWORDS]
        self.crisis_patterns = [re.compile(p, re.IGNORECASE) for p in self.CRISIS_KEYWORDS]
        self.ticker_pattern = re.compile(r'\$([A-Z]{2,10})\b')
        self._price_cache: Dict[str, Dict] = {}

    async def analyze(self, tweets: List[dict]) -> TemporalReport:
        """Analyze timing patterns in tweets."""
        if not tweets:
            return TemporalReport()

        events: List[TimingEvent] = []
        predictive = 0
        reactive = 0
        lead_times = []
        crash_tweets = []

        # Sort by timestamp
        sorted_tweets = sorted(
            tweets,
            key=lambda t: t.get('timestamp', ''),
            reverse=False
        )

        for tweet in sorted_tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            if not timestamp:
                continue

            # Extract tickers
            tickers = self.ticker_pattern.findall(text)
            if not tickers:
                continue

            # Classify sentiment
            sentiment = self._classify_sentiment(text)
            if sentiment == 'neutral':
                continue

            # Check if crisis-related
            is_crisis_tweet = any(p.search(text) for p in self.crisis_patterns)
            if is_crisis_tweet:
                crash_tweets.append({
                    'text': text,
                    'timestamp': timestamp,
                    'sentiment': sentiment
                })

            # Analyze timing for each mentioned token
            for ticker in set(tickers):
                ticker = ticker.upper()
                token_id = self.TICKER_TO_ID.get(ticker)
                if not token_id:
                    continue

                # Get price data around tweet time
                price_data = await self._get_price_around_time(token_id, timestamp)
                if not price_data:
                    continue

                price_at_tweet = price_data.get('price_at_time')
                price_after = price_data.get('price_after')
                price_before = price_data.get('price_before')

                if not price_at_tweet:
                    continue

                # Calculate price changes
                pct_after = ((price_after - price_at_tweet) / price_at_tweet * 100) if price_after else None
                pct_before = ((price_at_tweet - price_before) / price_before * 100) if price_before else None

                # Determine if predictive or reactive
                was_predictive = False
                timing_type = "neutral"

                if pct_after is not None:
                    if sentiment == 'bullish' and pct_after > self.SIGNIFICANT_MOVE_PCT:
                        was_predictive = True
                        timing_type = "before_move"
                        predictive += 1
                        if pct_after > self.SUSPICIOUSLY_EARLY_PCT:
                            lead_times.append(pct_after)
                    elif sentiment == 'bearish' and pct_after < -self.SIGNIFICANT_MOVE_PCT:
                        was_predictive = True
                        timing_type = "before_move"
                        predictive += 1

                if pct_before is not None and abs(pct_before) > self.SIGNIFICANT_MOVE_PCT:
                    # Price already moved before tweet
                    if not was_predictive:
                        timing_type = "after_move"
                        reactive += 1

                events.append(TimingEvent(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    token=ticker,
                    sentiment=sentiment,
                    price_at_tweet=price_at_tweet,
                    price_24h_later=price_after,
                    price_change_pct=pct_after,
                    was_predictive=was_predictive,
                    timing_type=timing_type
                ))

                # Rate limiting
                await asyncio.sleep(0.5)

        # Calculate scores and patterns
        total_timed = predictive + reactive

        # Front-run score (higher = more suspicious)
        front_run_score = 0.0
        if total_timed > 0:
            predictive_ratio = predictive / total_timed
            if predictive_ratio > 0.7:
                front_run_score = 80 + (predictive_ratio - 0.7) * 66
            elif predictive_ratio > 0.5:
                front_run_score = 50 + (predictive_ratio - 0.5) * 150
            else:
                front_run_score = predictive_ratio * 100

        # Determine primary pattern
        primary_pattern = self._determine_pattern(
            predictive, reactive, lead_times, crash_tweets
        )

        # Analyze crisis behavior
        crash_sentiment = self._analyze_crisis_behavior(crash_tweets)

        # Generate flags
        red_flags, green_flags = self._generate_flags(
            front_run_score, predictive, reactive, lead_times, crash_sentiment
        )

        # Calculate timing score (higher = more concerning)
        timing_score = self._calculate_timing_score(
            front_run_score, len(lead_times), primary_pattern
        )

        patterns_detected = self._detect_patterns(
            predictive, reactive, lead_times, crash_sentiment
        )

        return TemporalReport(
            timing_score=timing_score,
            primary_pattern=primary_pattern,
            events=events,
            predictive_calls=predictive,
            reactive_calls=reactive,
            total_timed_calls=total_timed,
            front_run_score=front_run_score,
            avg_lead_time_hours=sum(lead_times) / len(lead_times) if lead_times else 0,
            suspiciously_early_calls=len([l for l in lead_times if l > self.SUSPICIOUSLY_EARLY_PCT]),
            crash_tweet_count=len(crash_tweets),
            crash_sentiment=crash_sentiment,
            patterns_detected=patterns_detected,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _classify_sentiment(self, text: str) -> str:
        """Classify tweet sentiment."""
        bullish = sum(1 for p in self.bullish_patterns if p.search(text))
        bearish = sum(1 for p in self.bearish_patterns if p.search(text))

        if bullish > bearish:
            return 'bullish'
        elif bearish > bullish:
            return 'bearish'
        return 'neutral'

    async def _get_price_around_time(
        self,
        token_id: str,
        timestamp: str
    ) -> Optional[Dict]:
        """Get price data around a specific time."""
        cache_key = f"{token_id}_{timestamp[:10]}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            tweet_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # Get historical data
            async with httpx.AsyncClient() as client:
                url = f"{self.COINGECKO_API}/coins/{token_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': 90
                }
                response = await client.get(url, params=params, timeout=10)

                if response.status_code != 200:
                    return None

                data = response.json()
                prices = data.get('prices', [])

                if not prices:
                    return None

                # Find prices at, before, and after tweet time
                tweet_ts = tweet_time.timestamp() * 1000
                before_ts = (tweet_time - timedelta(hours=24)).timestamp() * 1000
                after_ts = (tweet_time + timedelta(hours=48)).timestamp() * 1000

                price_at = self._find_closest_price(prices, tweet_ts)
                price_before = self._find_closest_price(prices, before_ts)
                price_after = self._find_closest_price(prices, after_ts)

                result = {
                    'price_at_time': price_at,
                    'price_before': price_before,
                    'price_after': price_after
                }

                self._price_cache[cache_key] = result
                return result

        except Exception:
            return None

    def _find_closest_price(
        self,
        prices: List[List],
        target_ts: float
    ) -> Optional[float]:
        """Find price closest to target timestamp."""
        if not prices:
            return None

        closest = min(prices, key=lambda p: abs(p[0] - target_ts))
        return closest[1]

    def _determine_pattern(
        self,
        predictive: int,
        reactive: int,
        lead_times: List[float],
        crash_tweets: List[dict]
    ) -> TimingPattern:
        """Determine primary timing pattern."""
        total = predictive + reactive
        if total < 3:
            return TimingPattern.NEUTRAL_TIMING

        predictive_ratio = predictive / total

        if predictive_ratio > 0.7 and len(lead_times) > 2:
            return TimingPattern.FRONT_RUNNER
        elif reactive / total > 0.7:
            return TimingPattern.LAG_CALLER

        return TimingPattern.NEUTRAL_TIMING

    def _analyze_crisis_behavior(self, crash_tweets: List[dict]) -> str:
        """Analyze behavior during crashes."""
        if not crash_tweets:
            return "neutral"

        bullish_count = sum(1 for t in crash_tweets if t['sentiment'] == 'bullish')
        bearish_count = sum(1 for t in crash_tweets if t['sentiment'] == 'bearish')

        if bullish_count > bearish_count * 2:
            return "supportive"
        elif bearish_count > bullish_count * 2:
            return "exploitative"
        return "mixed"

    def _generate_flags(
        self,
        front_run_score: float,
        predictive: int,
        reactive: int,
        lead_times: List[float],
        crash_sentiment: str
    ) -> Tuple[List[str], List[str]]:
        """Generate red and green flags."""
        red_flags = []
        green_flags = []

        if front_run_score > 70:
            red_flags.append(f"Suspiciously good timing on calls ({front_run_score:.0f}% front-run score)")

        if len(lead_times) > 3:
            avg_gain = sum(lead_times) / len(lead_times)
            red_flags.append(f"Multiple calls before {avg_gain:.0f}%+ moves - possible insider info")

        if reactive > predictive * 2 and reactive > 5:
            red_flags.append("Primarily calls AFTER price moves (hindsight trading)")

        if crash_sentiment == "exploitative":
            red_flags.append("Spreads FUD during market crashes")

        # Green flags
        if predictive > 0 and front_run_score < 50:
            green_flags.append("Some predictive calls with reasonable timing")

        if crash_sentiment == "supportive":
            green_flags.append("Supportive during market downturns")

        total = predictive + reactive
        if total > 5 and 0.4 < predictive / total < 0.6:
            green_flags.append("Balanced timing - not suspiciously good or bad")

        return red_flags[:3], green_flags[:3]

    def _calculate_timing_score(
        self,
        front_run_score: float,
        suspicious_calls: int,
        pattern: TimingPattern
    ) -> float:
        """Calculate overall timing suspicion score."""
        score = 50.0  # Neutral baseline

        # Front-running adds to score (more suspicious)
        if front_run_score > 60:
            score += (front_run_score - 60) * 0.5

        # Suspicious calls
        score += min(20, suspicious_calls * 5)

        # Pattern adjustments
        if pattern == TimingPattern.FRONT_RUNNER:
            score += 15
        elif pattern == TimingPattern.LAG_CALLER:
            score -= 10  # Less concerning, just not useful

        return max(0, min(100, score))

    def _detect_patterns(
        self,
        predictive: int,
        reactive: int,
        lead_times: List[float],
        crash_sentiment: str
    ) -> List[str]:
        """Detect and describe patterns."""
        patterns = []

        if lead_times and len(lead_times) >= 3:
            patterns.append(f"Frequently posts before major price moves ({len(lead_times)} instances)")

        if reactive > predictive * 2:
            patterns.append("Posts mostly after price action (reactive trading)")

        if crash_sentiment == "supportive":
            patterns.append("Encourages holding during crashes")
        elif crash_sentiment == "exploitative":
            patterns.append("Amplifies panic during downturns")

        return patterns

    def generate_summary(self, report: TemporalReport) -> str:
        """Generate human-readable summary."""
        if report.total_timed_calls < 3:
            return "Insufficient data to analyze timing patterns."

        parts = []

        if report.primary_pattern == TimingPattern.FRONT_RUNNER:
            parts.append("Shows suspiciously good timing on calls.")
        elif report.primary_pattern == TimingPattern.LAG_CALLER:
            parts.append("Typically calls after the move already happened.")
        else:
            parts.append("Timing patterns appear normal.")

        if report.suspiciously_early_calls > 2:
            parts.append(f"Made {report.suspiciously_early_calls} calls before 20%+ moves.")

        if report.crash_sentiment == "supportive":
            parts.append("Supportive during market stress.")

        return " ".join(parts)
