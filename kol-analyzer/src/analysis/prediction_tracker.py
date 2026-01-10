"""
Prediction Accuracy Tracker - Track accuracy of token calls and predictions.

Uses CoinGecko API (free) to verify if KOL predictions were accurate.
Tracks mentions of tokens and their subsequent price performance.
"""

import re
import time
import httpx
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import asyncio


class CallType(Enum):
    """Type of call made by KOL."""
    BULLISH = "bullish"  # Expecting price to go up
    BEARISH = "bearish"  # Expecting price to go down
    NEUTRAL = "neutral"  # Just mentioning, no direction


class CallOutcome(Enum):
    """Outcome of a prediction."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"  # Not enough time passed
    UNKNOWN = "unknown"  # Couldn't verify


@dataclass
class TokenCall:
    """A prediction/call made about a token."""
    tweet_id: str
    tweet_text: str
    timestamp: str
    token_symbol: str
    token_id: Optional[str]  # CoinGecko ID
    call_type: CallType
    confidence_words: List[str]  # Words that indicated confidence level
    price_at_call: Optional[float] = None
    current_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    outcome: CallOutcome = CallOutcome.PENDING
    days_since_call: int = 0

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:120] + '...' if len(self.tweet_text) > 120 else self.tweet_text,
            'timestamp': self.timestamp,
            'token_symbol': self.token_symbol,
            'call_type': self.call_type.value,
            'confidence_words': self.confidence_words,
            'price_at_call': self.price_at_call,
            'current_price': self.current_price,
            'price_change_pct': round(self.price_change_pct, 2) if self.price_change_pct else None,
            'outcome': self.outcome.value,
            'days_since_call': self.days_since_call
        }


@dataclass
class PredictionReport:
    """Report on prediction accuracy."""
    calls: List[TokenCall] = field(default_factory=list)
    accuracy_score: float = 50.0  # 0-100
    total_calls: int = 0
    correct_calls: int = 0
    incorrect_calls: int = 0
    pending_calls: int = 0
    hit_rate: float = 0.0  # Percentage of correct calls
    avg_gain_on_correct: float = 0.0  # Average % gain when right
    avg_loss_on_incorrect: float = 0.0  # Average % loss when wrong
    most_shilled_tokens: List[Tuple[str, int]] = field(default_factory=list)
    high_confidence_accuracy: float = 0.0  # Accuracy on "100x", "gem" calls
    confidence_calibration: str = ""  # "overconfident", "calibrated", "underconfident"
    token_performance: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'calls': [c.to_dict() for c in self.calls[:20]],
            'accuracy_score': round(self.accuracy_score, 1),
            'total_calls': self.total_calls,
            'correct_calls': self.correct_calls,
            'incorrect_calls': self.incorrect_calls,
            'pending_calls': self.pending_calls,
            'hit_rate': round(self.hit_rate, 1),
            'avg_gain_on_correct': round(self.avg_gain_on_correct, 2),
            'avg_loss_on_incorrect': round(self.avg_loss_on_incorrect, 2),
            'most_shilled_tokens': self.most_shilled_tokens[:10],
            'high_confidence_accuracy': round(self.high_confidence_accuracy, 1),
            'confidence_calibration': self.confidence_calibration,
            'token_performance': self.token_performance
        }


class PredictionTracker:
    """
    Tracks prediction accuracy using CoinGecko API.

    Methodology:
    1. Extract token mentions ($TICKER format)
    2. Classify as bullish/bearish based on context
    3. Look up historical price at tweet time
    4. Compare to current price or price 30 days later
    5. Score based on accuracy
    """

    # CoinGecko API base URL (free tier)
    COINGECKO_API = "https://api.coingecko.com/api/v3"

    # Common ticker to CoinGecko ID mappings
    TICKER_TO_ID = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu',
        'PEPE': 'pepe',
        'ARB': 'arbitrum',
        'OP': 'optimism',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'AAVE': 'aave',
        'CRV': 'curve-dao-token',
        'LDO': 'lido-dao',
        'APE': 'apecoin',
        'BLUR': 'blur',
        'SUI': 'sui',
        'APT': 'aptos',
        'SEI': 'sei-network',
        'TIA': 'celestia',
        'INJ': 'injective-protocol',
        'FTM': 'fantom',
        'NEAR': 'near',
        'ATOM': 'cosmos',
        'DOT': 'polkadot',
        'ADA': 'cardano',
        'XRP': 'ripple',
        'BNB': 'binancecoin',
        'TRX': 'tron',
        'TON': 'the-open-network',
        'WIF': 'dogwifcoin',
        'BONK': 'bonk',
        'JUP': 'jupiter-exchange-solana',
        'JTO': 'jito-governance-token',
        'PYTH': 'pyth-network',
        'WLD': 'worldcoin-wld',
        'STRK': 'starknet',
        'MEME': 'memecoin-2',
        'ORDI': 'ordi',
        'RUNE': 'thorchain',
        'STX': 'stacks',
        'IMX': 'immutable-x',
        'RENDER': 'render-token',
        'FET': 'fetch-ai',
        'RNDR': 'render-token',
        'GRT': 'the-graph',
        'AR': 'arweave',
        'FIL': 'filecoin',
        'PENDLE': 'pendle',
        'GMX': 'gmx',
        'DYDX': 'dydx',
        'SNX': 'havven',
        'MKR': 'maker',
        'COMP': 'compound-governance-token',
        'ENS': 'ethereum-name-service',
    }

    # Bullish keywords
    BULLISH_PATTERNS = [
        r'\b(bullish|moon|pump|buy|accumulate|load|ape|long)\b',
        r'\b(undervalued|gem|alpha|conviction|breakout|reversal)\b',
        r'\b(100x|10x|1000x|massive|huge)\b',
        r'\b(going (up|higher|to the moon))\b',
        r'\b(next leg up|bottomed|bottom is in)\b',
        r'\b(this is it|generational|opportunity)\b',
    ]

    # Bearish keywords
    BEARISH_PATTERNS = [
        r'\b(bearish|dump|sell|exit|short|avoid)\b',
        r'\b(overvalued|scam|rug|dead|crash)\b',
        r'\b(going (down|lower|to zero))\b',
        r'\b(top is in|topped|distribution)\b',
        r'\b(stay away|don\'t buy|warning)\b',
    ]

    # High confidence indicators (for calibration)
    HIGH_CONFIDENCE_PATTERNS = [
        r'\b(100x|1000x|guaranteed|definitely|certainly)\b',
        r'\b(can\'t lose|easy money|free money)\b',
        r'\b(trust me|mark my words|save this)\b',
        r'\b(next (btc|eth|sol)|the next)\b',
        r'\b(generational|once in a lifetime)\b',
    ]

    # Price change thresholds for outcome
    BULLISH_CORRECT_THRESHOLD = 10.0  # +10% = correct bullish call
    BEARISH_CORRECT_THRESHOLD = -10.0  # -10% = correct bearish call
    MINIMUM_DAYS_FOR_OUTCOME = 7  # Wait at least 7 days to judge

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize tracker.

        Args:
            api_key: Optional CoinGecko Pro API key for higher rate limits
        """
        self.api_key = api_key
        self.bullish_patterns = [re.compile(p, re.IGNORECASE) for p in self.BULLISH_PATTERNS]
        self.bearish_patterns = [re.compile(p, re.IGNORECASE) for p in self.BEARISH_PATTERNS]
        self.high_confidence_patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_CONFIDENCE_PATTERNS]
        self.ticker_pattern = re.compile(r'\$([A-Z]{2,10})\b')
        self._price_cache: Dict[str, Dict] = {}

    async def analyze(self, tweets: List[dict]) -> PredictionReport:
        """
        Analyze tweets for prediction accuracy.

        Args:
            tweets: List of tweet dictionaries

        Returns:
            PredictionReport with accuracy analysis
        """
        if not tweets:
            return PredictionReport()

        calls: List[TokenCall] = []
        token_counts: Dict[str, int] = {}

        # Extract all token calls
        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Find token mentions
            tickers = self.ticker_pattern.findall(text)
            if not tickers:
                continue

            # Classify call type
            call_type = self._classify_call(text)
            if call_type == CallType.NEUTRAL:
                continue  # Skip neutral mentions

            # Get confidence words
            confidence_words = self._get_confidence_words(text)

            # Calculate days since call
            days_since = self._calculate_days_since(timestamp)

            for ticker in set(tickers):
                ticker = ticker.upper()
                token_id = self.TICKER_TO_ID.get(ticker)

                # Count token mentions
                token_counts[ticker] = token_counts.get(ticker, 0) + 1

                call = TokenCall(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    token_symbol=ticker,
                    token_id=token_id,
                    call_type=call_type,
                    confidence_words=confidence_words,
                    days_since_call=days_since
                )
                calls.append(call)

        # Fetch prices and evaluate outcomes (with rate limiting)
        await self._evaluate_calls(calls)

        # Calculate statistics
        total_calls = len(calls)
        correct_calls = sum(1 for c in calls if c.outcome == CallOutcome.CORRECT)
        incorrect_calls = sum(1 for c in calls if c.outcome == CallOutcome.INCORRECT)
        pending_calls = sum(1 for c in calls if c.outcome == CallOutcome.PENDING)

        # Hit rate
        evaluated = correct_calls + incorrect_calls
        hit_rate = (correct_calls / evaluated * 100) if evaluated > 0 else 50.0

        # Average gains/losses
        correct_gains = [c.price_change_pct for c in calls
                        if c.outcome == CallOutcome.CORRECT and c.price_change_pct]
        incorrect_losses = [c.price_change_pct for c in calls
                          if c.outcome == CallOutcome.INCORRECT and c.price_change_pct]

        avg_gain = sum(correct_gains) / len(correct_gains) if correct_gains else 0.0
        avg_loss = sum(incorrect_losses) / len(incorrect_losses) if incorrect_losses else 0.0

        # High confidence accuracy
        high_conf_calls = [c for c in calls if c.confidence_words]
        high_conf_correct = sum(1 for c in high_conf_calls if c.outcome == CallOutcome.CORRECT)
        high_conf_evaluated = sum(1 for c in high_conf_calls
                                  if c.outcome in [CallOutcome.CORRECT, CallOutcome.INCORRECT])
        high_conf_accuracy = (high_conf_correct / high_conf_evaluated * 100) if high_conf_evaluated > 0 else 50.0

        # Confidence calibration
        if high_conf_evaluated >= 3:
            if high_conf_accuracy < 40:
                calibration = "overconfident"
            elif high_conf_accuracy > 70:
                calibration = "well-calibrated"
            else:
                calibration = "moderately calibrated"
        else:
            calibration = "insufficient data"

        # Most shilled tokens
        most_shilled = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Token performance breakdown
        token_performance = self._calculate_token_performance(calls)

        # Calculate overall accuracy score
        # Weighted: 60% hit rate, 20% calibration, 20% risk-adjusted returns
        accuracy_score = self._calculate_accuracy_score(
            hit_rate, high_conf_accuracy, avg_gain, avg_loss, total_calls
        )

        return PredictionReport(
            calls=calls,
            accuracy_score=accuracy_score,
            total_calls=total_calls,
            correct_calls=correct_calls,
            incorrect_calls=incorrect_calls,
            pending_calls=pending_calls,
            hit_rate=hit_rate,
            avg_gain_on_correct=avg_gain,
            avg_loss_on_incorrect=avg_loss,
            most_shilled_tokens=most_shilled,
            high_confidence_accuracy=high_conf_accuracy,
            confidence_calibration=calibration,
            token_performance=token_performance
        )

    def _classify_call(self, text: str) -> CallType:
        """Classify tweet as bullish, bearish, or neutral."""
        bullish_score = sum(1 for p in self.bullish_patterns if p.search(text))
        bearish_score = sum(1 for p in self.bearish_patterns if p.search(text))

        if bullish_score > bearish_score:
            return CallType.BULLISH
        elif bearish_score > bullish_score:
            return CallType.BEARISH
        else:
            return CallType.NEUTRAL

    def _get_confidence_words(self, text: str) -> List[str]:
        """Extract high-confidence words from text."""
        words = []
        for pattern in self.high_confidence_patterns:
            matches = pattern.findall(text)
            words.extend(matches)
        return words[:3]

    def _calculate_days_since(self, timestamp: str) -> int:
        """Calculate days since the tweet."""
        try:
            tweet_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            days = (datetime.now(tweet_date.tzinfo) - tweet_date).days
            return max(0, days)
        except (ValueError, TypeError):
            return 30  # Default

    async def _evaluate_calls(self, calls: List[TokenCall]) -> None:
        """
        Evaluate call outcomes by fetching prices.

        Uses rate limiting to respect CoinGecko free tier (10-30 calls/min).
        """
        # Group calls by token to minimize API calls
        tokens_to_fetch = set()
        for call in calls:
            if call.token_id and call.days_since_call >= self.MINIMUM_DAYS_FOR_OUTCOME:
                tokens_to_fetch.add(call.token_id)

        if not tokens_to_fetch:
            return

        # Fetch current prices for all tokens (batch request)
        current_prices = await self._fetch_current_prices(list(tokens_to_fetch))

        # Evaluate each call
        for call in calls:
            if not call.token_id:
                call.outcome = CallOutcome.UNKNOWN
                continue

            if call.days_since_call < self.MINIMUM_DAYS_FOR_OUTCOME:
                call.outcome = CallOutcome.PENDING
                continue

            # Get current price
            current_price = current_prices.get(call.token_id)
            if not current_price:
                call.outcome = CallOutcome.UNKNOWN
                continue

            call.current_price = current_price

            # For historical price, we'd need CoinGecko Pro or estimate
            # For now, use a simplified approach: assume 30-day change
            # In production, you'd use /coins/{id}/market_chart for historical
            historical_price = await self._estimate_historical_price(
                call.token_id, call.days_since_call
            )

            if historical_price:
                call.price_at_call = historical_price
                call.price_change_pct = ((current_price - historical_price) / historical_price) * 100

                # Determine outcome
                if call.call_type == CallType.BULLISH:
                    if call.price_change_pct >= self.BULLISH_CORRECT_THRESHOLD:
                        call.outcome = CallOutcome.CORRECT
                    elif call.price_change_pct <= -self.BULLISH_CORRECT_THRESHOLD:
                        call.outcome = CallOutcome.INCORRECT
                    else:
                        call.outcome = CallOutcome.PENDING  # Too close to call
                elif call.call_type == CallType.BEARISH:
                    if call.price_change_pct <= self.BEARISH_CORRECT_THRESHOLD:
                        call.outcome = CallOutcome.CORRECT
                    elif call.price_change_pct >= -self.BEARISH_CORRECT_THRESHOLD:
                        call.outcome = CallOutcome.INCORRECT
                    else:
                        call.outcome = CallOutcome.PENDING
            else:
                call.outcome = CallOutcome.UNKNOWN

    async def _fetch_current_prices(self, token_ids: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple tokens."""
        if not token_ids:
            return {}

        prices = {}

        # Batch into groups of 50 (CoinGecko limit)
        for i in range(0, len(token_ids), 50):
            batch = token_ids[i:i+50]
            ids_param = ','.join(batch)

            try:
                async with httpx.AsyncClient() as client:
                    url = f"{self.COINGECKO_API}/simple/price"
                    params = {
                        'ids': ids_param,
                        'vs_currencies': 'usd'
                    }
                    if self.api_key:
                        params['x_cg_pro_api_key'] = self.api_key

                    response = await client.get(url, params=params, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        for token_id, price_data in data.items():
                            if 'usd' in price_data:
                                prices[token_id] = price_data['usd']

                    # Rate limiting: wait between batches
                    await asyncio.sleep(1.5)

            except Exception:
                pass  # Silently continue on API errors

        return prices

    async def _estimate_historical_price(
        self,
        token_id: str,
        days_ago: int
    ) -> Optional[float]:
        """
        Estimate historical price.

        Note: Full historical data requires CoinGecko Pro.
        This uses market chart endpoint for approximation.
        """
        # Check cache first
        cache_key = f"{token_id}_{days_ago}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            async with httpx.AsyncClient() as client:
                # Use market_chart for historical data (limited on free tier)
                url = f"{self.COINGECKO_API}/coins/{token_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': min(days_ago + 1, 90)  # Free tier limited to 90 days
                }
                if self.api_key:
                    params['x_cg_pro_api_key'] = self.api_key

                response = await client.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    prices = data.get('prices', [])

                    if prices:
                        # Get price closest to days_ago
                        target_time = (datetime.now() - timedelta(days=days_ago)).timestamp() * 1000

                        closest_price = None
                        min_diff = float('inf')

                        for timestamp, price in prices:
                            diff = abs(timestamp - target_time)
                            if diff < min_diff:
                                min_diff = diff
                                closest_price = price

                        if closest_price:
                            self._price_cache[cache_key] = closest_price
                            return closest_price

                # Rate limiting
                await asyncio.sleep(1.5)

        except Exception:
            pass

        return None

    def _calculate_token_performance(
        self,
        calls: List[TokenCall]
    ) -> Dict[str, Dict]:
        """Calculate performance breakdown by token."""
        performance = {}

        for call in calls:
            token = call.token_symbol
            if token not in performance:
                performance[token] = {
                    'total_calls': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'avg_price_change': 0.0,
                    'price_changes': []
                }

            performance[token]['total_calls'] += 1

            if call.outcome == CallOutcome.CORRECT:
                performance[token]['correct'] += 1
            elif call.outcome == CallOutcome.INCORRECT:
                performance[token]['incorrect'] += 1

            if call.price_change_pct is not None:
                performance[token]['price_changes'].append(call.price_change_pct)

        # Calculate averages
        for token, data in performance.items():
            if data['price_changes']:
                data['avg_price_change'] = sum(data['price_changes']) / len(data['price_changes'])
            del data['price_changes']  # Remove raw data

        return performance

    def _calculate_accuracy_score(
        self,
        hit_rate: float,
        high_conf_accuracy: float,
        avg_gain: float,
        avg_loss: float,
        total_calls: int
    ) -> float:
        """
        Calculate overall accuracy score (0-100).

        Factors:
        - Base: hit rate (60% weight)
        - High confidence calibration (20% weight)
        - Risk-adjusted returns (20% weight)
        """
        # Base from hit rate
        score = hit_rate * 0.6

        # High confidence calibration bonus/penalty
        if high_conf_accuracy >= 60:
            score += 20
        elif high_conf_accuracy >= 40:
            score += 10
        else:
            score += 0  # Overconfident penalty

        # Risk-adjusted returns
        if avg_gain > 0:
            risk_ratio = abs(avg_gain) / max(abs(avg_loss), 1) if avg_loss else 2.0
            if risk_ratio >= 2:
                score += 20
            elif risk_ratio >= 1:
                score += 10
            else:
                score += 5

        # Confidence penalty for low sample size
        if total_calls < 5:
            score *= 0.8  # 20% penalty
        elif total_calls < 10:
            score *= 0.9  # 10% penalty

        return min(100.0, max(0.0, score))

    def generate_summary(self, report: PredictionReport) -> str:
        """Generate human-readable summary."""
        if report.total_calls == 0:
            return "No token predictions found to evaluate."

        parts = []

        # Main finding
        if report.hit_rate >= 60:
            parts.append(f"Above-average prediction accuracy ({report.hit_rate:.0f}% hit rate).")
        elif report.hit_rate >= 40:
            parts.append(f"Mixed prediction accuracy ({report.hit_rate:.0f}% hit rate).")
        else:
            parts.append(f"Below-average predictions ({report.hit_rate:.0f}% hit rate).")

        # Confidence calibration
        if report.confidence_calibration == "overconfident":
            parts.append("Often overconfident on 'guaranteed' calls.")
        elif report.confidence_calibration == "well-calibrated":
            parts.append("Well-calibrated confidence on predictions.")

        # Top shilled token
        if report.most_shilled_tokens:
            top_token = report.most_shilled_tokens[0]
            parts.append(f"Most mentioned: ${top_token[0]} ({top_token[1]} times).")

        return " ".join(parts)


# Synchronous wrapper for non-async contexts
def analyze_predictions_sync(tweets: List[dict]) -> PredictionReport:
    """Synchronous wrapper for analyze method."""
    tracker = PredictionTracker()
    return asyncio.run(tracker.analyze(tweets))
