"""
Contradiction Analyzer

Detects contradictory statements and actions in a KOL's tweet history.
Finds instances where they say one thing and do/say the opposite.

Enhanced with sentence embeddings (All-MiniLM-L6-v2) for semantic similarity
detection of contradictions that may not match regex patterns.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    """A detected contradiction between two statements."""
    severity: str  # "high", "medium", "low"
    original_tweet: Dict[str, Any]
    contradicting_tweet: Dict[str, Any]
    category: str  # "holding", "promotion", "sentiment", "advice"
    description: str
    time_between_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "time_between_days": self.time_between_days,
            "original": {
                "text": (self.original_tweet.get("text") or "")[:280],
                "date": self.original_tweet.get("timestamp") or "",
                "id": str(self.original_tweet.get("id") or "")
            },
            "contradicting": {
                "text": (self.contradicting_tweet.get("text") or "")[:280],
                "date": self.contradicting_tweet.get("timestamp") or "",
                "id": str(self.contradicting_tweet.get("id") or "")
            }
        }


@dataclass
class ContradictionAnalysis:
    """Results from contradiction analysis."""
    contradiction_count: int = 0
    contradictions: List[Contradiction] = field(default_factory=list)
    bs_score: float = 0.0  # 0 = trustworthy, 100 = full of BS

    # Breakdown
    holding_contradictions: int = 0
    promotion_contradictions: int = 0
    sentiment_contradictions: int = 0
    advice_contradictions: int = 0

    # Semantic contradictions (ML-detected)
    semantic_contradictions: int = 0
    ml_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "contradiction_count": self.contradiction_count,
            "bs_score": self.bs_score,
            "holding_contradictions": self.holding_contradictions,
            "promotion_contradictions": self.promotion_contradictions,
            "sentiment_contradictions": self.sentiment_contradictions,
            "advice_contradictions": self.advice_contradictions,
            "contradictions": [c.to_dict() for c in self.contradictions[:10]]  # Top 10
        }

        if self.ml_available:
            result["semantic_contradictions"] = self.semantic_contradictions

        return result


class ContradictionAnalyzer:
    """Analyzes tweets for contradictory statements."""

    # Patterns for holding/selling contradictions
    HOLDING_PATTERNS = [
        (r'\b(hodl|holding|hold|never sell|diamond hands?|long term)\b.*\$([A-Z]{2,10})\b', 'hold'),
        (r'\$([A-Z]{2,10})\b.*\b(hodl|holding|hold|never sell|diamond hands?|long term)\b', 'hold'),
        (r'\b(sold|selling|dumped|exited|took profits?|closed)\b.*\$([A-Z]{2,10})\b', 'sell'),
        (r'\$([A-Z]{2,10})\b.*\b(sold|selling|dumped|exited|took profits?|closed)\b', 'sell'),
    ]

    # Patterns for bullish/bearish sentiment
    SENTIMENT_PATTERNS = [
        (r'\$([A-Z]{2,10})\b.*\b(bullish|moon|100x|generational|buying|accumulating|loading)\b', 'bullish'),
        (r'\b(bullish|moon|100x|generational|buying|accumulating|loading)\b.*\$([A-Z]{2,10})\b', 'bullish'),
        (r'\$([A-Z]{2,10})\b.*\b(bearish|dump|crash|avoid|stay away|selling|dead)\b', 'bearish'),
        (r'\b(bearish|dump|crash|avoid|stay away|selling|dead)\b.*\$([A-Z]{2,10})\b', 'bearish'),
    ]

    # Patterns for promotion honesty
    PROMOTION_PATTERNS = [
        (r'\b(not? paid|no promotion|genuine|organic|not sponsored|i don.t take money)\b', 'claims_organic'),
        (r'\b(#ad|#sponsored|paid partnership|sponsored|promotion)\b', 'admits_paid'),
        (r'\b(check out|use my code|sign up|referral|airdrop)\b', 'promoting'),
    ]

    # Patterns for financial advice contradictions
    ADVICE_PATTERNS = [
        (r'\b(nfa|not financial advice|dyor|do your own research|not a financial advisor)\b', 'disclaimer'),
        (r'\b(you should (buy|sell)|must (buy|sell)|guaranteed|will (100x|moon)|free money)\b', 'direct_advice'),
        (r'\b(buy now|sell now|last chance|don.t miss|hurry)\b', 'urgency'),
    ]

    # Semantic contradiction threshold
    SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # High similarity needed
    SEMANTIC_SENTIMENT_DIFF_THRESHOLD = 0.5  # Sentiment must differ significantly

    def __init__(self, use_ml: bool = True):
        """
        Initialize the analyzer.

        Args:
            use_ml: Whether to use ML models for semantic contradiction detection
        """
        # Compile patterns
        self.holding_re = [(re.compile(p, re.IGNORECASE), action) for p, action in self.HOLDING_PATTERNS]
        self.sentiment_re = [(re.compile(p, re.IGNORECASE), sentiment) for p, sentiment in self.SENTIMENT_PATTERNS]
        self.promotion_re = [(re.compile(p, re.IGNORECASE), ptype) for p, ptype in self.PROMOTION_PATTERNS]
        self.advice_re = [(re.compile(p, re.IGNORECASE), atype) for p, atype in self.ADVICE_PATTERNS]

        self.use_ml = use_ml
        self._ml_available = None
        self._embeddings_cache = None

    def _check_ml_available(self) -> bool:
        """Check if ML models are available."""
        if self._ml_available is None:
            try:
                from .ml_models import is_model_available
                self._ml_available = (
                    is_model_available('embeddings') and
                    is_model_available('sentiment')
                )
            except ImportError:
                self._ml_available = False
        return self._ml_available

    def _find_semantic_contradictions(
        self,
        tweets: List[Dict[str, Any]]
    ) -> List[Contradiction]:
        """
        Find semantic contradictions using embeddings and sentiment.

        This detects contradictions that regex patterns might miss by:
        1. Finding tweets about similar topics (high embedding similarity)
        2. Checking if they have opposite sentiments

        Args:
            tweets: List of tweet dicts

        Returns:
            List of detected contradictions
        """
        if not self.use_ml or not self._check_ml_available():
            return []

        try:
            from .ml_models import get_embeddings, analyze_sentiment_batch
            import numpy as np
            from scipy.spatial.distance import cosine

            # Get texts and filter
            texts = [t.get('text', '') for t in tweets if t.get('text', '').strip()]
            if len(texts) < 5:
                return []

            # Get embeddings
            embeddings = get_embeddings(texts)
            if embeddings is None:
                return []

            # Get sentiments
            sentiments = analyze_sentiment_batch(texts)

            contradictions = []
            checked_pairs = set()

            # Compare tweets for semantic contradictions
            for i in range(len(texts)):
                if sentiments[i] is None:
                    continue

                for j in range(i + 1, len(texts)):
                    if sentiments[j] is None:
                        continue

                    # Skip if already checked
                    pair_key = (min(i, j), max(i, j))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)

                    # Calculate similarity
                    try:
                        similarity = 1 - cosine(embeddings[i], embeddings[j])
                    except Exception:
                        continue

                    # If tweets are semantically similar (same topic)
                    if similarity < self.SEMANTIC_SIMILARITY_THRESHOLD:
                        continue

                    # Check if sentiments are opposite
                    sent_i = sentiments[i]
                    sent_j = sentiments[j]

                    # Calculate sentiment difference
                    pos_diff = abs(sent_i['positive'] - sent_j['positive'])
                    neg_diff = abs(sent_i['negative'] - sent_j['negative'])

                    # One positive, one negative
                    is_contradiction = (
                        (sent_i['positive'] > 0.5 and sent_j['negative'] > 0.5) or
                        (sent_i['negative'] > 0.5 and sent_j['positive'] > 0.5) or
                        (pos_diff > self.SEMANTIC_SENTIMENT_DIFF_THRESHOLD and
                         neg_diff > self.SEMANTIC_SENTIMENT_DIFF_THRESHOLD)
                    )

                    if is_contradiction:
                        # Find the original tweet dicts
                        tweet_i = tweets[i]
                        tweet_j = tweets[j]

                        days_between = abs(self._days_between(tweet_i, tweet_j))

                        # Determine severity based on time gap
                        if days_between <= 7:
                            severity = "high"
                        elif days_between <= 30:
                            severity = "medium"
                        else:
                            severity = "low"

                        contradiction = Contradiction(
                            severity=severity,
                            original_tweet=tweet_i,
                            contradicting_tweet=tweet_j,
                            category="semantic",
                            description=f"Semantically similar statements with opposing sentiments (similarity: {similarity:.2f})",
                            time_between_days=days_between
                        )
                        contradictions.append(contradiction)

            # Limit to top contradictions to avoid overwhelming results
            return contradictions[:10]

        except Exception as e:
            logger.warning(f"Semantic contradiction detection failed: {e}")
            return []

    def analyze(self, tweets: List[Dict[str, Any]], username: str = "") -> ContradictionAnalysis:
        """Analyze tweets for contradictions."""
        result = ContradictionAnalysis()

        if len(tweets) < 5:
            return result

        # Sort tweets by timestamp (oldest first for analysis)
        sorted_tweets = sorted(
            tweets,
            key=lambda t: t.get('timestamp', '') or '',
            reverse=False
        )

        # Track statements by token/topic
        token_holdings: Dict[str, List[Tuple[Dict, str]]] = {}  # token -> [(tweet, action)]
        token_sentiments: Dict[str, List[Tuple[Dict, str]]] = {}  # token -> [(tweet, sentiment)]
        promotion_claims: List[Tuple[Dict, str]] = []  # [(tweet, claim_type)]
        advice_statements: List[Tuple[Dict, str]] = []  # [(tweet, advice_type)]

        # Extract all statements
        for tweet in sorted_tweets:
            text = tweet.get('text') or ''
            if not text:
                continue

            # Check holding/selling patterns
            for pattern, action in self.holding_re:
                matches = pattern.findall(text)
                for match in matches:
                    # Extract token (could be in different group positions)
                    token = None
                    if isinstance(match, tuple):
                        for part in match:
                            if part and re.match(r'^[A-Z]{2,10}$', part.upper()):
                                token = part.upper()
                                break
                    if token:
                        if token not in token_holdings:
                            token_holdings[token] = []
                        token_holdings[token].append((tweet, action))

            # Check sentiment patterns
            for pattern, sentiment in self.sentiment_re:
                matches = pattern.findall(text)
                for match in matches:
                    token = None
                    if isinstance(match, tuple):
                        for part in match:
                            if part and re.match(r'^[A-Z]{2,10}$', part.upper()):
                                token = part.upper()
                                break
                    if token:
                        if token not in token_sentiments:
                            token_sentiments[token] = []
                        token_sentiments[token].append((tweet, sentiment))

            # Check promotion patterns
            for pattern, ptype in self.promotion_re:
                if pattern.search(text):
                    promotion_claims.append((tweet, ptype))

            # Check advice patterns
            for pattern, atype in self.advice_re:
                if pattern.search(text):
                    advice_statements.append((tweet, atype))

        # Find holding contradictions (said hold, then sold quickly)
        for token, statements in token_holdings.items():
            holds = [(t, a) for t, a in statements if a == 'hold']
            sells = [(t, a) for t, a in statements if a == 'sell']

            for hold_tweet, _ in holds:
                for sell_tweet, _ in sells:
                    days_between = self._days_between(hold_tweet, sell_tweet)
                    # If they sold within 30 days of saying "holding long term"
                    if 0 < days_between <= 30:
                        severity = "high" if days_between <= 7 else "medium"
                        contradiction = Contradiction(
                            severity=severity,
                            original_tweet=hold_tweet,
                            contradicting_tweet=sell_tweet,
                            category="holding",
                            description=f"Claimed holding ${token} long term, sold within {days_between} days",
                            time_between_days=days_between
                        )
                        result.contradictions.append(contradiction)
                        result.holding_contradictions += 1

        # Find sentiment contradictions (bullish then bearish on same token)
        for token, statements in token_sentiments.items():
            bullish = [(t, s) for t, s in statements if s == 'bullish']
            bearish = [(t, s) for t, s in statements if s == 'bearish']

            for bull_tweet, _ in bullish:
                for bear_tweet, _ in bearish:
                    days_between = abs(self._days_between(bull_tweet, bear_tweet))
                    # Flip-flopping within 14 days is suspicious
                    if days_between <= 14:
                        severity = "high" if days_between <= 3 else "medium"
                        contradiction = Contradiction(
                            severity=severity,
                            original_tweet=bull_tweet,
                            contradicting_tweet=bear_tweet,
                            category="sentiment",
                            description=f"Flipped from bullish to bearish on ${token} within {days_between} days",
                            time_between_days=days_between
                        )
                        result.contradictions.append(contradiction)
                        result.sentiment_contradictions += 1

        # Find promotion contradictions
        organic_claims = [t for t, p in promotion_claims if p == 'claims_organic']
        paid_admits = [t for t, p in promotion_claims if p == 'admits_paid']
        promotions = [t for t, p in promotion_claims if p == 'promoting']

        for organic in organic_claims:
            # Check if they later admitted to paid promotions
            for paid in paid_admits:
                days_between = self._days_between(organic, paid)
                if -90 <= days_between <= 90:  # Within 90 days either way
                    contradiction = Contradiction(
                        severity="high",
                        original_tweet=organic,
                        contradicting_tweet=paid,
                        category="promotion",
                        description="Claimed to not take money for promotions, but has sponsored content",
                        time_between_days=abs(days_between)
                    )
                    result.contradictions.append(contradiction)
                    result.promotion_contradictions += 1
                    break

        # Find advice contradictions (NFA but gives direct advice)
        disclaimers = [t for t, a in advice_statements if a == 'disclaimer']
        direct_advice = [t for t, a in advice_statements if a in ['direct_advice', 'urgency']]

        for disclaimer in disclaimers:
            for advice in direct_advice:
                days_between = abs(self._days_between(disclaimer, advice))
                if days_between <= 7:  # Within a week
                    contradiction = Contradiction(
                        severity="medium",
                        original_tweet=disclaimer,
                        contradicting_tweet=advice,
                        category="advice",
                        description="Says 'not financial advice' but gives direct trading instructions",
                        time_between_days=days_between
                    )
                    result.contradictions.append(contradiction)
                    result.advice_contradictions += 1

        # Find semantic contradictions using ML
        semantic_contradictions = self._find_semantic_contradictions(sorted_tweets)
        if semantic_contradictions:
            result.ml_available = True
            result.semantic_contradictions = len(semantic_contradictions)
            # Add semantic contradictions to the list
            for sc in semantic_contradictions:
                # Avoid duplicates (if regex already caught it)
                is_duplicate = any(
                    c.original_tweet.get('id') == sc.original_tweet.get('id') and
                    c.contradicting_tweet.get('id') == sc.contradicting_tweet.get('id')
                    for c in result.contradictions
                )
                if not is_duplicate:
                    result.contradictions.append(sc)

        # Sort contradictions by severity and recency
        severity_order = {"high": 0, "medium": 1, "low": 2}
        result.contradictions.sort(
            key=lambda c: (severity_order.get(c.severity, 2), c.time_between_days)
        )

        # Calculate total and BS score
        result.contradiction_count = len(result.contradictions)

        # BS Score: based on contradiction density and severity
        if len(tweets) > 0:
            high_count = sum(1 for c in result.contradictions if c.severity == "high")
            medium_count = sum(1 for c in result.contradictions if c.severity == "medium")
            low_count = sum(1 for c in result.contradictions if c.severity == "low")

            # Weight: high=10, medium=5, low=2
            weighted_score = high_count * 10 + medium_count * 5 + low_count * 2

            # Normalize to 0-100, cap at 100
            # Assume 20 weighted points = 100 BS score for high contradiction density
            result.bs_score = min(100, (weighted_score / max(len(tweets) / 100, 1)) * 5)

        return result

    def _days_between(self, tweet1: Dict, tweet2: Dict) -> int:
        """Calculate days between two tweets (positive if tweet2 is later)."""
        try:
            ts1 = tweet1.get('timestamp', '')
            ts2 = tweet2.get('timestamp', '')

            if not ts1 or not ts2:
                return 999  # Unknown

            # Parse timestamps
            dt1 = self._parse_timestamp(ts1)
            dt2 = self._parse_timestamp(ts2)

            if dt1 and dt2:
                delta = dt2 - dt1
                return delta.days
        except Exception:
            pass
        return 999

    def _parse_timestamp(self, ts: str) -> Optional[datetime]:
        """Parse various timestamp formats."""
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts[:26], fmt)
            except (ValueError, TypeError):
                continue
        return None
