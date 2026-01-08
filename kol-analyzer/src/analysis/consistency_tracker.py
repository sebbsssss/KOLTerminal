"""
Consistency Tracker - Track position changes on tokens/topics for flip detection.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Sentiment(Enum):
    """Sentiment classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PositionChange:
    """Represents a change in position on a topic/token."""
    topic: str  # e.g., "$BTC", "$SOL", "MARKET"
    old_position: str  # "bullish" or "bearish"
    new_position: str
    old_tweet_id: str
    old_tweet_text: str
    old_tweet_date: str
    new_tweet_id: str
    new_tweet_text: str
    new_tweet_date: str
    severity: str  # "minor", "moderate", "major", "flip"
    self_acknowledged: bool = False  # Did they admit the flip?
    days_between: int = 0

    def to_dict(self) -> dict:
        return {
            'topic': self.topic,
            'old_position': self.old_position,
            'new_position': self.new_position,
            'old_tweet_id': self.old_tweet_id,
            'old_tweet_text': self.old_tweet_text[:100] + '...' if len(self.old_tweet_text) > 100 else self.old_tweet_text,
            'old_tweet_date': self.old_tweet_date,
            'new_tweet_id': self.new_tweet_id,
            'new_tweet_text': self.new_tweet_text[:100] + '...' if len(self.new_tweet_text) > 100 else self.new_tweet_text,
            'new_tweet_date': self.new_tweet_date,
            'severity': self.severity,
            'self_acknowledged': self.self_acknowledged,
            'days_between': self.days_between
        }


@dataclass
class ConsistencyReport:
    """Report on position consistency over time."""
    position_changes: List[PositionChange] = field(default_factory=list)
    consistency_score: float = 100.0  # 0-100
    flip_count: int = 0
    major_flip_count: int = 0
    analysis_period_days: int = 0
    topics_tracked: List[str] = field(default_factory=list)
    acknowledged_flips: int = 0
    unacknowledged_flips: int = 0
    topic_positions: Dict[str, str] = field(default_factory=dict)  # Current positions

    def to_dict(self) -> dict:
        return {
            'position_changes': [pc.to_dict() for pc in self.position_changes],
            'consistency_score': round(self.consistency_score, 1),
            'flip_count': self.flip_count,
            'major_flip_count': self.major_flip_count,
            'analysis_period_days': self.analysis_period_days,
            'topics_tracked': self.topics_tracked,
            'acknowledged_flips': self.acknowledged_flips,
            'unacknowledged_flips': self.unacknowledged_flips,
            'topic_positions': self.topic_positions
        }


class ConsistencyTracker:
    """
    Tracks position changes on tokens/topics over time.

    Detection logic:
    - Extract $TICKER mentions using regex
    - Classify sentiment using keyword lists
    - Track sentiment per ticker over time
    - Detect reversals (bullish→bearish or vice versa)
    - Check for self-acknowledgment (transparency is good)
    """

    # Ticker extraction pattern
    TICKER_PATTERN = r'\$([A-Z]{2,10})\b'

    # Sentiment keywords
    BULLISH_KEYWORDS = [
        'bullish', 'moon', 'pump', 'buy', 'accumulate', 'conviction',
        'hodl', 'hold', '100x', 'alpha', 'long', 'undervalued', 'gem',
        'loading', 'adding', 'ape', 'send', 'rip', 'breakout', 'reversal',
        'bottomed', 'bottom', 'support', 'strong', 'bullrun', 'rally'
    ]

    BEARISH_KEYWORDS = [
        'bearish', 'dump', 'sell', 'exit', 'short', 'overvalued', 'scam',
        'rug', 'dead', 'crash', 'collapse', 'avoid', 'warning', 'careful',
        'resistance', 'weak', 'top', 'topped', 'distribution', 'selling',
        'reducing', 'trimming', 'derisk', 'de-risk'
    ]

    # Self-acknowledgment patterns
    ACKNOWLEDGMENT_PATTERNS = [
        r'\b(flipping|flip)\b',
        r'\b(180|one-eighty)\b',
        r'\b(i was wrong|was wrong)\b',
        r'\b(changing my|changed my)\b',
        r'\b(admittedly|i admit)\b',
        r'\b(reconsidering|reconsider)\b',
        r'\b(adjusting|adjusted).*(thesis|view|stance)\b',
        r'\b(updating|updated).*(view|outlook)\b',
    ]

    # Penalties for flips
    FLIP_PENALTY_MAJOR = 15.0
    FLIP_PENALTY_MODERATE = 8.0
    FLIP_PENALTY_MINOR = 3.0
    SELF_ACKNOWLEDGED_REDUCTION = 0.5  # 50% reduction for transparency

    def __init__(self):
        self.bullish_pattern = re.compile(
            r'\b(' + '|'.join(self.BULLISH_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.bearish_pattern = re.compile(
            r'\b(' + '|'.join(self.BEARISH_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        self.ticker_pattern = re.compile(self.TICKER_PATTERN)
        self.acknowledgment_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.ACKNOWLEDGMENT_PATTERNS
        ]

    def analyze(self, tweets: List[dict]) -> ConsistencyReport:
        """
        Analyze tweets for position consistency.

        Args:
            tweets: List of tweet dictionaries, sorted by timestamp (newest first ideally)

        Returns:
            ConsistencyReport with analysis results
        """
        if not tweets:
            return ConsistencyReport(
                consistency_score=50.0,
                analysis_period_days=0
            )

        # Sort tweets by timestamp (oldest first for chronological tracking)
        sorted_tweets = sorted(
            tweets,
            key=lambda t: t.get('timestamp', ''),
            reverse=False
        )

        # Track positions over time: {ticker: [(sentiment, tweet_id, timestamp, text)]}
        position_history: Dict[str, List[Tuple[Sentiment, str, str, str]]] = {}

        for tweet in sorted_tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Extract tickers
            tickers = self.ticker_pattern.findall(text)

            # Also check for general market sentiment
            if any(word in text.lower() for word in ['market', 'crypto', 'btc', 'bitcoin']):
                tickers.append('MARKET')

            # Classify sentiment
            sentiment = self._classify_sentiment(text)

            if sentiment == Sentiment.NEUTRAL:
                continue

            # Record position for each ticker mentioned
            for ticker in set(tickers):
                ticker = ticker.upper()
                if ticker not in position_history:
                    position_history[ticker] = []

                position_history[ticker].append((
                    sentiment,
                    tweet_id,
                    timestamp,
                    text
                ))

        # Detect position changes
        position_changes = []
        total_penalty = 0.0

        for ticker, history in position_history.items():
            if len(history) < 2:
                continue

            # Look for sentiment flips
            for i in range(1, len(history)):
                prev = history[i - 1]
                curr = history[i]

                if prev[0] != curr[0]:  # Sentiment changed
                    # Calculate days between
                    days_between = self._calculate_days_between(prev[2], curr[2])

                    # Determine severity
                    severity = self._determine_severity(
                        prev[0], curr[0], days_between
                    )

                    # Check for self-acknowledgment
                    acknowledged = self._check_acknowledgment(curr[3])

                    change = PositionChange(
                        topic=f"${ticker}",
                        old_position=prev[0].value,
                        new_position=curr[0].value,
                        old_tweet_id=prev[1],
                        old_tweet_text=prev[3],
                        old_tweet_date=prev[2],
                        new_tweet_id=curr[1],
                        new_tweet_text=curr[3],
                        new_tweet_date=curr[2],
                        severity=severity,
                        self_acknowledged=acknowledged,
                        days_between=days_between
                    )

                    position_changes.append(change)

                    # Calculate penalty
                    penalty = self._get_penalty(severity)
                    if acknowledged:
                        penalty *= self.SELF_ACKNOWLEDGED_REDUCTION

                    total_penalty += penalty

        # Calculate analysis period
        analysis_period_days = 0
        if sorted_tweets:
            try:
                first = datetime.fromisoformat(
                    sorted_tweets[0].get('timestamp', '').replace('Z', '+00:00')
                )
                last = datetime.fromisoformat(
                    sorted_tweets[-1].get('timestamp', '').replace('Z', '+00:00')
                )
                analysis_period_days = abs((last - first).days)
            except (ValueError, TypeError):
                analysis_period_days = 90  # Default estimate

        # Calculate final score
        consistency_score = max(0.0, 100.0 - total_penalty)

        # Count flips
        flip_count = len(position_changes)
        major_flip_count = sum(1 for pc in position_changes if pc.severity in ['major', 'flip'])
        acknowledged_flips = sum(1 for pc in position_changes if pc.self_acknowledged)
        unacknowledged_flips = flip_count - acknowledged_flips

        # Current positions
        topic_positions = {}
        for ticker, history in position_history.items():
            if history:
                topic_positions[f"${ticker}"] = history[-1][0].value

        return ConsistencyReport(
            position_changes=position_changes,
            consistency_score=consistency_score,
            flip_count=flip_count,
            major_flip_count=major_flip_count,
            analysis_period_days=analysis_period_days,
            topics_tracked=list(position_history.keys()),
            acknowledged_flips=acknowledged_flips,
            unacknowledged_flips=unacknowledged_flips,
            topic_positions=topic_positions
        )

    def _classify_sentiment(self, text: str) -> Sentiment:
        """Classify the sentiment of a tweet."""
        text_lower = text.lower()

        bullish_matches = len(self.bullish_pattern.findall(text_lower))
        bearish_matches = len(self.bearish_pattern.findall(text_lower))

        if bullish_matches > bearish_matches:
            return Sentiment.BULLISH
        elif bearish_matches > bullish_matches:
            return Sentiment.BEARISH
        else:
            return Sentiment.NEUTRAL

    def _check_acknowledgment(self, text: str) -> bool:
        """Check if a tweet acknowledges a position change."""
        for pattern in self.acknowledgment_patterns:
            if pattern.search(text):
                return True
        return False

    def _calculate_days_between(self, timestamp1: str, timestamp2: str) -> int:
        """Calculate days between two timestamps."""
        try:
            dt1 = datetime.fromisoformat(timestamp1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(timestamp2.replace('Z', '+00:00'))
            return abs((dt2 - dt1).days)
        except (ValueError, TypeError):
            return 30  # Default estimate

    def _determine_severity(
        self,
        old_sentiment: Sentiment,
        new_sentiment: Sentiment,
        days_between: int
    ) -> str:
        """Determine the severity of a position change."""
        # Complete flip (bullish <-> bearish)
        is_complete_flip = (
            (old_sentiment == Sentiment.BULLISH and new_sentiment == Sentiment.BEARISH) or
            (old_sentiment == Sentiment.BEARISH and new_sentiment == Sentiment.BULLISH)
        )

        if is_complete_flip:
            if days_between <= 7:
                return "flip"  # Very quick reversal
            elif days_between <= 30:
                return "major"
            elif days_between <= 90:
                return "moderate"
            else:
                return "minor"  # Over 90 days, positions can reasonably change
        else:
            return "minor"

    def _get_penalty(self, severity: str) -> float:
        """Get the penalty for a position change severity."""
        penalties = {
            "flip": self.FLIP_PENALTY_MAJOR,
            "major": self.FLIP_PENALTY_MAJOR * 0.8,
            "moderate": self.FLIP_PENALTY_MODERATE,
            "minor": self.FLIP_PENALTY_MINOR
        }
        return penalties.get(severity, self.FLIP_PENALTY_MINOR)

    def generate_summary(self, report: ConsistencyReport) -> str:
        """Generate a human-readable summary of consistency analysis."""
        if report.flip_count == 0:
            return f"No position flips detected across {len(report.topics_tracked)} tracked topics over {report.analysis_period_days} days."

        summary_parts = [
            f"Detected {report.flip_count} position change(s) over {report.analysis_period_days} days."
        ]

        if report.major_flip_count > 0:
            summary_parts.append(f"{report.major_flip_count} were major/quick reversals.")

        if report.acknowledged_flips > 0:
            summary_parts.append(
                f"{report.acknowledged_flips} change(s) were transparently acknowledged."
            )

        if report.unacknowledged_flips > 0:
            summary_parts.append(
                f"{report.unacknowledged_flips} change(s) were not acknowledged."
            )

        return " ".join(summary_parts)
