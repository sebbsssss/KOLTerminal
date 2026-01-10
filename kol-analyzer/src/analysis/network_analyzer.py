"""
Network/Reply Pattern Analyzer - Analyze interaction patterns and network behavior.

Detects:
- Reply guy behavior: Ratio of replies to original content
- Target selection: Who do they reply to? (clout chasing?)
- Mutual promotion: Coordinated shilling with specific accounts
- Engagement with criticism: How do they handle pushback?
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter


@dataclass
class InteractionPattern:
    """A detected interaction pattern."""
    account_mentioned: str
    interaction_type: str  # "reply", "quote", "mention", "mutual_shill"
    count: int
    sentiment: str  # "positive", "negative", "neutral"
    example_tweet: Optional[str]

    def to_dict(self) -> dict:
        return {
            'account_mentioned': self.account_mentioned,
            'interaction_type': self.interaction_type,
            'count': self.count,
            'sentiment': self.sentiment,
            'example_tweet': self.example_tweet[:100] if self.example_tweet else None
        }


@dataclass
class NetworkReport:
    """Report on network and interaction patterns."""
    network_score: float = 50.0  # 0-100, higher = healthier network behavior

    # Content type breakdown
    original_tweets: int = 0
    replies: int = 0
    quotes: int = 0
    retweets: int = 0

    # Ratios
    reply_ratio: float = 0.0  # Replies / total
    original_ratio: float = 0.0  # Original / total
    reply_guy_score: float = 0.0  # 0-100, higher = more reply guy behavior

    # Target analysis
    unique_accounts_engaged: int = 0
    top_engaged_accounts: List[Tuple[str, int]] = field(default_factory=list)
    engages_larger_accounts: bool = False
    clout_chasing_score: float = 0.0

    # Mutual promotion
    mutual_mentions: List[Tuple[str, int]] = field(default_factory=list)
    potential_shill_ring: bool = False
    shill_ring_accounts: List[str] = field(default_factory=list)

    # Criticism handling
    responds_to_criticism: bool = False
    defensive_responses: int = 0
    constructive_responses: int = 0
    blocks_critics: bool = False  # Can't detect directly

    # Interaction patterns
    patterns: List[InteractionPattern] = field(default_factory=list)

    patterns_detected: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'network_score': round(self.network_score, 1),
            'original_tweets': self.original_tweets,
            'replies': self.replies,
            'quotes': self.quotes,
            'retweets': self.retweets,
            'reply_ratio': round(self.reply_ratio, 3),
            'original_ratio': round(self.original_ratio, 3),
            'reply_guy_score': round(self.reply_guy_score, 1),
            'unique_accounts_engaged': self.unique_accounts_engaged,
            'top_engaged_accounts': self.top_engaged_accounts[:10],
            'engages_larger_accounts': self.engages_larger_accounts,
            'clout_chasing_score': round(self.clout_chasing_score, 1),
            'mutual_mentions': self.mutual_mentions[:10],
            'potential_shill_ring': self.potential_shill_ring,
            'shill_ring_accounts': self.shill_ring_accounts[:5],
            'responds_to_criticism': self.responds_to_criticism,
            'defensive_responses': self.defensive_responses,
            'constructive_responses': self.constructive_responses,
            'patterns': [p.to_dict() for p in self.patterns[:10]],
            'patterns_detected': self.patterns_detected,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class NetworkAnalyzer:
    """
    Analyzes network interactions and reply patterns.

    Key questions:
    1. Are they mostly replying to others or creating original content?
    2. Do they target larger accounts (clout chasing)?
    3. Is there a mutual promotion network (shill ring)?
    4. How do they handle criticism?
    """

    # Mention pattern
    MENTION_PATTERN = r'@([A-Za-z0-9_]+)'

    # Known large accounts (simplified - would be expanded)
    LARGE_ACCOUNTS = {
        'elonmusk', 'clovisvd', 'vitalikbuterin', 'cz_binance', 'satloshipen',
        'cobie', 'hsaka_', 'zhusu', 'lightcrypto', 'pentosh1',
        'ansem', 'blknoiz06', 'cryptokaleo', 'cryptoyieldinfo',
    }

    # Defensive response patterns
    DEFENSIVE_PATTERNS = [
        r'\b(you\'re wrong|you don\'t understand|clearly you)\b',
        r'\b(blocked|muted|bye|have fun staying poor)\b',
        r'\b(ngmi|hfsp|cope|seethe)\b',
        r'\b(ratio|L take|bad take)\b',
        r'\b(clown|idiot|stupid|dumb)\b',
    ]

    # Constructive response patterns
    CONSTRUCTIVE_PATTERNS = [
        r'\b(good point|fair enough|you\'re right)\b',
        r'\b(let me explain|here\'s why|actually)\b',
        r'\b(thanks for|appreciate|valid)\b',
        r'\b(i see your point|makes sense)\b',
        r'\b(agree to disagree|fair criticism)\b',
    ]

    # Shill patterns
    SHILL_PATTERNS = [
        r'\b(check out|go follow|must follow)\b',
        r'\b(alpha from|best account|goat)\b',
        r'\b(this guy|this account|legend)\b',
        r'\b(underrated|undervalued|sleeping on)\b',
    ]

    # Clout chasing patterns
    CLOUT_PATTERNS = [
        r'^@\w+\s+(this|so true|facts|exactly|real)',
        r'@\w+.*\b(please|plz|pls)\b.*(follow|notice|see)',
        r'^@\w+\s+(gm|gn|wagmi)',
    ]

    def __init__(self):
        self.mention_pattern = re.compile(self.MENTION_PATTERN)
        self.defensive_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEFENSIVE_PATTERNS]
        self.constructive_patterns = [re.compile(p, re.IGNORECASE) for p in self.CONSTRUCTIVE_PATTERNS]
        self.shill_patterns = [re.compile(p, re.IGNORECASE) for p in self.SHILL_PATTERNS]
        self.clout_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLOUT_PATTERNS]

    def analyze(self, tweets: List[dict]) -> NetworkReport:
        """Analyze network and interaction patterns."""
        if not tweets:
            return NetworkReport()

        # Categorize tweets
        original = 0
        replies = 0
        quotes = 0

        mention_counts: Counter = Counter()
        reply_targets: Counter = Counter()
        shill_targets: Counter = Counter()

        defensive = 0
        constructive = 0

        for tweet in tweets:
            text = tweet.get('text', '')
            is_reply = tweet.get('is_reply', False) or tweet.get('reply_to')
            is_quote = tweet.get('is_quote', False)

            # Categorize tweet type
            if is_reply:
                replies += 1
                reply_to = tweet.get('reply_to', '')
                if reply_to:
                    reply_targets[reply_to.lower()] += 1
            elif is_quote:
                quotes += 1
            else:
                original += 1

            # Extract all mentions
            mentions = self.mention_pattern.findall(text)
            for mention in mentions:
                mention_counts[mention.lower()] += 1

            # Check for shill patterns with mentions
            if any(p.search(text) for p in self.shill_patterns):
                for mention in mentions:
                    shill_targets[mention.lower()] += 1

            # Check response style (if it's a reply)
            if is_reply:
                if any(p.search(text) for p in self.defensive_patterns):
                    defensive += 1
                if any(p.search(text) for p in self.constructive_patterns):
                    constructive += 1

        total = original + replies + quotes
        if total == 0:
            return NetworkReport()

        # Calculate ratios
        reply_ratio = replies / total
        original_ratio = original / total

        # Reply guy score (high replies, targeting large accounts)
        reply_guy_score = self._calculate_reply_guy_score(
            reply_ratio, reply_targets, total
        )

        # Clout chasing analysis
        large_account_replies = sum(
            count for acc, count in reply_targets.items()
            if acc in self.LARGE_ACCOUNTS
        )
        engages_larger = large_account_replies > replies * 0.3 if replies > 5 else False
        clout_score = self._calculate_clout_score(
            large_account_replies, replies, reply_targets
        )

        # Mutual promotion / shill ring detection
        mutual_mentions = [(acc, count) for acc, count in shill_targets.most_common(20) if count >= 3]
        potential_ring = len([m for m in mutual_mentions if m[1] >= 5]) >= 2
        ring_accounts = [acc for acc, count in mutual_mentions if count >= 5]

        # Build interaction patterns
        patterns = self._build_interaction_patterns(
            reply_targets, mention_counts, shill_targets
        )

        # Calculate network score
        network_score = self._calculate_network_score(
            original_ratio, reply_guy_score, clout_score,
            potential_ring, defensive, constructive
        )

        # Detect patterns
        patterns_detected = self._detect_patterns(
            reply_ratio, original_ratio, reply_guy_score,
            engages_larger, potential_ring, defensive, constructive
        )

        # Generate flags
        red_flags, green_flags = self._generate_flags(
            reply_guy_score, clout_score, potential_ring,
            defensive, constructive, original_ratio
        )

        # Top engaged accounts
        top_engaged = reply_targets.most_common(10)

        return NetworkReport(
            network_score=network_score,
            original_tweets=original,
            replies=replies,
            quotes=quotes,
            retweets=0,  # Would need separate data
            reply_ratio=reply_ratio,
            original_ratio=original_ratio,
            reply_guy_score=reply_guy_score,
            unique_accounts_engaged=len(mention_counts),
            top_engaged_accounts=top_engaged,
            engages_larger_accounts=engages_larger,
            clout_chasing_score=clout_score,
            mutual_mentions=mutual_mentions,
            potential_shill_ring=potential_ring,
            shill_ring_accounts=ring_accounts,
            responds_to_criticism=defensive + constructive > 0,
            defensive_responses=defensive,
            constructive_responses=constructive,
            patterns=patterns,
            patterns_detected=patterns_detected,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _calculate_reply_guy_score(
        self,
        reply_ratio: float,
        reply_targets: Counter,
        total: int
    ) -> float:
        """Calculate reply guy score (0-100)."""
        if total < 10:
            return 0

        # High reply ratio contributes to score
        ratio_score = min(50, reply_ratio * 100)

        # Targeting few unique accounts increases score
        unique_targets = len(reply_targets)
        total_replies = sum(reply_targets.values())

        if total_replies > 0:
            concentration = 1 - (unique_targets / total_replies)
            concentration_score = concentration * 30
        else:
            concentration_score = 0

        # Targeting large accounts increases score
        large_targets = sum(
            count for acc, count in reply_targets.items()
            if acc in self.LARGE_ACCOUNTS
        )
        large_score = min(20, (large_targets / max(1, total_replies)) * 50)

        return min(100, ratio_score + concentration_score + large_score)

    def _calculate_clout_score(
        self,
        large_replies: int,
        total_replies: int,
        reply_targets: Counter
    ) -> float:
        """Calculate clout chasing score."""
        if total_replies < 5:
            return 0

        large_ratio = large_replies / total_replies
        score = large_ratio * 80

        # Bonus for replying to same large accounts repeatedly
        repeat_large = sum(
            count for acc, count in reply_targets.items()
            if acc in self.LARGE_ACCOUNTS and count >= 3
        )
        if repeat_large > 0:
            score += 20

        return min(100, score)

    def _build_interaction_patterns(
        self,
        reply_targets: Counter,
        mentions: Counter,
        shills: Counter
    ) -> List[InteractionPattern]:
        """Build list of interaction patterns."""
        patterns = []

        # Top reply targets
        for acc, count in reply_targets.most_common(5):
            if count >= 3:
                patterns.append(InteractionPattern(
                    account_mentioned=acc,
                    interaction_type="reply",
                    count=count,
                    sentiment="neutral",
                    example_tweet=None
                ))

        # Top shill targets
        for acc, count in shills.most_common(5):
            if count >= 3:
                patterns.append(InteractionPattern(
                    account_mentioned=acc,
                    interaction_type="mutual_shill",
                    count=count,
                    sentiment="positive",
                    example_tweet=None
                ))

        return patterns

    def _calculate_network_score(
        self,
        original_ratio: float,
        reply_guy: float,
        clout: float,
        ring: bool,
        defensive: int,
        constructive: int
    ) -> float:
        """Calculate overall network health score."""
        score = 50.0

        # Original content is good
        score += original_ratio * 30

        # Reply guy behavior is concerning
        score -= reply_guy * 0.3

        # Clout chasing is concerning
        score -= clout * 0.2

        # Shill ring is very bad
        if ring:
            score -= 25

        # Constructive responses are good
        if constructive > defensive:
            score += 10
        elif defensive > constructive * 2:
            score -= 15

        return max(0, min(100, score))

    def _detect_patterns(
        self,
        reply_ratio: float,
        original_ratio: float,
        reply_guy: float,
        engages_larger: bool,
        ring: bool,
        defensive: int,
        constructive: int
    ) -> List[str]:
        """Detect and describe patterns."""
        patterns = []

        if reply_ratio > 0.6:
            patterns.append(f"Primarily replies to others ({reply_ratio*100:.0f}% replies)")

        if original_ratio > 0.7:
            patterns.append(f"Mostly original content ({original_ratio*100:.0f}%)")

        if reply_guy > 60:
            patterns.append("Reply guy behavior - frequently replies to same accounts")

        if engages_larger:
            patterns.append("Targets larger accounts for replies (clout chasing)")

        if ring:
            patterns.append("Potential mutual promotion network detected")

        if defensive > constructive * 2 and defensive > 3:
            patterns.append("Often defensive in response to criticism")
        elif constructive > defensive and constructive > 3:
            patterns.append("Handles criticism constructively")

        return patterns

    def _generate_flags(
        self,
        reply_guy: float,
        clout: float,
        ring: bool,
        defensive: int,
        constructive: int,
        original: float
    ) -> Tuple[List[str], List[str]]:
        """Generate red and green flags."""
        red_flags = []
        green_flags = []

        if reply_guy > 70:
            red_flags.append("Heavy reply guy behavior - little original content")

        if clout > 60:
            red_flags.append("Frequently replies to large accounts (clout chasing)")

        if ring:
            red_flags.append("Part of mutual promotion network (shill ring)")

        if defensive > 5 and defensive > constructive * 2:
            red_flags.append("Defensive when challenged - blocks/attacks critics")

        # Green flags
        if original > 0.7:
            green_flags.append("Primarily creates original content")

        if constructive > defensive and constructive > 3:
            green_flags.append("Engages constructively with criticism")

        if reply_guy < 30 and original > 0.5:
            green_flags.append("Healthy balance of content types")

        return red_flags[:4], green_flags[:3]

    def generate_summary(self, report: NetworkReport) -> str:
        """Generate human-readable summary."""
        parts = []

        if report.reply_guy_score > 60:
            parts.append("Heavy reply guy behavior.")
        elif report.original_ratio > 0.7:
            parts.append("Primarily creates original content.")

        if report.potential_shill_ring:
            parts.append("Part of a mutual promotion network.")

        if report.clout_chasing_score > 50:
            parts.append("Frequently engages with larger accounts.")

        if report.constructive_responses > report.defensive_responses:
            parts.append("Handles feedback well.")
        elif report.defensive_responses > 5:
            parts.append("Often defensive to criticism.")

        if not parts:
            parts.append("Network patterns appear normal.")

        return " ".join(parts)
