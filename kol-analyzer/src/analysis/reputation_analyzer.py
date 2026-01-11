"""
Reputation Analyzer - Analyze what others say ABOUT the KOL.

This module searches for:
- Mentions of the KOL by other users
- Call-outs, warnings, and accusations
- Community sentiment and trust signals
- Notable accounts warning about the KOL
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MentionSentiment(Enum):
    """Sentiment of mentions about the KOL."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    WARNING = "warning"  # Explicit warnings/call-outs
    ACCUSATION = "accusation"  # Scam/fraud accusations


@dataclass
class MentionAnalysis:
    """Analysis of a single mention."""
    author: str
    author_followers: int
    text: str
    sentiment: MentionSentiment
    keywords_found: List[str]
    engagement: int  # likes + retweets
    is_notable_account: bool = False  # Known fraud investigators, large accounts


@dataclass
class ReputationReport:
    """Report on KOL's external reputation."""
    reputation_score: float = 50.0  # 0-100, higher = better reputation

    # Mention counts
    total_mentions_analyzed: int = 0
    positive_mentions: int = 0
    negative_mentions: int = 0
    warning_mentions: int = 0
    accusation_mentions: int = 0

    # Weighted by engagement
    weighted_sentiment: float = 0.0  # -100 to +100

    # Notable findings
    notable_callouts: List[str] = field(default_factory=list)  # From large/verified accounts
    scam_accusations: List[str] = field(default_factory=list)
    fraud_warnings: List[str] = field(default_factory=list)

    # Positive signals
    endorsements: List[str] = field(default_factory=list)
    accuracy_praise: List[str] = field(default_factory=list)

    # Patterns
    recurring_complaints: List[str] = field(default_factory=list)

    # Flags
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'reputation_score': round(self.reputation_score, 1),
            'total_mentions_analyzed': self.total_mentions_analyzed,
            'positive_mentions': self.positive_mentions,
            'negative_mentions': self.negative_mentions,
            'warning_mentions': self.warning_mentions,
            'accusation_mentions': self.accusation_mentions,
            'weighted_sentiment': round(self.weighted_sentiment, 1),
            'notable_callouts': self.notable_callouts[:5],
            'scam_accusations': self.scam_accusations[:5],
            'fraud_warnings': self.fraud_warnings[:5],
            'endorsements': self.endorsements[:5],
            'recurring_complaints': self.recurring_complaints[:5],
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class ReputationAnalyzer:
    """Analyzes external reputation by examining mentions from other users."""

    # Known fraud investigators and trusted accounts (example list)
    NOTABLE_ACCOUNTS = {
        'zachxbt', 'coffeezilla', 'web3isgreat', 'molaboratorio',
        'fatmanterra', 'lookonchain', 'whale_alert', 'unusual_whales',
        'theblock__', 'tier10k', 'deltecbank', 'bitfinexed'
    }

    # Negative patterns - scam/fraud accusations
    SCAM_PATTERNS = [
        r'\b(scam|scammer|scamming|scammed)\b',
        r'\b(rug|rugged|rugpull|rug\s*pull)\b',
        r'\b(fraud|fraudster|fraudulent)\b',
        r'\b(ponzi|pyramid)\b',
        r'\b(exit\s*scam)\b',
        r'\b(steal|stole|stolen|stealing)\b',
    ]

    # Warning patterns
    WARNING_PATTERNS = [
        r'\b(warning|warn|beware|careful|caution)\b',
        r'\b(don\'?t\s*trust|do\s*not\s*trust|never\s*trust)\b',
        r'\b(exposed|exposing|calling\s*out|called\s*out)\b',
        r'\b(fake|faker|faking|faked)\b',
        r'\b(paid\s*promo|paid\s*shill|paid\s*promotion)\b',
        r'\b(baiting|baited|bait)\b',
        r'\b(manipulation|manipulating|manipulated)\b',
    ]

    # Negative patterns - bad calls/performance
    BAD_CALL_PATTERNS = [
        r'\b(wrong\s*again|always\s*wrong|never\s*right)\b',
        r'\b(bad\s*call|terrible\s*call|awful\s*call)\b',
        r'\b(lost\s*money|lost\s*everything|rekt)\b',
        r'\b(inverse|fade|do\s*opposite)\b',
        r'\b(worst\s*advice|terrible\s*advice)\b',
        r'\b(clown|joke|laughing\s*stock)\b',
    ]

    # Positive patterns
    POSITIVE_PATTERNS = [
        r'\b(good\s*call|great\s*call|nice\s*call)\b',
        r'\b(was\s*right|called\s*it|nailed\s*it)\b',
        r'\b(accurate|reliable|consistent)\b',
        r'\b(trusted|trustworthy|honest)\b',
        r'\b(thank\s*you|thanks|grateful)\b',
        r'\b(legend|goat|based)\b',
        r'\b(helpful|valuable|insightful)\b',
    ]

    def __init__(self):
        # Compile patterns
        self.scam_patterns = [re.compile(p, re.IGNORECASE) for p in self.SCAM_PATTERNS]
        self.warning_patterns = [re.compile(p, re.IGNORECASE) for p in self.WARNING_PATTERNS]
        self.bad_call_patterns = [re.compile(p, re.IGNORECASE) for p in self.BAD_CALL_PATTERNS]
        self.positive_patterns = [re.compile(p, re.IGNORECASE) for p in self.POSITIVE_PATTERNS]

    def analyze(
        self,
        mentions: List[Dict],
        kol_username: str
    ) -> ReputationReport:
        """
        Analyze mentions of a KOL from other users.

        Args:
            mentions: List of tweet dicts containing mentions of the KOL
                     Each should have: text, author, author_followers, likes, retweets
            kol_username: The KOL's username (to exclude their own tweets)

        Returns:
            ReputationReport with reputation analysis
        """
        if not mentions:
            return self._empty_report()

        # Filter out the KOL's own tweets
        external_mentions = [
            m for m in mentions
            if m.get('author', '').lower() != kol_username.lower()
        ]

        if not external_mentions:
            return self._empty_report()

        # Analyze each mention
        analyses = []
        for mention in external_mentions:
            analysis = self._analyze_mention(mention)
            analyses.append(analysis)

        # Aggregate results
        return self._aggregate_analyses(analyses, kol_username)

    def _analyze_mention(self, mention: Dict) -> MentionAnalysis:
        """Analyze a single mention."""
        text = mention.get('text', '')
        author = mention.get('author', 'unknown')
        author_followers = mention.get('author_followers', 0)
        likes = mention.get('likes', 0)
        retweets = mention.get('retweets', 0)

        # Check for patterns
        scam_keywords = self._find_matches(text, self.scam_patterns)
        warning_keywords = self._find_matches(text, self.warning_patterns)
        bad_call_keywords = self._find_matches(text, self.bad_call_patterns)
        positive_keywords = self._find_matches(text, self.positive_patterns)

        # Determine sentiment
        all_negative = scam_keywords + warning_keywords + bad_call_keywords

        if scam_keywords:
            sentiment = MentionSentiment.ACCUSATION
        elif warning_keywords:
            sentiment = MentionSentiment.WARNING
        elif bad_call_keywords:
            sentiment = MentionSentiment.NEGATIVE
        elif positive_keywords:
            sentiment = MentionSentiment.POSITIVE
        else:
            sentiment = MentionSentiment.NEUTRAL

        # Check if notable account
        is_notable = author.lower() in self.NOTABLE_ACCOUNTS or author_followers > 100000

        return MentionAnalysis(
            author=author,
            author_followers=author_followers,
            text=text[:500],  # Truncate for storage
            sentiment=sentiment,
            keywords_found=all_negative + positive_keywords,
            engagement=likes + retweets,
            is_notable_account=is_notable
        )

    def _find_matches(self, text: str, patterns: List[re.Pattern]) -> List[str]:
        """Find all matching patterns in text."""
        matches = []
        for pattern in patterns:
            found = pattern.findall(text)
            matches.extend(found)
        return matches

    def _aggregate_analyses(
        self,
        analyses: List[MentionAnalysis],
        kol_username: str
    ) -> ReputationReport:
        """Aggregate individual mention analyses into a report."""
        report = ReputationReport()
        report.total_mentions_analyzed = len(analyses)

        total_weight = 0
        weighted_sum = 0

        for analysis in analyses:
            # Weight by engagement and follower count
            weight = 1 + (analysis.engagement / 100) + (analysis.author_followers / 10000)
            weight = min(weight, 50)  # Cap weight

            total_weight += weight

            # Count by sentiment
            if analysis.sentiment == MentionSentiment.POSITIVE:
                report.positive_mentions += 1
                weighted_sum += weight * 1
                if analysis.is_notable_account:
                    report.endorsements.append(
                        f"@{analysis.author}: {analysis.text[:100]}..."
                    )
            elif analysis.sentiment == MentionSentiment.ACCUSATION:
                report.accusation_mentions += 1
                weighted_sum -= weight * 3  # Heavy penalty for accusations
                report.scam_accusations.append(
                    f"@{analysis.author}: {analysis.text[:100]}..."
                )
                if analysis.is_notable_account:
                    report.notable_callouts.append(
                        f"@{analysis.author} ({analysis.author_followers:,} followers): {analysis.text[:100]}..."
                    )
            elif analysis.sentiment == MentionSentiment.WARNING:
                report.warning_mentions += 1
                weighted_sum -= weight * 2
                report.fraud_warnings.append(
                    f"@{analysis.author}: {analysis.text[:100]}..."
                )
                if analysis.is_notable_account:
                    report.notable_callouts.append(
                        f"@{analysis.author} ({analysis.author_followers:,} followers): {analysis.text[:100]}..."
                    )
            elif analysis.sentiment == MentionSentiment.NEGATIVE:
                report.negative_mentions += 1
                weighted_sum -= weight * 1

        # Calculate weighted sentiment (-100 to +100)
        if total_weight > 0:
            report.weighted_sentiment = (weighted_sum / total_weight) * 50
            report.weighted_sentiment = max(-100, min(100, report.weighted_sentiment))

        # Calculate reputation score (0-100)
        # Start at 50, adjust based on sentiment
        report.reputation_score = 50 + (report.weighted_sentiment / 2)
        report.reputation_score = max(0, min(100, report.reputation_score))

        # Find recurring complaints
        report.recurring_complaints = self._find_recurring_themes(analyses)

        # Generate flags
        report.red_flags, report.green_flags = self._generate_flags(report)

        return report

    def _find_recurring_themes(self, analyses: List[MentionAnalysis]) -> List[str]:
        """Find recurring complaint themes."""
        keyword_counts: Dict[str, int] = {}

        for analysis in analyses:
            if analysis.sentiment in [MentionSentiment.ACCUSATION, MentionSentiment.WARNING, MentionSentiment.NEGATIVE]:
                for keyword in analysis.keywords_found:
                    keyword_lower = keyword.lower()
                    keyword_counts[keyword_lower] = keyword_counts.get(keyword_lower, 0) + 1

        # Return keywords mentioned more than once
        recurring = [k for k, v in keyword_counts.items() if v > 1]
        return sorted(recurring, key=lambda k: keyword_counts[k], reverse=True)[:5]

    def _generate_flags(self, report: ReputationReport) -> Tuple[List[str], List[str]]:
        """Generate red and green flags from the report."""
        red_flags = []
        green_flags = []

        # Red flags
        if report.accusation_mentions > 0:
            red_flags.append(f"Scam/fraud accusations from {report.accusation_mentions} users")

        if report.notable_callouts:
            red_flags.append(f"Called out by notable accounts ({len(report.notable_callouts)} instances)")

        if report.warning_mentions >= 3:
            red_flags.append(f"Multiple warning posts about this account ({report.warning_mentions})")

        if report.weighted_sentiment < -30:
            red_flags.append("Strongly negative community sentiment")

        if report.recurring_complaints:
            red_flags.append(f"Recurring complaints: {', '.join(report.recurring_complaints[:3])}")

        if report.negative_mentions > report.positive_mentions * 2:
            red_flags.append("Negative mentions significantly outweigh positive")

        # Green flags
        if report.reputation_score >= 70 and report.total_mentions_analyzed >= 5:
            green_flags.append("Generally positive community reputation")

        if report.endorsements:
            green_flags.append(f"Endorsed by notable accounts ({len(report.endorsements)})")

        if report.positive_mentions > report.negative_mentions * 2 and report.positive_mentions >= 3:
            green_flags.append("Strong positive sentiment from community")

        if report.accusation_mentions == 0 and report.warning_mentions == 0 and report.total_mentions_analyzed >= 5:
            green_flags.append("No scam accusations or warnings found")

        return red_flags, green_flags

    def _empty_report(self) -> ReputationReport:
        """Return an empty report when no mentions available."""
        report = ReputationReport()
        report.reputation_score = 50  # Neutral when no data
        return report

    def analyze_demo(self, kol_username: str) -> ReputationReport:
        """
        Generate demo reputation data for testing.
        In production, this would search Twitter for mentions.
        """
        import random
        random.seed(hash(kol_username.lower()))

        # Generate based on username patterns
        is_suspicious = any(x in kol_username.lower() for x in ['gem', 'call', '100x', 'alpha'])

        report = ReputationReport()
        report.total_mentions_analyzed = random.randint(10, 50)

        if is_suspicious:
            # More negative for suspicious-looking accounts
            report.accusation_mentions = random.randint(2, 8)
            report.warning_mentions = random.randint(3, 10)
            report.negative_mentions = random.randint(5, 15)
            report.positive_mentions = random.randint(1, 5)
            report.scam_accusations = [
                "@user1: This guy rugged his followers on $FAKE...",
                "@user2: Classic pump and dump, been calling this out..."
            ]
            report.notable_callouts = [
                "@zachxbt (500K followers): Another paid shill exposed..."
            ]
            report.recurring_complaints = ["scam", "paid promotion", "rug"]
        else:
            # More balanced for normal accounts
            report.accusation_mentions = random.randint(0, 2)
            report.warning_mentions = random.randint(0, 3)
            report.negative_mentions = random.randint(2, 8)
            report.positive_mentions = random.randint(5, 15)

            if report.positive_mentions > report.negative_mentions:
                report.endorsements = [
                    "@trusted_trader: Solid analysis as always...",
                ]

        # Calculate scores
        total_neg = report.accusation_mentions * 3 + report.warning_mentions * 2 + report.negative_mentions
        total_pos = report.positive_mentions

        if total_neg + total_pos > 0:
            report.weighted_sentiment = ((total_pos - total_neg) / (total_pos + total_neg)) * 50

        report.reputation_score = 50 + report.weighted_sentiment
        report.reputation_score = max(0, min(100, report.reputation_score))

        # Generate flags
        report.red_flags, report.green_flags = self._generate_flags(report)

        random.seed()
        return report
