"""
Engagement Bait Analyzer - Detect manipulation tactics in KOL tweets.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class BaitType(Enum):
    """Types of engagement bait."""
    RAGE_BAIT = "rage_bait"
    FOMO_MANUFACTURING = "fomo_manufacturing"
    ENGAGEMENT_FARMING = "engagement_farming"
    CLIFFHANGER_ABUSE = "cliffhanger_abuse"
    REPLY_TRAP = "reply_trap"
    REWARD_GAMING = "reward_gaming"  # Kaito, Galxe, etc.
    SYMPATHY_FARMING = "sympathy_farming"
    HUMBLE_BRAG_BAIT = "humble_brag_bait"


@dataclass
class BaitInstance:
    """A detected instance of engagement bait."""
    tweet_id: str
    tweet_text: str
    bait_type: BaitType
    matched_patterns: List[str]
    severity: str  # "low", "medium", "high"
    timestamp: str

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:150] + '...' if len(self.tweet_text) > 150 else self.tweet_text,
            'bait_type': self.bait_type.value,
            'matched_patterns': self.matched_patterns[:2],
            'severity': self.severity,
            'timestamp': self.timestamp
        }


@dataclass
class EngagementBaitReport:
    """Report on engagement baiting behavior."""
    bait_instances: List[BaitInstance] = field(default_factory=list)
    bait_type_counts: Dict[str, int] = field(default_factory=dict)
    authenticity_score: float = 100.0  # 100 = genuine
    manipulation_index: float = 0.0  # 0-100, higher = more manipulative
    reward_platform_mentions: List[str] = field(default_factory=list)
    engagement_reward_optimization: bool = False
    verdict: str = "GENUINE"
    recommendations: List[str] = field(default_factory=list)
    total_tweets_analyzed: int = 0
    bait_percentage: float = 0.0

    def to_dict(self) -> dict:
        return {
            'bait_instances': [bi.to_dict() for bi in self.bait_instances[:15]],
            'bait_type_counts': self.bait_type_counts,
            'authenticity_score': round(self.authenticity_score, 1),
            'manipulation_index': round(self.manipulation_index, 1),
            'reward_platform_mentions': self.reward_platform_mentions,
            'engagement_reward_optimization': self.engagement_reward_optimization,
            'verdict': self.verdict,
            'recommendations': self.recommendations,
            'total_tweets_analyzed': self.total_tweets_analyzed,
            'bait_percentage': round(self.bait_percentage, 1)
        }


class EngagementBaitAnalyzer:
    """
    Analyzes tweets for engagement manipulation tactics.

    Detection categories:
    - FOMO manufacturing
    - Engagement farming (like/RT requests)
    - Reward gaming (Kaito, Galxe mentions)
    - Reply traps (simple question farming)
    - Rage bait
    - Cliffhanger abuse
    """

    # FOMO patterns
    FOMO_PATTERNS = [
        r'\b(last chance|final opportunity|don\'t miss)\b',
        r'\b(about to (pump|moon|explode|send))\b',
        r'\b(insider|secret|exclusive|alpha leak)\b',
        r'\b(next 100x|guaranteed|easy money)\b',
        r'\b(won\'t last|running out|limited time)\b',
        r'\b(before it\'s too late|while you can)\b',
        r'\b(this is (it|the one)|the play)\b',
        r'\b(never seen (before|anything like))\b',
    ]

    # Engagement farming patterns
    ENGAGEMENT_FARMING_PATTERNS = [
        r'\b(like if you agree|rt if you|retweet if)\b',
        r'\b(drop a|comment below|let me know)\b',
        r'\b(follow for more|bookmark this)\b',
        r'\b(share (this|if)|spread the word)\b',
        r'\b(tag (a friend|someone)|@ your)\b',
        r'\b(give this a like|hit that like)\b',
        r'👇|⬇️',  # Arrow down (often used with "drop below")
    ]

    # Reward gaming patterns
    REWARD_GAMING_PATTERNS = [
        r'\b(kaito|yaps?|galxe|zealy|crew3|layer3)\b',
        r'\b(points?|airdrop|rewards?).{0,20}(tweet|post|engage)',
        r'\b(farm|farming).{0,15}(points?|rewards?|airdrop)',
        r'\b(quest|mission|task).{0,15}complete',
        r'\b(daily|weekly).{0,10}(points?|rewards?)',
    ]

    # Reply trap patterns
    REPLY_TRAP_PATTERNS = [
        r'^(what\'s your|name your).{0,30}\?$',
        r'^(favorite|best|worst).{0,50}\?$',
        r'\b(wrong answers only)\b',
        r'^(hot take|unpopular opinion)[:.]',
        r'\b(fill in the blank|complete this)\b',
        r'^(one word|two words?|three words?).{0,20}[:.]',
        r'^(yes or no|agree or disagree)\b',
    ]

    # Rage bait patterns
    RAGE_BAIT_PATTERNS = [
        r'\b(is (dead|trash|over|finished))\b',
        r'\b(worst (take|opinion|decision))\b',
        r'\b(can\'t believe (people|anyone))\b',
        r'\b(overrated|overhyped|scam)\b',
        r'\b(prove me wrong|change my mind)\b',
        r'\b(ratio|ratioed)\b',
        r'\b(horrible|terrible|disgusting).{0,20}(take|opinion)',
    ]

    # Cliffhanger patterns
    CLIFFHANGER_PATTERNS = [
        r'\b(thread (coming|incoming|soon))\b',
        r'\b(wait (for|til|until) you see)\b',
        r'\b(you won\'t believe)\b',
        r'\b(stay tuned|more coming)\b',
        r'\b(announcement (coming|soon))\b',
        r'\.\.\.$',  # Ends with ellipsis
        r'\b(big news|huge announcement)\b.{0,20}(soon|coming)',
    ]

    # Sympathy farming patterns
    SYMPATHY_FARMING_PATTERNS = [
        r'\b(been (struggling|hard|tough))\b',
        r'\b(no one (cares|listens|supports))\b',
        r'\b(feeling (down|low|depressed))\b',
        r'\b(almost gave up|wanted to quit)\b',
        r'\b(if this flops|if no one (sees|engages))\b',
    ]

    # Humble brag bait patterns
    HUMBLE_BRAG_BAIT_PATTERNS = [
        r'\b(accidentally|somehow).{0,30}(made|gained|profit)',
        r'\b(crazy|insane|wild).{0,20}(returns?|gains?|profit)',
        r'\b(just realized|looking at).{0,20}(up|gains|profit)',
        r'\b(wasn\'t even trying|didn\'t expect)\b',
    ]

    # Known reward platforms
    REWARD_PLATFORMS = ['kaito', 'galxe', 'zealy', 'crew3', 'layer3', 'rabbithole', 'quest']

    def __init__(self):
        # Compile all patterns
        self.pattern_groups = {
            BaitType.FOMO_MANUFACTURING: [re.compile(p, re.IGNORECASE) for p in self.FOMO_PATTERNS],
            BaitType.ENGAGEMENT_FARMING: [re.compile(p, re.IGNORECASE) for p in self.ENGAGEMENT_FARMING_PATTERNS],
            BaitType.REWARD_GAMING: [re.compile(p, re.IGNORECASE) for p in self.REWARD_GAMING_PATTERNS],
            BaitType.REPLY_TRAP: [re.compile(p, re.IGNORECASE) for p in self.REPLY_TRAP_PATTERNS],
            BaitType.RAGE_BAIT: [re.compile(p, re.IGNORECASE) for p in self.RAGE_BAIT_PATTERNS],
            BaitType.CLIFFHANGER_ABUSE: [re.compile(p, re.IGNORECASE) for p in self.CLIFFHANGER_PATTERNS],
            BaitType.SYMPATHY_FARMING: [re.compile(p, re.IGNORECASE) for p in self.SYMPATHY_FARMING_PATTERNS],
            BaitType.HUMBLE_BRAG_BAIT: [re.compile(p, re.IGNORECASE) for p in self.HUMBLE_BRAG_BAIT_PATTERNS],
        }

    def analyze(self, tweets: List[dict]) -> EngagementBaitReport:
        """
        Analyze tweets for engagement bait patterns.

        Args:
            tweets: List of tweet dictionaries

        Returns:
            EngagementBaitReport with analysis results
        """
        if not tweets:
            return EngagementBaitReport()

        bait_instances: List[BaitInstance] = []
        bait_type_counts: Dict[str, int] = {bt.value: 0 for bt in BaitType}
        reward_platform_mentions: List[str] = []
        total_bait_severity = 0.0

        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Check each bait type
            for bait_type, patterns in self.pattern_groups.items():
                matched = []
                for pattern in patterns:
                    if pattern.search(text):
                        matched.append(pattern.pattern)

                if matched:
                    severity = self._determine_severity(bait_type, len(matched), text)

                    bait_instances.append(BaitInstance(
                        tweet_id=tweet_id,
                        tweet_text=text,
                        bait_type=bait_type,
                        matched_patterns=matched,
                        severity=severity,
                        timestamp=timestamp
                    ))

                    bait_type_counts[bait_type.value] += 1
                    total_bait_severity += {'low': 1, 'medium': 2, 'high': 3}.get(severity, 1)

            # Track reward platform mentions specifically
            for platform in self.REWARD_PLATFORMS:
                if platform.lower() in text.lower():
                    if platform not in reward_platform_mentions:
                        reward_platform_mentions.append(platform)

        # Calculate scores
        total_tweets = len(tweets)
        total_bait = len(bait_instances)
        bait_percentage = (total_bait / total_tweets * 100) if total_tweets > 0 else 0

        # Manipulation index (0-100)
        manipulation_index = min(100, (total_bait_severity / total_tweets * 20)) if total_tweets > 0 else 0

        # Authenticity score (inverse of manipulation)
        authenticity_score = max(0, 100 - manipulation_index)

        # Check for engagement reward optimization
        engagement_reward_optimization = (
            bait_type_counts.get(BaitType.REWARD_GAMING.value, 0) > 3 or
            len(reward_platform_mentions) > 0
        )

        # Generate verdict
        verdict = self._generate_verdict(manipulation_index, bait_type_counts)

        # Generate recommendations
        recommendations = self._generate_recommendations(bait_type_counts, manipulation_index)

        return EngagementBaitReport(
            bait_instances=bait_instances,
            bait_type_counts=bait_type_counts,
            authenticity_score=authenticity_score,
            manipulation_index=manipulation_index,
            reward_platform_mentions=reward_platform_mentions,
            engagement_reward_optimization=engagement_reward_optimization,
            verdict=verdict,
            recommendations=recommendations,
            total_tweets_analyzed=total_tweets,
            bait_percentage=bait_percentage
        )

    def _determine_severity(
        self,
        bait_type: BaitType,
        match_count: int,
        text: str
    ) -> str:
        """Determine the severity of a bait instance."""
        # High severity bait types
        high_severity_types = {
            BaitType.FOMO_MANUFACTURING,
            BaitType.REWARD_GAMING,
        }

        # Check for multiple pattern matches
        if match_count >= 3:
            return "high"
        elif match_count >= 2:
            return "medium" if bait_type in high_severity_types else "low"
        else:
            return "medium" if bait_type in high_severity_types else "low"

    def _generate_verdict(
        self,
        manipulation_index: float,
        bait_type_counts: Dict[str, int]
    ) -> str:
        """Generate an overall verdict."""
        if manipulation_index < 10:
            return "GENUINE"
        elif manipulation_index < 25:
            return "MOSTLY GENUINE"
        elif manipulation_index < 45:
            return "MODERATE BAITING"
        elif manipulation_index < 65:
            return "SIGNIFICANT BAITING"
        else:
            return "HEAVY MANIPULATION"

    def _generate_recommendations(
        self,
        bait_type_counts: Dict[str, int],
        manipulation_index: float
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if bait_type_counts.get(BaitType.FOMO_MANUFACTURING.value, 0) > 5:
            recommendations.append("High FOMO tactics detected - verify claims independently before acting")

        if bait_type_counts.get(BaitType.ENGAGEMENT_FARMING.value, 0) > 10:
            recommendations.append("Frequent engagement farming - content may prioritize metrics over value")

        if bait_type_counts.get(BaitType.REWARD_GAMING.value, 0) > 3:
            recommendations.append("Active reward platform participation - posts may be incentive-driven")

        if bait_type_counts.get(BaitType.REPLY_TRAP.value, 0) > 10:
            recommendations.append("Heavy use of reply traps - engagement may be artificially inflated")

        if bait_type_counts.get(BaitType.RAGE_BAIT.value, 0) > 5:
            recommendations.append("Rage bait detected - may prioritize controversy over accuracy")

        if manipulation_index < 15:
            recommendations.append("Engagement patterns appear organic and genuine")

        return recommendations

    def generate_summary(self, report: EngagementBaitReport) -> str:
        """Generate a human-readable summary."""
        parts = [f"Verdict: {report.verdict}"]

        if report.bait_percentage > 20:
            parts.append(f"High bait percentage ({report.bait_percentage:.1f}%)")

        # Find top bait types
        top_types = sorted(
            [(k, v) for k, v in report.bait_type_counts.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:2]

        if top_types:
            type_strs = [f"{t[0].replace('_', ' ')} ({t[1]})" for t in top_types]
            parts.append(f"Top patterns: {', '.join(type_strs)}")

        if report.engagement_reward_optimization:
            parts.append("Active Kaito/reward platform participant")

        return ". ".join(parts) + "."
