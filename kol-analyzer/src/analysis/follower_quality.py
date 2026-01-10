"""
Follower Quality Analyzer - Analyze engagement patterns to assess follower authenticity.

Since we can't directly access follower lists, we infer quality from:
- Engagement patterns and ratios
- Reply quality and diversity
- Retweet patterns
- Temporal engagement distribution
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter
import statistics


@dataclass
class FollowerQualityReport:
    """Report on follower quality analysis."""
    quality_score: float = 50.0  # 0-100
    estimated_real_follower_pct: float = 50.0  # Estimated % of real followers
    engagement_authenticity: float = 50.0  # How authentic engagement appears

    # Engagement metrics
    avg_engagement_rate: float = 0.0
    engagement_variance: float = 0.0  # Low variance = suspicious
    like_to_reply_ratio: float = 0.0
    like_to_retweet_ratio: float = 0.0

    # Pattern analysis
    engagement_spikes: int = 0  # Unusual spikes suggesting bought engagement
    dead_hours_engagement: int = 0  # Engagement during typically dead hours
    first_hour_engagement_pct: float = 0.0  # % of engagement in first hour

    # Inferred metrics
    bot_follower_estimate_pct: float = 0.0
    inactive_follower_estimate_pct: float = 0.0
    engaged_follower_estimate_pct: float = 0.0

    # Flags
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'quality_score': round(self.quality_score, 1),
            'estimated_real_follower_pct': round(self.estimated_real_follower_pct, 1),
            'engagement_authenticity': round(self.engagement_authenticity, 1),
            'avg_engagement_rate': round(self.avg_engagement_rate, 4),
            'engagement_variance': round(self.engagement_variance, 4),
            'like_to_reply_ratio': round(self.like_to_reply_ratio, 2),
            'like_to_retweet_ratio': round(self.like_to_retweet_ratio, 2),
            'engagement_spikes': self.engagement_spikes,
            'dead_hours_engagement': self.dead_hours_engagement,
            'first_hour_engagement_pct': round(self.first_hour_engagement_pct, 1),
            'bot_follower_estimate_pct': round(self.bot_follower_estimate_pct, 1),
            'inactive_follower_estimate_pct': round(self.inactive_follower_estimate_pct, 1),
            'engaged_follower_estimate_pct': round(self.engaged_follower_estimate_pct, 1),
            'red_flags': self.red_flags,
            'green_flags': self.green_flags,
            'patterns_detected': self.patterns_detected
        }


class FollowerQualityAnalyzer:
    """
    Analyzes engagement patterns to infer follower quality.

    Key indicators of fake/bot followers:
    1. Engagement rate too high (>10%) or too low (<0.1%)
    2. Very consistent engagement (low variance = bots)
    3. High like-to-reply ratio (bots like but don't reply)
    4. Engagement in dead hours (3-6 AM local time)
    5. Engagement spikes on random posts
    6. No correlation between content quality and engagement
    """

    # Healthy engagement rate ranges by follower count
    HEALTHY_ENGAGEMENT_RATES = {
        'micro': (0.05, 0.15),      # <10k followers: 5-15%
        'small': (0.02, 0.08),      # 10k-50k: 2-8%
        'medium': (0.01, 0.05),     # 50k-200k: 1-5%
        'large': (0.005, 0.03),     # 200k-1M: 0.5-3%
        'mega': (0.002, 0.02),      # 1M+: 0.2-2%
    }

    # Suspicious thresholds
    SUSPICIOUSLY_CONSISTENT_CV = 0.15  # Coefficient of variation < 15% is sus
    SUSPICIOUSLY_HIGH_LIKE_REPLY_RATIO = 50.0  # 50:1 likes to replies is sus
    DEAD_HOURS = [3, 4, 5]  # 3-5 AM typically dead
    SPIKE_THRESHOLD = 3.0  # 3x median = spike

    def __init__(self):
        pass

    def analyze(
        self,
        tweets: List[dict],
        follower_count: int = 0
    ) -> FollowerQualityReport:
        """
        Analyze tweets to infer follower quality.

        Args:
            tweets: List of tweet dictionaries with engagement data
            follower_count: Account's follower count

        Returns:
            FollowerQualityReport with analysis
        """
        if not tweets or follower_count == 0:
            return FollowerQualityReport()

        # Collect engagement data
        engagement_data = self._collect_engagement_data(tweets)

        if not engagement_data['engagement_rates']:
            return FollowerQualityReport()

        # Calculate engagement metrics
        avg_engagement = statistics.mean(engagement_data['engagement_rates'])
        engagement_variance = self._calculate_cv(engagement_data['engagement_rates'])

        total_likes = sum(engagement_data['likes'])
        total_replies = sum(engagement_data['replies'])
        total_retweets = sum(engagement_data['retweets'])

        like_to_reply = total_likes / total_replies if total_replies > 0 else float('inf')
        like_to_retweet = total_likes / total_retweets if total_retweets > 0 else float('inf')

        # Detect anomalies
        spikes = self._detect_engagement_spikes(engagement_data['engagement_rates'])
        dead_hours = self._count_dead_hour_engagement(engagement_data['hours'])

        # Estimate follower composition
        estimates = self._estimate_follower_composition(
            avg_engagement,
            engagement_variance,
            like_to_reply,
            follower_count,
            spikes,
            dead_hours
        )

        # Calculate scores
        engagement_authenticity = self._calculate_authenticity_score(
            avg_engagement,
            engagement_variance,
            like_to_reply,
            follower_count,
            spikes
        )

        quality_score = self._calculate_quality_score(
            engagement_authenticity,
            estimates['real_pct'],
            estimates['bot_pct']
        )

        # Generate flags
        red_flags = self._generate_red_flags(
            avg_engagement,
            engagement_variance,
            like_to_reply,
            spikes,
            dead_hours,
            follower_count,
            estimates
        )

        green_flags = self._generate_green_flags(
            avg_engagement,
            engagement_variance,
            like_to_reply,
            follower_count,
            estimates
        )

        patterns = self._detect_patterns(
            avg_engagement,
            engagement_variance,
            like_to_reply,
            spikes,
            engagement_data
        )

        return FollowerQualityReport(
            quality_score=quality_score,
            estimated_real_follower_pct=estimates['real_pct'],
            engagement_authenticity=engagement_authenticity,
            avg_engagement_rate=avg_engagement,
            engagement_variance=engagement_variance,
            like_to_reply_ratio=like_to_reply,
            like_to_retweet_ratio=like_to_retweet,
            engagement_spikes=spikes,
            dead_hours_engagement=dead_hours,
            first_hour_engagement_pct=0.0,  # Would need timestamp data
            bot_follower_estimate_pct=estimates['bot_pct'],
            inactive_follower_estimate_pct=estimates['inactive_pct'],
            engaged_follower_estimate_pct=estimates['engaged_pct'],
            red_flags=red_flags,
            green_flags=green_flags,
            patterns_detected=patterns
        )

    def _collect_engagement_data(self, tweets: List[dict]) -> Dict:
        """Collect engagement metrics from tweets."""
        data = {
            'engagement_rates': [],
            'likes': [],
            'replies': [],
            'retweets': [],
            'hours': [],
            'has_media': [],
        }

        for tweet in tweets:
            likes = tweet.get('likes', 0) or 0
            replies = tweet.get('replies', 0) or 0
            retweets = tweet.get('retweets', 0) or 0

            total_engagement = likes + replies + retweets

            data['likes'].append(likes)
            data['replies'].append(replies)
            data['retweets'].append(retweets)
            data['has_media'].append(tweet.get('has_media', False))

            # Calculate engagement rate (per tweet, normalized later)
            if total_engagement > 0:
                data['engagement_rates'].append(total_engagement)

            # Extract hour from timestamp
            timestamp = tweet.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    data['hours'].append(dt.hour)
                except (ValueError, TypeError):
                    pass

        return data

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if len(values) < 2:
            return 0.5  # Default

        mean = statistics.mean(values)
        if mean == 0:
            return 0.5

        std = statistics.stdev(values)
        return std / mean

    def _detect_engagement_spikes(self, engagement_rates: List[float]) -> int:
        """Detect unusual engagement spikes."""
        if len(engagement_rates) < 5:
            return 0

        median = statistics.median(engagement_rates)
        if median == 0:
            return 0

        spikes = sum(1 for rate in engagement_rates if rate > median * self.SPIKE_THRESHOLD)
        return spikes

    def _count_dead_hour_engagement(self, hours: List[int]) -> int:
        """Count engagement in typically dead hours."""
        return sum(1 for h in hours if h in self.DEAD_HOURS)

    def _get_follower_tier(self, follower_count: int) -> str:
        """Get follower tier for engagement rate comparison."""
        if follower_count < 10000:
            return 'micro'
        elif follower_count < 50000:
            return 'small'
        elif follower_count < 200000:
            return 'medium'
        elif follower_count < 1000000:
            return 'large'
        else:
            return 'mega'

    def _estimate_follower_composition(
        self,
        avg_engagement: float,
        engagement_cv: float,
        like_to_reply: float,
        follower_count: int,
        spikes: int,
        dead_hours: int
    ) -> Dict[str, float]:
        """Estimate follower composition (bot %, real %, inactive %)."""
        # Start with baseline
        bot_pct = 10.0
        inactive_pct = 60.0
        engaged_pct = 30.0

        tier = self._get_follower_tier(follower_count)
        healthy_range = self.HEALTHY_ENGAGEMENT_RATES[tier]

        # Engagement rate analysis
        engagement_rate = avg_engagement / follower_count if follower_count > 0 else 0

        if engagement_rate > healthy_range[1] * 2:
            # Suspiciously high - likely fake engagement
            bot_pct += 30
            engaged_pct -= 15
        elif engagement_rate < healthy_range[0] / 2:
            # Very low - many fake/inactive followers
            inactive_pct += 20
            engaged_pct -= 15
        elif healthy_range[0] <= engagement_rate <= healthy_range[1]:
            # Healthy range
            bot_pct -= 5
            engaged_pct += 10

        # Variance analysis
        if engagement_cv < self.SUSPICIOUSLY_CONSISTENT_CV:
            bot_pct += 20  # Bots are very consistent

        # Like to reply ratio
        if like_to_reply > self.SUSPICIOUSLY_HIGH_LIKE_REPLY_RATIO:
            bot_pct += 15  # Bots like but don't reply

        # Spikes suggest purchased engagement
        if spikes > 5:
            bot_pct += 15

        # Dead hours engagement is suspicious
        if dead_hours > 10:
            bot_pct += 10

        # Normalize to 100%
        total = bot_pct + inactive_pct + engaged_pct
        bot_pct = (bot_pct / total) * 100
        inactive_pct = (inactive_pct / total) * 100
        engaged_pct = (engaged_pct / total) * 100

        return {
            'bot_pct': bot_pct,
            'inactive_pct': inactive_pct,
            'engaged_pct': engaged_pct,
            'real_pct': 100 - bot_pct
        }

    def _calculate_authenticity_score(
        self,
        avg_engagement: float,
        engagement_cv: float,
        like_to_reply: float,
        follower_count: int,
        spikes: int
    ) -> float:
        """Calculate engagement authenticity score."""
        score = 100.0

        tier = self._get_follower_tier(follower_count)
        healthy_range = self.HEALTHY_ENGAGEMENT_RATES[tier]
        engagement_rate = avg_engagement / follower_count if follower_count > 0 else 0

        # Engagement rate penalty
        if engagement_rate > healthy_range[1] * 3:
            score -= 30  # Way too high
        elif engagement_rate > healthy_range[1] * 1.5:
            score -= 15  # Somewhat high
        elif engagement_rate < healthy_range[0] * 0.3:
            score -= 20  # Very low

        # Consistency penalty
        if engagement_cv < self.SUSPICIOUSLY_CONSISTENT_CV:
            score -= 25  # Too consistent

        # Like/reply ratio penalty
        if like_to_reply > 100:
            score -= 20
        elif like_to_reply > 50:
            score -= 10

        # Spike penalty
        if spikes > 10:
            score -= 20
        elif spikes > 5:
            score -= 10

        return max(0.0, min(100.0, score))

    def _calculate_quality_score(
        self,
        authenticity: float,
        real_pct: float,
        bot_pct: float
    ) -> float:
        """Calculate overall follower quality score."""
        # Weighted combination
        score = (
            authenticity * 0.5 +
            real_pct * 0.3 +
            (100 - bot_pct) * 0.2
        )
        return max(0.0, min(100.0, score))

    def _generate_red_flags(
        self,
        avg_engagement: float,
        engagement_cv: float,
        like_to_reply: float,
        spikes: int,
        dead_hours: int,
        follower_count: int,
        estimates: Dict
    ) -> List[str]:
        """Generate red flags for follower quality."""
        flags = []

        tier = self._get_follower_tier(follower_count)
        healthy_range = self.HEALTHY_ENGAGEMENT_RATES[tier]
        engagement_rate = avg_engagement / follower_count if follower_count > 0 else 0

        if engagement_rate > healthy_range[1] * 2:
            flags.append(f"Suspiciously high engagement rate ({engagement_rate*100:.2f}%)")

        if engagement_rate < healthy_range[0] * 0.3:
            flags.append(f"Very low engagement rate ({engagement_rate*100:.2f}%) - many inactive followers")

        if engagement_cv < self.SUSPICIOUSLY_CONSISTENT_CV:
            flags.append(f"Engagement too consistent (CV: {engagement_cv:.2f}) - suggests bot activity")

        if like_to_reply > 100:
            flags.append(f"Extreme like-to-reply ratio ({like_to_reply:.0f}:1) - bot-like pattern")
        elif like_to_reply > 50:
            flags.append(f"High like-to-reply ratio ({like_to_reply:.0f}:1)")

        if spikes > 10:
            flags.append(f"Many engagement spikes ({spikes}) - possible bought engagement")

        if dead_hours > 15:
            flags.append("Significant engagement during dead hours")

        if estimates['bot_pct'] > 30:
            flags.append(f"Estimated {estimates['bot_pct']:.0f}% bot/fake followers")

        return flags[:5]

    def _generate_green_flags(
        self,
        avg_engagement: float,
        engagement_cv: float,
        like_to_reply: float,
        follower_count: int,
        estimates: Dict
    ) -> List[str]:
        """Generate green flags for follower quality."""
        flags = []

        tier = self._get_follower_tier(follower_count)
        healthy_range = self.HEALTHY_ENGAGEMENT_RATES[tier]
        engagement_rate = avg_engagement / follower_count if follower_count > 0 else 0

        if healthy_range[0] <= engagement_rate <= healthy_range[1]:
            flags.append("Healthy engagement rate for account size")

        if 0.3 <= engagement_cv <= 0.8:
            flags.append("Natural engagement variance")

        if 5 <= like_to_reply <= 30:
            flags.append("Healthy like-to-reply ratio - real engagement")

        if estimates['engaged_pct'] > 25:
            flags.append(f"Estimated {estimates['engaged_pct']:.0f}% actively engaged followers")

        if estimates['bot_pct'] < 15:
            flags.append("Low estimated bot follower percentage")

        return flags[:4]

    def _detect_patterns(
        self,
        avg_engagement: float,
        engagement_cv: float,
        like_to_reply: float,
        spikes: int,
        engagement_data: Dict
    ) -> List[str]:
        """Detect engagement patterns."""
        patterns = []

        if engagement_cv < 0.2:
            patterns.append("Unusually consistent engagement (potential automation)")

        if like_to_reply > 50:
            patterns.append("Low reply engagement (passive audience or bots)")

        if spikes > 5:
            patterns.append("Sporadic engagement spikes")

        # Check if media posts get more engagement
        media_engagement = [
            e for e, has_media in zip(
                engagement_data['engagement_rates'],
                engagement_data['has_media']
            ) if has_media
        ]
        non_media = [
            e for e, has_media in zip(
                engagement_data['engagement_rates'],
                engagement_data['has_media']
            ) if not has_media
        ]

        if media_engagement and non_media:
            media_avg = statistics.mean(media_engagement)
            non_media_avg = statistics.mean(non_media)
            if media_avg > non_media_avg * 1.5:
                patterns.append("Media posts get significantly more engagement (normal)")
            elif non_media_avg > media_avg * 1.5:
                patterns.append("Text posts outperform media (unusual)")

        return patterns[:4]

    def generate_summary(self, report: FollowerQualityReport) -> str:
        """Generate human-readable summary."""
        parts = []

        if report.quality_score >= 75:
            parts.append("High-quality follower base with authentic engagement.")
        elif report.quality_score >= 50:
            parts.append("Mixed follower quality with some authenticity concerns.")
        else:
            parts.append("Low follower quality - significant fake/bot presence likely.")

        parts.append(
            f"Estimated {report.estimated_real_follower_pct:.0f}% real followers, "
            f"{report.bot_follower_estimate_pct:.0f}% bots."
        )

        if report.engagement_spikes > 5:
            parts.append("Multiple engagement anomalies detected.")

        return " ".join(parts)
