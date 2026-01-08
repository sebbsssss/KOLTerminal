"""
Engagement Analyzer - Social Blade-style analysis for detecting fake/bot engagement.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter


@dataclass
class EngagementProfile:
    """Profile of engagement patterns for a KOL."""
    avg_engagement_rate: float  # (likes + retweets + replies) / follower_count
    engagement_consistency: float  # Coefficient of variation - low = suspicious
    like_reply_ratio: float  # >50 is suspicious
    bot_follower_estimate: float  # Percentage estimate
    suspicious_patterns: List[str] = field(default_factory=list)
    authenticity_score: float = 100.0  # 0-100

    # Additional metrics
    median_likes: float = 0.0
    median_retweets: float = 0.0
    median_replies: float = 0.0
    engagement_spikes: int = 0
    posting_hour_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'avg_engagement_rate': round(self.avg_engagement_rate, 4),
            'engagement_consistency': round(self.engagement_consistency, 4),
            'like_reply_ratio': round(self.like_reply_ratio, 2),
            'bot_follower_estimate': round(self.bot_follower_estimate, 2),
            'suspicious_patterns': self.suspicious_patterns,
            'authenticity_score': round(self.authenticity_score, 1),
            'median_likes': self.median_likes,
            'median_retweets': self.median_retweets,
            'median_replies': self.median_replies,
            'engagement_spikes': self.engagement_spikes,
            'posting_hour_distribution': self.posting_hour_distribution
        }


class EngagementAnalyzer:
    """
    Analyzes engagement patterns to detect fake/bot activity.

    Detection logic:
    - Calculate engagement rate per tweet: (likes + retweets + replies) / follower_count
    - Flag if coefficient of variation < 0.1 (too consistent = likely bots)
    - Flag if like-to-reply ratio > 50 (bots like but don't comment)
    - Flag engagement spikes on text-only tweets (coordinated promotion)
    - Flag posts clustered in specific hours (automated posting)
    """

    # Thresholds
    CV_THRESHOLD = 0.1  # Below this = suspicious consistency
    LIKE_REPLY_RATIO_THRESHOLD = 50.0  # Above this = suspicious
    ENGAGEMENT_SPIKE_MULTIPLIER = 5.0  # 5x median = spike
    HOUR_CONCENTRATION_THRESHOLD = 0.4  # 40% in one hour = suspicious

    def __init__(self):
        self.cv_threshold = self.CV_THRESHOLD
        self.like_reply_ratio_threshold = self.LIKE_REPLY_RATIO_THRESHOLD

    def analyze(
        self,
        tweets: List[dict],
        follower_count: int
    ) -> EngagementProfile:
        """
        Analyze engagement patterns from tweets.

        Args:
            tweets: List of tweet dictionaries with likes, retweets, replies, etc.
            follower_count: The KOL's follower count

        Returns:
            EngagementProfile with analysis results
        """
        if not tweets or follower_count <= 0:
            return EngagementProfile(
                avg_engagement_rate=0.0,
                engagement_consistency=0.0,
                like_reply_ratio=0.0,
                bot_follower_estimate=0.0,
                suspicious_patterns=["Insufficient data for analysis"],
                authenticity_score=50.0
            )

        suspicious_patterns = []
        penalty = 0.0

        # Calculate engagement metrics
        engagement_rates = []
        likes_list = []
        retweets_list = []
        replies_list = []
        posting_hours = []

        for tweet in tweets:
            likes = tweet.get('likes', 0)
            retweets = tweet.get('retweets', 0)
            replies = tweet.get('replies', 0)

            likes_list.append(likes)
            retweets_list.append(retweets)
            replies_list.append(replies)

            total_engagement = likes + retweets + replies
            rate = total_engagement / follower_count if follower_count > 0 else 0
            engagement_rates.append(rate)

            # Extract posting hour
            timestamp = tweet.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    posting_hours.append(dt.hour)
                except (ValueError, TypeError):
                    pass

        # Calculate statistics
        avg_engagement_rate = statistics.mean(engagement_rates) if engagement_rates else 0
        median_likes = statistics.median(likes_list) if likes_list else 0
        median_retweets = statistics.median(retweets_list) if retweets_list else 0
        median_replies = statistics.median(replies_list) if replies_list else 0

        # Calculate coefficient of variation (CV)
        if len(engagement_rates) > 1 and avg_engagement_rate > 0:
            std_dev = statistics.stdev(engagement_rates)
            engagement_cv = std_dev / avg_engagement_rate
        else:
            engagement_cv = 0.5  # Default to moderate variance

        # Calculate like-to-reply ratio
        total_likes = sum(likes_list)
        total_replies = sum(replies_list)
        like_reply_ratio = total_likes / total_replies if total_replies > 0 else 100.0

        # Hour distribution
        hour_distribution = Counter(posting_hours)
        posting_hour_dict = dict(hour_distribution)

        # === Detection Logic ===

        # 1. Check engagement consistency (CV too low = bots)
        if engagement_cv < self.cv_threshold:
            suspicious_patterns.append(
                f"Suspiciously consistent engagement (CV: {engagement_cv:.3f})"
            )
            penalty += 20.0

        # 2. Check like-to-reply ratio
        if like_reply_ratio > self.like_reply_ratio_threshold:
            suspicious_patterns.append(
                f"High like-to-reply ratio ({like_reply_ratio:.1f}:1) suggests bot likes"
            )
            penalty += 15.0

        # 3. Check for engagement spikes on text-only tweets
        engagement_spikes = 0
        median_engagement = statistics.median([sum(x) for x in zip(likes_list, retweets_list, replies_list)])

        for i, tweet in enumerate(tweets):
            total = tweets[i].get('likes', 0) + tweets[i].get('retweets', 0) + tweets[i].get('replies', 0)
            has_media = tweet.get('has_media', False) or tweet.get('has_video', False)

            if not has_media and total > median_engagement * self.ENGAGEMENT_SPIKE_MULTIPLIER:
                engagement_spikes += 1

        if engagement_spikes > len(tweets) * 0.1:  # More than 10% are spikes
            suspicious_patterns.append(
                f"Unusual engagement spikes on text-only tweets ({engagement_spikes} instances)"
            )
            penalty += 15.0

        # 4. Check posting hour concentration (automated posting)
        if posting_hours:
            max_hour_count = max(hour_distribution.values()) if hour_distribution else 0
            hour_concentration = max_hour_count / len(posting_hours)

            if hour_concentration > self.HOUR_CONCENTRATION_THRESHOLD:
                peak_hour = max(hour_distribution, key=hour_distribution.get)
                suspicious_patterns.append(
                    f"Posts concentrated at specific hour (UTC {peak_hour}:00 - {hour_concentration:.0%})"
                )
                penalty += 10.0

        # 5. Check for unrealistic engagement rate
        if avg_engagement_rate > 0.2:  # 20% engagement is very high
            suspicious_patterns.append(
                f"Unusually high engagement rate ({avg_engagement_rate:.1%})"
            )
            penalty += 10.0
        elif avg_engagement_rate < 0.001:  # Less than 0.1% is very low
            suspicious_patterns.append(
                f"Very low engagement rate ({avg_engagement_rate:.2%}) - possible fake followers"
            )
            penalty += 10.0

        # Estimate bot follower percentage
        bot_estimate = self._estimate_bot_followers(
            engagement_cv,
            like_reply_ratio,
            avg_engagement_rate,
            follower_count
        )

        if bot_estimate > 30:
            suspicious_patterns.append(f"Estimated {bot_estimate:.0f}% bot followers")
            penalty += min(bot_estimate / 5, 15.0)

        # Calculate authenticity score
        authenticity_score = max(0.0, 100.0 - penalty)

        # Add positive signals
        if engagement_cv > 0.5 and len(suspicious_patterns) == 0:
            # Healthy variance and no issues
            pass

        return EngagementProfile(
            avg_engagement_rate=avg_engagement_rate,
            engagement_consistency=engagement_cv,
            like_reply_ratio=like_reply_ratio,
            bot_follower_estimate=bot_estimate,
            suspicious_patterns=suspicious_patterns,
            authenticity_score=authenticity_score,
            median_likes=median_likes,
            median_retweets=median_retweets,
            median_replies=median_replies,
            engagement_spikes=engagement_spikes,
            posting_hour_distribution=posting_hour_dict
        )

    def _estimate_bot_followers(
        self,
        cv: float,
        like_reply_ratio: float,
        engagement_rate: float,
        follower_count: int
    ) -> float:
        """
        Estimate the percentage of bot followers based on engagement signals.

        This is a heuristic estimate based on multiple factors.
        """
        bot_score = 0.0

        # Low CV suggests coordinated engagement
        if cv < 0.1:
            bot_score += 30.0
        elif cv < 0.2:
            bot_score += 15.0

        # High like-to-reply ratio suggests bots
        if like_reply_ratio > 100:
            bot_score += 25.0
        elif like_reply_ratio > 50:
            bot_score += 15.0
        elif like_reply_ratio > 30:
            bot_score += 5.0

        # Abnormal engagement rates
        if engagement_rate > 0.15:
            bot_score += 20.0
        elif engagement_rate < 0.001:
            bot_score += 15.0

        # Large accounts with very high engagement are suspicious
        if follower_count > 100000 and engagement_rate > 0.1:
            bot_score += 15.0

        return min(bot_score, 80.0)  # Cap at 80%

    def get_engagement_grade(self, profile: EngagementProfile) -> str:
        """Get a letter grade for engagement authenticity."""
        score = profile.authenticity_score
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    def generate_summary(self, profile: EngagementProfile) -> str:
        """Generate a human-readable summary of engagement analysis."""
        grade = self.get_engagement_grade(profile)

        summary_parts = [f"Engagement Grade: {grade}"]

        if profile.authenticity_score >= 80:
            summary_parts.append("Engagement patterns appear authentic with healthy variance.")
        elif profile.authenticity_score >= 60:
            summary_parts.append("Engagement shows some concerning patterns but is mostly normal.")
        elif profile.authenticity_score >= 40:
            summary_parts.append("Multiple red flags detected in engagement patterns.")
        else:
            summary_parts.append("Significant evidence of artificial engagement detected.")

        if profile.suspicious_patterns:
            summary_parts.append(f"Issues: {', '.join(profile.suspicious_patterns[:3])}")

        return " ".join(summary_parts)
