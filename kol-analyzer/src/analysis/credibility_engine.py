"""
Credibility Engine - Combine all analysis scores into a final credibility score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .engagement_analyzer import EngagementAnalyzer, EngagementProfile
from .consistency_tracker import ConsistencyTracker, ConsistencyReport
from .dissonance_analyzer import DissonanceAnalyzer, DissonanceReport
from .engagement_bait_analyzer import EngagementBaitAnalyzer, EngagementBaitReport


@dataclass
class CredibilityScore:
    """Final credibility score and assessment."""
    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    confidence: float  # How confident in assessment (0-100)
    assessment: str  # "HIGH CREDIBILITY", etc.

    engagement_score: float
    consistency_score: float
    dissonance_score: float
    baiting_score: float

    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    summary: str = ""

    # Detailed reports
    engagement_report: Optional[Dict] = None
    consistency_report: Optional[Dict] = None
    dissonance_report: Optional[Dict] = None
    baiting_report: Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            'overall_score': round(self.overall_score, 1),
            'grade': self.grade,
            'confidence': round(self.confidence, 1),
            'assessment': self.assessment,
            'engagement_score': round(self.engagement_score, 1),
            'consistency_score': round(self.consistency_score, 1),
            'dissonance_score': round(self.dissonance_score, 1),
            'baiting_score': round(self.baiting_score, 1),
            'red_flags': self.red_flags,
            'green_flags': self.green_flags,
            'summary': self.summary,
            'detailed_analysis': {
                'engagement': self.engagement_report,
                'consistency': self.consistency_report,
                'dissonance': self.dissonance_report,
                'baiting': self.baiting_report
            }
        }


class CredibilityEngine:
    """
    Combines all analysis modules to generate a final credibility score.

    Scoring weights:
    - Engagement: 0.20
    - Consistency: 0.25
    - Dissonance: 0.25 (hypocrisy + authenticity) / 2
    - Baiting: 0.30

    Grade thresholds:
    - A: 85+
    - B: 70-84
    - C: 55-69
    - D: 40-54
    - F: <40
    """

    # Default weights
    DEFAULT_WEIGHTS = {
        'engagement': 0.20,
        'consistency': 0.25,
        'dissonance': 0.25,
        'baiting': 0.30
    }

    # Grade thresholds
    GRADE_THRESHOLDS = {
        'A': 85.0,
        'B': 70.0,
        'C': 55.0,
        'D': 40.0,
        'F': 0.0
    }

    # Assessment descriptions
    ASSESSMENTS = {
        'A': 'HIGH CREDIBILITY',
        'B': 'MODERATE CREDIBILITY',
        'C': 'MIXED SIGNALS',
        'D': 'LOW CREDIBILITY',
        'F': 'POOR CREDIBILITY'
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Initialize analyzers
        self.engagement_analyzer = EngagementAnalyzer()
        self.consistency_tracker = ConsistencyTracker()
        self.dissonance_analyzer = DissonanceAnalyzer()
        self.bait_analyzer = EngagementBaitAnalyzer()

    def analyze(
        self,
        tweets: List[dict],
        follower_count: int,
        username: str = ""
    ) -> CredibilityScore:
        """
        Perform complete credibility analysis.

        Args:
            tweets: List of tweet dictionaries
            follower_count: The KOL's follower count
            username: Optional username for context

        Returns:
            CredibilityScore with complete analysis
        """
        if not tweets:
            return CredibilityScore(
                overall_score=50.0,
                grade='C',
                confidence=0.0,
                assessment='INSUFFICIENT DATA',
                engagement_score=50.0,
                consistency_score=50.0,
                dissonance_score=50.0,
                baiting_score=50.0,
                summary="Insufficient data for analysis. Please provide more tweets."
            )

        # Run all analyzers
        engagement_profile = self.engagement_analyzer.analyze(tweets, follower_count)
        consistency_report = self.consistency_tracker.analyze(tweets)
        dissonance_report = self.dissonance_analyzer.analyze(tweets)
        baiting_report = self.bait_analyzer.analyze(tweets)

        # Extract scores
        engagement_score = engagement_profile.authenticity_score
        consistency_score = consistency_report.consistency_score
        dissonance_score = (dissonance_report.hypocrisy_score + dissonance_report.authenticity_score) / 2
        baiting_score = baiting_report.authenticity_score

        # Calculate weighted overall score
        overall_score = (
            engagement_score * self.weights['engagement'] +
            consistency_score * self.weights['consistency'] +
            dissonance_score * self.weights['dissonance'] +
            baiting_score * self.weights['baiting']
        )

        # Determine grade
        grade = self._calculate_grade(overall_score)
        assessment = self.ASSESSMENTS.get(grade, 'UNKNOWN')

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(len(tweets), follower_count)

        # Collect red flags
        red_flags = self._collect_red_flags(
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report
        )

        # Collect green flags
        green_flags = self._collect_green_flags(
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report
        )

        # Generate summary
        summary = self._generate_summary(
            overall_score,
            grade,
            username,
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report
        )

        return CredibilityScore(
            overall_score=overall_score,
            grade=grade,
            confidence=confidence,
            assessment=assessment,
            engagement_score=engagement_score,
            consistency_score=consistency_score,
            dissonance_score=dissonance_score,
            baiting_score=baiting_score,
            red_flags=red_flags,
            green_flags=green_flags,
            summary=summary,
            engagement_report=engagement_profile.to_dict(),
            consistency_report=consistency_report.to_dict(),
            dissonance_report=dissonance_report.to_dict(),
            baiting_report=baiting_report.to_dict()
        )

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= self.GRADE_THRESHOLDS['A']:
            return 'A'
        elif score >= self.GRADE_THRESHOLDS['B']:
            return 'B'
        elif score >= self.GRADE_THRESHOLDS['C']:
            return 'C'
        elif score >= self.GRADE_THRESHOLDS['D']:
            return 'D'
        else:
            return 'F'

    def _calculate_confidence(self, tweet_count: int, follower_count: int) -> float:
        """
        Calculate confidence level in the assessment.

        Based on amount of data available.
        """
        confidence = 0.0

        # Tweet count factor (up to 50%)
        if tweet_count >= 500:
            confidence += 50.0
        elif tweet_count >= 200:
            confidence += 40.0
        elif tweet_count >= 100:
            confidence += 30.0
        elif tweet_count >= 50:
            confidence += 20.0
        else:
            confidence += 10.0

        # Follower count factor (meaningful sample size) - up to 30%
        if follower_count >= 100000:
            confidence += 30.0
        elif follower_count >= 50000:
            confidence += 25.0
        elif follower_count >= 10000:
            confidence += 20.0
        elif follower_count >= 1000:
            confidence += 15.0
        else:
            confidence += 10.0

        # Base confidence for having any data (20%)
        confidence += 20.0

        return min(100.0, confidence)

    def _collect_red_flags(
        self,
        engagement: EngagementProfile,
        consistency: ConsistencyReport,
        dissonance: DissonanceReport,
        baiting: EngagementBaitReport
    ) -> List[str]:
        """Collect all red flags from analysis."""
        red_flags = []

        # Engagement red flags
        red_flags.extend(engagement.suspicious_patterns)

        # Consistency red flags
        if consistency.major_flip_count > 0:
            red_flags.append(
                f"Major position flips detected ({consistency.major_flip_count} instances)"
            )
        if consistency.unacknowledged_flips > 2:
            red_flags.append(
                f"Multiple unacknowledged position changes ({consistency.unacknowledged_flips})"
            )

        # Dissonance red flags
        red_flags.extend(dissonance.two_faced_indicators)
        if dissonance.power_dynamic_violations:
            red_flags.append(
                f"Power dynamic violations ({len(dissonance.power_dynamic_violations)} instances)"
            )
        if dissonance.derisive_percentage > 15:
            red_flags.append(
                f"High derisive content ({dissonance.derisive_percentage:.1f}%)"
            )

        # Baiting red flags
        if baiting.engagement_reward_optimization:
            red_flags.append("Active Kaito/reward platform participant - incentive-driven content")
        if baiting.bait_type_counts.get('fomo_manufacturing', 0) > 5:
            red_flags.append(
                f"Frequent FOMO tactics ({baiting.bait_type_counts['fomo_manufacturing']} instances)"
            )
        if baiting.bait_type_counts.get('engagement_farming', 0) > 10:
            red_flags.append(
                f"Heavy engagement farming ({baiting.bait_type_counts['engagement_farming']} instances)"
            )
        if baiting.manipulation_index > 50:
            red_flags.append(f"High manipulation index ({baiting.manipulation_index:.0f}/100)")

        return red_flags[:10]  # Limit to top 10

    def _collect_green_flags(
        self,
        engagement: EngagementProfile,
        consistency: ConsistencyReport,
        dissonance: DissonanceReport,
        baiting: EngagementBaitReport
    ) -> List[str]:
        """Collect positive indicators from analysis."""
        green_flags = []

        # Engagement green flags
        if engagement.authenticity_score >= 80:
            green_flags.append("Healthy engagement patterns")
        if engagement.engagement_consistency > 0.3:
            green_flags.append("Natural engagement variance")

        # Consistency green flags
        if consistency.flip_count == 0:
            green_flags.append("Consistent positions over time")
        if consistency.acknowledged_flips > 0 and consistency.unacknowledged_flips == 0:
            green_flags.append("Transparently acknowledges position changes")
        if consistency.consistency_score >= 90:
            green_flags.append("Highly consistent messaging")

        # Dissonance green flags
        if dissonance.authenticity_score >= 80:
            green_flags.append("Authentic communication style")
        if dissonance.primary_tone == 'instructional':
            green_flags.append("Primarily instructional tone")
        if dissonance.primary_tone == 'inclusive':
            green_flags.append("Welcoming and inclusive tone")
        if len(dissonance.hypocrisy_instances) == 0:
            green_flags.append("No hypocrisy detected")

        # Baiting green flags
        if baiting.authenticity_score >= 85:
            green_flags.append("Genuine engagement patterns")
        if baiting.manipulation_index < 15:
            green_flags.append("Low manipulation tactics")
        if not baiting.engagement_reward_optimization:
            green_flags.append("Not optimizing for reward platforms")

        return green_flags[:8]  # Limit to top 8

    def _generate_summary(
        self,
        overall_score: float,
        grade: str,
        username: str,
        engagement: EngagementProfile,
        consistency: ConsistencyReport,
        dissonance: DissonanceReport,
        baiting: EngagementBaitReport
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        name = f"@{username}" if username else "This KOL"

        # Opening statement based on grade
        if grade == 'A':
            opening = f"{name} shows strong credibility signals across all metrics."
        elif grade == 'B':
            opening = f"{name} demonstrates moderate credibility with minor concerns."
        elif grade == 'C':
            opening = f"{name} shows mixed signals with notable areas of concern."
        elif grade == 'D':
            opening = f"{name} has multiple credibility concerns that warrant caution."
        else:
            opening = f"{name} shows significant credibility issues across multiple areas."

        # Key insights
        insights = []

        # Engagement insight
        if engagement.authenticity_score < 60:
            insights.append("engagement patterns suggest artificial activity")
        elif engagement.authenticity_score >= 80:
            insights.append("engagement appears organic")

        # Consistency insight
        if consistency.flip_count > 3:
            insights.append(f"detected {consistency.flip_count} position changes")
        elif consistency.consistency_score >= 90:
            insights.append("maintains consistent positions")

        # Tone insight
        if dissonance.derisive_percentage > 10:
            insights.append("notable derisive content")
        elif dissonance.primary_tone == 'instructional':
            insights.append("primarily educational content")

        # Baiting insight
        if baiting.engagement_reward_optimization:
            insights.append("actively participates in reward platforms")
        if baiting.manipulation_index > 40:
            insights.append("uses frequent engagement tactics")

        # Combine
        if insights:
            detail = "Analysis shows: " + ", ".join(insights) + "."
        else:
            detail = ""

        # Recommendation
        if grade in ['A', 'B']:
            recommendation = "Content can generally be trusted with normal due diligence."
        elif grade == 'C':
            recommendation = "Recommend additional verification before acting on advice."
        else:
            recommendation = "Exercise significant caution with this account's recommendations."

        return f"{opening} {detail} {recommendation}".strip()

    def compare(
        self,
        score1: CredibilityScore,
        score2: CredibilityScore,
        username1: str,
        username2: str
    ) -> Dict[str, Any]:
        """
        Compare two KOLs' credibility scores.

        Returns a comparison dictionary.
        """
        comparison = {
            'overall': {
                username1: score1.overall_score,
                username2: score2.overall_score,
                'winner': username1 if score1.overall_score > score2.overall_score else username2,
                'difference': abs(score1.overall_score - score2.overall_score)
            },
            'engagement': {
                username1: score1.engagement_score,
                username2: score2.engagement_score,
                'winner': username1 if score1.engagement_score > score2.engagement_score else username2
            },
            'consistency': {
                username1: score1.consistency_score,
                username2: score2.consistency_score,
                'winner': username1 if score1.consistency_score > score2.consistency_score else username2
            },
            'dissonance': {
                username1: score1.dissonance_score,
                username2: score2.dissonance_score,
                'winner': username1 if score1.dissonance_score > score2.dissonance_score else username2
            },
            'baiting': {
                username1: score1.baiting_score,
                username2: score2.baiting_score,
                'winner': username1 if score1.baiting_score > score2.baiting_score else username2
            },
            'grades': {
                username1: score1.grade,
                username2: score2.grade
            },
            'summary': self._generate_comparison_summary(
                score1, score2, username1, username2
            )
        }

        return comparison

    def _generate_comparison_summary(
        self,
        score1: CredibilityScore,
        score2: CredibilityScore,
        username1: str,
        username2: str
    ) -> str:
        """Generate a comparison summary."""
        diff = score1.overall_score - score2.overall_score

        if abs(diff) < 5:
            return f"@{username1} and @{username2} have similar credibility scores."
        elif diff > 0:
            return f"@{username1} scores {abs(diff):.1f} points higher than @{username2}."
        else:
            return f"@{username2} scores {abs(diff):.1f} points higher than @{username1}."
