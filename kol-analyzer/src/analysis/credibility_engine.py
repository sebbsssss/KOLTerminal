"""
Credibility Engine - Combine all analysis scores into a final credibility score.

Enhanced with 13 analysis modules:
- Privilege/High Horse Analysis
- Prediction Accuracy Tracking
- Sponsored Content Detection
- Follower Quality Analysis
- Archetype Classification
- Temporal Analysis (timing vs price action)
- Linguistic Authenticity Analysis
- Accountability Tracking
- Network/Reply Pattern Analysis
- Reputation Analysis (what others say about the KOL)
"""

import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


def run_async_safely(coro):
    """Run an async coroutine safely, handling nested event loops."""
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # We're in an async context - run in a thread pool
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()

from .engagement_analyzer import EngagementAnalyzer, EngagementProfile
from .consistency_tracker import ConsistencyTracker, ConsistencyReport
from .dissonance_analyzer import DissonanceAnalyzer, DissonanceReport
from .engagement_bait_analyzer import EngagementBaitAnalyzer, EngagementBaitReport
from .privilege_analyzer import PrivilegeAnalyzer, PrivilegeReport
from .prediction_tracker import PredictionTracker, PredictionReport
from .sponsored_detector import SponsoredDetector, SponsoredReport
from .follower_quality import FollowerQualityAnalyzer, FollowerQualityReport
from .archetype_classifier import ArchetypeClassifier, ArchetypeProfile
from .temporal_analyzer import TemporalAnalyzer, TemporalReport
from .linguistic_analyzer import LinguisticAnalyzer, LinguisticReport
from .accountability_tracker import AccountabilityTracker, AccountabilityReport
from .network_analyzer import NetworkAnalyzer, NetworkReport
from .reputation_analyzer import ReputationAnalyzer, ReputationReport
from .asshole_analyzer import AssholeAnalyzer, AssholeAnalysis


@dataclass
class CredibilityScore:
    """Final credibility score and assessment."""
    overall_score: float  # 0-100
    grade: str  # A, B, C, D, F
    confidence: float  # How confident in assessment (0-100)
    assessment: str  # "HIGH CREDIBILITY", etc.

    # Core scores
    engagement_score: float
    consistency_score: float
    dissonance_score: float
    baiting_score: float

    # New enhanced scores
    privilege_score: float = 50.0
    prediction_score: float = 50.0
    transparency_score: float = 50.0
    follower_quality_score: float = 50.0

    # Additional depth scores (13 total modules)
    temporal_score: float = 50.0
    linguistic_score: float = 50.0
    accountability_score: float = 50.0
    network_score: float = 50.0
    reputation_score: float = 50.0  # What others say about the KOL

    # Asshole Meter (personality analysis - separate from credibility)
    asshole_score: float = 50.0  # 0 = saint, 100 = toxic
    toxicity_level: str = "mid"  # saint, chill, mid, prickly, toxic
    toxicity_emoji: str = "😐"

    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    summary: str = ""

    # Detailed reports
    engagement_report: Optional[Dict] = None
    consistency_report: Optional[Dict] = None
    dissonance_report: Optional[Dict] = None
    baiting_report: Optional[Dict] = None
    privilege_report: Optional[Dict] = None
    prediction_report: Optional[Dict] = None
    sponsored_report: Optional[Dict] = None
    follower_quality_report: Optional[Dict] = None
    temporal_report: Optional[Dict] = None
    linguistic_report: Optional[Dict] = None
    accountability_report: Optional[Dict] = None
    network_report: Optional[Dict] = None
    reputation_report: Optional[Dict] = None
    asshole_report: Optional[Dict] = None

    # Archetype classification
    archetype: Optional[str] = None
    archetype_emoji: Optional[str] = None
    archetype_one_liner: Optional[str] = None
    trust_level: Optional[str] = None
    archetype_report: Optional[Dict] = None

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
            'privilege_score': round(self.privilege_score, 1),
            'prediction_score': round(self.prediction_score, 1),
            'transparency_score': round(self.transparency_score, 1),
            'follower_quality_score': round(self.follower_quality_score, 1),
            'temporal_score': round(self.temporal_score, 1),
            'linguistic_score': round(self.linguistic_score, 1),
            'accountability_score': round(self.accountability_score, 1),
            'network_score': round(self.network_score, 1),
            'reputation_score': round(self.reputation_score, 1),
            'asshole_score': round(self.asshole_score, 1),
            'toxicity_level': self.toxicity_level,
            'toxicity_emoji': self.toxicity_emoji,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags,
            'summary': self.summary,
            'archetype': self.archetype,
            'archetype_emoji': self.archetype_emoji,
            'archetype_one_liner': self.archetype_one_liner,
            'trust_level': self.trust_level,
            'detailed_analysis': {
                'engagement': self.engagement_report,
                'consistency': self.consistency_report,
                'dissonance': self.dissonance_report,
                'baiting': self.baiting_report,
                'privilege': self.privilege_report,
                'prediction': self.prediction_report,
                'sponsored': self.sponsored_report,
                'follower_quality': self.follower_quality_report,
                'temporal': self.temporal_report,
                'linguistic': self.linguistic_report,
                'accountability': self.accountability_report,
                'network': self.network_report,
                'reputation': self.reputation_report,
                'archetype': self.archetype_report
            }
        }


class CredibilityEngine:
    """
    Combines all analysis modules to generate a final credibility score.

    Enhanced scoring weights (13 modules):
    - Engagement: 0.07
    - Consistency: 0.09
    - Dissonance: 0.07
    - Baiting: 0.09
    - Privilege: 0.08 (moral high horse detection)
    - Prediction: 0.09 (track record accuracy)
    - Transparency: 0.07 (sponsored content disclosure)
    - Follower Quality: 0.07 (audience authenticity)
    - Temporal: 0.07 (timing vs price action)
    - Linguistic: 0.07 (language authenticity)
    - Accountability: 0.07 (owns mistakes)
    - Network: 0.05 (interaction patterns)
    - Reputation: 0.11 (what others say - heavily weighted!)

    Grade thresholds:
    - A: 85+
    - B: 70-84
    - C: 55-69
    - D: 40-54
    - F: <40
    """

    # Enhanced weights for 13 modules (totals 1.00)
    DEFAULT_WEIGHTS = {
        'engagement': 0.07,
        'consistency': 0.09,
        'dissonance': 0.07,
        'baiting': 0.09,
        'privilege': 0.08,
        'prediction': 0.09,
        'transparency': 0.07,
        'follower_quality': 0.07,
        'temporal': 0.07,
        'linguistic': 0.07,
        'accountability': 0.07,
        'network': 0.05,
        'reputation': 0.11  # What others say - heavily weighted
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

        # Initialize all analyzers (original)
        self.engagement_analyzer = EngagementAnalyzer()
        self.consistency_tracker = ConsistencyTracker()
        self.dissonance_analyzer = DissonanceAnalyzer()
        self.bait_analyzer = EngagementBaitAnalyzer()

        # Initialize enhanced analyzers
        self.privilege_analyzer = PrivilegeAnalyzer()
        self.prediction_tracker = PredictionTracker()
        self.sponsored_detector = SponsoredDetector()
        self.follower_quality_analyzer = FollowerQualityAnalyzer()

        # Initialize depth analyzers
        self.temporal_analyzer = TemporalAnalyzer()
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.accountability_tracker = AccountabilityTracker()
        self.network_analyzer = NetworkAnalyzer()

        # Initialize reputation analyzer (what others say)
        self.reputation_analyzer = ReputationAnalyzer()

        # Initialize asshole analyzer (personality/toxicity)
        self.asshole_analyzer = AssholeAnalyzer()

        # Archetype classifier
        self.archetype_classifier = ArchetypeClassifier()

    def analyze(
        self,
        tweets: List[dict],
        follower_count: int,
        username: str = "",
        account_age_days: int = 365,
        mentions: Optional[List[dict]] = None
    ) -> CredibilityScore:
        """
        Perform complete credibility analysis with enhanced modules.

        Args:
            tweets: List of tweet dictionaries
            follower_count: The KOL's follower count
            username: Optional username for context
            account_age_days: Account age in days (for privilege analysis)
            mentions: Optional list of tweets mentioning the KOL from other users

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
                privilege_score=50.0,
                prediction_score=50.0,
                transparency_score=50.0,
                follower_quality_score=50.0,
                summary="Insufficient data for analysis. Please provide more tweets."
            )

        # Run original analyzers
        engagement_profile = self.engagement_analyzer.analyze(tweets, follower_count)
        consistency_report = self.consistency_tracker.analyze(tweets)
        dissonance_report = self.dissonance_analyzer.analyze(tweets)
        baiting_report = self.bait_analyzer.analyze(tweets)

        # Run new analyzers
        privilege_report = self.privilege_analyzer.analyze(
            tweets, follower_count, account_age_days
        )
        sponsored_report = self.sponsored_detector.analyze(tweets)
        follower_quality_report = self.follower_quality_analyzer.analyze(
            tweets, follower_count
        )

        # Run prediction tracker (async) and temporal analyzer (async)
        prediction_report = run_async_safely(self.prediction_tracker.analyze(tweets))
        temporal_report = run_async_safely(self.temporal_analyzer.analyze(tweets))

        # Run depth analyzers (sync)
        linguistic_report = self.linguistic_analyzer.analyze(tweets)
        accountability_report = self.accountability_tracker.analyze(tweets)
        network_report = self.network_analyzer.analyze(tweets)

        # Run reputation analyzer (what others say about the KOL)
        if mentions:
            reputation_report = self.reputation_analyzer.analyze(mentions, username)
        else:
            # Use demo mode if no mentions provided
            reputation_report = self.reputation_analyzer.analyze_demo(username)

        # Run asshole analyzer (personality/toxicity meter)
        asshole_report = self.asshole_analyzer.analyze(tweets, username)

        # Extract scores from original analyzers
        engagement_score = engagement_profile.authenticity_score
        consistency_score = consistency_report.consistency_score
        dissonance_score = (dissonance_report.hypocrisy_score + dissonance_report.authenticity_score) / 2
        baiting_score = baiting_report.authenticity_score

        # Extract scores from enhanced analyzers
        privilege_score = (privilege_report.privilege_score + privilege_report.empathy_score) / 2
        prediction_score = prediction_report.accuracy_score
        transparency_score = sponsored_report.transparency_score
        follower_quality_score = follower_quality_report.quality_score

        # Extract scores from depth analyzers
        temporal_score = temporal_report.timing_score
        linguistic_score = linguistic_report.authenticity_score
        accountability_score = accountability_report.accountability_score
        network_score = network_report.network_score
        reputation_score = reputation_report.reputation_score

        # Calculate weighted overall score (13 modules)
        overall_score = (
            engagement_score * self.weights['engagement'] +
            consistency_score * self.weights['consistency'] +
            dissonance_score * self.weights['dissonance'] +
            baiting_score * self.weights['baiting'] +
            privilege_score * self.weights['privilege'] +
            prediction_score * self.weights['prediction'] +
            transparency_score * self.weights['transparency'] +
            follower_quality_score * self.weights['follower_quality'] +
            temporal_score * self.weights['temporal'] +
            linguistic_score * self.weights['linguistic'] +
            accountability_score * self.weights['accountability'] +
            network_score * self.weights['network'] +
            reputation_score * self.weights['reputation']
        )

        # Determine grade
        grade = self._calculate_grade(overall_score)
        assessment = self.ASSESSMENTS.get(grade, 'UNKNOWN')

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(len(tweets), follower_count)

        # Collect red flags from all modules
        red_flags = self._collect_red_flags(
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report,
            privilege_report,
            prediction_report,
            sponsored_report,
            follower_quality_report,
            temporal_report,
            linguistic_report,
            accountability_report,
            network_report,
            reputation_report
        )

        # Collect green flags from all modules
        green_flags = self._collect_green_flags(
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report,
            privilege_report,
            prediction_report,
            sponsored_report,
            follower_quality_report,
            temporal_report,
            linguistic_report,
            accountability_report,
            network_report,
            reputation_report
        )

        # Filter out contradictory flags to prevent confusing results
        red_flags, green_flags = self._filter_contradictory_flags(red_flags, green_flags)

        # Classify archetype
        archetype_profile = self.archetype_classifier.classify(
            engagement_score=engagement_score,
            consistency_score=consistency_score,
            dissonance_score=dissonance_score,
            baiting_score=baiting_score,
            privilege_score=privilege_score,
            prediction_score=prediction_score,
            transparency_score=transparency_score,
            follower_quality_score=follower_quality_score,
            follower_count=len(tweets) * 100,  # Estimate if not provided
            tweet_count=len(tweets),
            account_age_days=account_age_days,
            privilege_report=privilege_report.to_dict(),
            baiting_report=baiting_report.to_dict(),
            sponsored_report=sponsored_report.to_dict(),
            prediction_report=prediction_report.to_dict()
        )

        # Get archetype metadata
        from .archetype_classifier import ARCHETYPE_DEFINITIONS
        archetype_def = ARCHETYPE_DEFINITIONS.get(archetype_profile.primary_archetype, {})

        # Generate summary
        summary = self._generate_summary(
            overall_score,
            grade,
            username,
            engagement_profile,
            consistency_report,
            dissonance_report,
            baiting_report,
            privilege_report,
            prediction_report
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
            privilege_score=privilege_score,
            prediction_score=prediction_score,
            transparency_score=transparency_score,
            follower_quality_score=follower_quality_score,
            temporal_score=temporal_score,
            linguistic_score=linguistic_score,
            accountability_score=accountability_score,
            network_score=network_score,
            reputation_score=reputation_score,
            asshole_score=asshole_report.asshole_score,
            toxicity_level=asshole_report.toxicity_level,
            toxicity_emoji=asshole_report.toxicity_emoji,
            red_flags=red_flags,
            green_flags=green_flags,
            summary=summary,
            archetype=archetype_def.get("name", "Unknown"),
            archetype_emoji=archetype_def.get("emoji", "❓"),
            archetype_one_liner=archetype_profile.one_liner,
            trust_level=archetype_profile.trust_level.value,
            engagement_report=engagement_profile.to_dict(),
            consistency_report=consistency_report.to_dict(),
            dissonance_report=dissonance_report.to_dict(),
            baiting_report=baiting_report.to_dict(),
            privilege_report=privilege_report.to_dict(),
            prediction_report=prediction_report.to_dict(),
            sponsored_report=sponsored_report.to_dict(),
            follower_quality_report=follower_quality_report.to_dict(),
            temporal_report=temporal_report.to_dict(),
            linguistic_report=linguistic_report.to_dict(),
            accountability_report=accountability_report.to_dict(),
            network_report=network_report.to_dict(),
            reputation_report=reputation_report.to_dict(),
            asshole_report=asshole_report.to_dict(),
            archetype_report=archetype_profile.to_dict()
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
        baiting: EngagementBaitReport,
        privilege: PrivilegeReport = None,
        prediction: PredictionReport = None,
        sponsored: SponsoredReport = None,
        follower_quality: FollowerQualityReport = None,
        temporal: TemporalReport = None,
        linguistic: LinguisticReport = None,
        accountability: AccountabilityReport = None,
        network: NetworkReport = None,
        reputation: ReputationReport = None
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

        # Privilege/High Horse red flags
        if privilege:
            red_flags.extend(privilege.high_horse_indicators[:3])
            if privilege.privilege_score < 50:
                red_flags.append("Significant privilege blindness detected")
            if privilege.empathy_score < 40:
                red_flags.append("Low empathy for those still struggling")

        # Prediction accuracy red flags
        if prediction:
            if prediction.hit_rate < 35 and prediction.total_calls >= 5:
                red_flags.append(f"Poor prediction accuracy ({prediction.hit_rate:.0f}% hit rate)")
            if prediction.confidence_calibration == "overconfident":
                red_flags.append("Overconfident on 'guaranteed' calls that often fail")

        # Sponsored content red flags
        if sponsored:
            red_flags.extend(sponsored.red_flags[:2])
            if sponsored.disclosure_rate < 50 and sponsored.total_promotional > 3:
                red_flags.append(f"Low disclosure rate on promotions ({sponsored.disclosure_rate:.0f}%)")

        # Follower quality red flags
        if follower_quality:
            red_flags.extend(follower_quality.red_flags[:2])
            if follower_quality.bot_follower_estimate_pct > 30:
                red_flags.append(f"High estimated bot followers ({follower_quality.bot_follower_estimate_pct:.0f}%)")

        # Temporal red flags
        if temporal:
            red_flags.extend(temporal.red_flags[:2])
            if temporal.front_run_score > 60:
                red_flags.append(f"Potential front-running behavior ({temporal.front_run_score:.0f}% suspicious timing)")

        # Linguistic red flags
        if linguistic:
            red_flags.extend(linguistic.red_flags[:2])
            if linguistic.manipulation_score > 60:
                red_flags.append("High linguistic manipulation patterns detected")
            if linguistic.avg_certainty_level > 80:
                red_flags.append("Overuse of certainty language (\"guaranteed\", \"100%\")")

        # Accountability red flags
        if accountability:
            red_flags.extend(accountability.red_flags[:2])
            if accountability.cherry_picks_wins:
                red_flags.append("Only posts wins, hides losses (cherry-picking)")
            if accountability.deflection_count > 3:
                red_flags.append(f"Deflects blame frequently ({accountability.deflection_count} instances)")

        # Network red flags
        if network:
            red_flags.extend(network.red_flags[:2])
            if network.reply_guy_score > 70:
                red_flags.append("Heavy reply guy behavior - little original content")
            if network.potential_shill_ring:
                red_flags.append("Part of mutual promotion network (potential shill ring)")

        # Reputation red flags (what others say)
        if reputation:
            red_flags.extend(reputation.red_flags[:3])  # Include up to 3 reputation flags

        return red_flags[:15]  # Limit to top 15

    def _collect_green_flags(
        self,
        engagement: EngagementProfile,
        consistency: ConsistencyReport,
        dissonance: DissonanceReport,
        baiting: EngagementBaitReport,
        privilege: PrivilegeReport = None,
        prediction: PredictionReport = None,
        sponsored: SponsoredReport = None,
        follower_quality: FollowerQualityReport = None,
        temporal: TemporalReport = None,
        linguistic: LinguisticReport = None,
        accountability: AccountabilityReport = None,
        network: NetworkReport = None,
        reputation: ReputationReport = None
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

        # Privilege/Empathy green flags
        if privilege:
            green_flags.extend(privilege.empathy_indicators[:2])
            if privilege.privilege_score >= 80:
                green_flags.append("Shows awareness of their privileged position")
            if privilege.empathy_score >= 80:
                green_flags.append("Demonstrates empathy for those still struggling")

        # Prediction accuracy green flags
        if prediction:
            if prediction.hit_rate >= 60 and prediction.total_calls >= 5:
                green_flags.append(f"Strong prediction track record ({prediction.hit_rate:.0f}% hit rate)")
            if prediction.confidence_calibration == "well-calibrated":
                green_flags.append("Well-calibrated confidence on predictions")

        # Sponsored content green flags
        if sponsored:
            green_flags.extend(sponsored.green_flags[:2])
            if sponsored.disclosure_rate >= 90:
                green_flags.append("Excellent transparency on sponsored content")

        # Follower quality green flags
        if follower_quality:
            green_flags.extend(follower_quality.green_flags[:2])
            if follower_quality.quality_score >= 75:
                green_flags.append("High-quality authentic follower base")

        # Temporal green flags
        if temporal:
            green_flags.extend(temporal.green_flags[:2])
            if temporal.front_run_score < 20:
                green_flags.append("Timing patterns appear natural")
            if temporal.crash_sentiment == "supportive":
                green_flags.append("Supportive during market downturns")

        # Linguistic green flags
        if linguistic:
            green_flags.extend(linguistic.green_flags[:2])
            if linguistic.authenticity_score >= 80:
                green_flags.append("Natural, authentic language patterns")
            if linguistic.certainty_calibration == "well_calibrated":
                green_flags.append("Appropriately hedges uncertain predictions")

        # Accountability green flags
        if accountability:
            green_flags.extend(accountability.green_flags[:2])
            if accountability.takes_responsibility:
                green_flags.append("Takes responsibility for predictions")
            if accountability.correction_count > 0:
                green_flags.append("Publicly corrects mistakes")

        # Network green flags
        if network:
            green_flags.extend(network.green_flags[:2])
            if network.original_ratio > 0.7:
                green_flags.append("Primarily creates original content")
            if network.constructive_responses > network.defensive_responses:
                green_flags.append("Handles criticism constructively")

        # Reputation green flags (what others say)
        if reputation:
            green_flags.extend(reputation.green_flags[:2])

        return green_flags[:12]  # Limit to top 12

    def _filter_contradictory_flags(
        self,
        red_flags: List[str],
        green_flags: List[str]
    ) -> tuple[List[str], List[str]]:
        """
        Filter out contradictory flags to prevent confusing results.

        When red flags indicate problems in an area, suppress green flags
        that would claim the opposite is true.
        """
        filtered_green = green_flags.copy()

        # Define contradiction rules: (red_flag_pattern, green_flags_to_remove)
        contradictions = [
            # Engagement farming vs healthy engagement
            (
                "engagement farming",
                ["Healthy engagement patterns", "Genuine engagement patterns", "Natural engagement variance"]
            ),
            # FOMO/manipulation tactics contradictions
            (
                "FOMO tactics",
                ["Low manipulation tactics"]
            ),
            # High manipulation index contradicts low manipulation
            (
                "manipulation index",
                ["Low manipulation tactics"]
            ),
            # Position flips contradict "no hypocrisy" (they indicate inconsistency even if not technical hypocrisy)
            (
                "position flips detected",
                ["No hypocrisy detected", "Consistent positions over time"]
            ),
            # Unacknowledged position changes contradict transparency claims
            (
                "unacknowledged position changes",
                ["Transparently acknowledges position changes", "No hypocrisy detected"]
            ),
            # Derisive content contradicts authentic/instructional tone claims
            (
                "derisive",
                ["Authentic communication style", "Primarily instructional tone"]
            ),
            # Bot followers contradict quality follower base
            (
                "bot followers",
                ["High-quality authentic follower base"]
            ),
            # Front-running contradicts natural timing
            (
                "front-running",
                ["Timing patterns appear natural"]
            ),
            # Linguistic manipulation contradicts authentic language
            (
                "linguistic manipulation",
                ["Natural, authentic language patterns"]
            ),
            # Cherry-picking contradicts accountability
            (
                "cherry-pick",
                ["Takes responsibility for predictions", "Publicly corrects mistakes"]
            ),
            # Deflection contradicts accountability
            (
                "Deflects blame",
                ["Takes responsibility for predictions"]
            ),
            # Reward platform optimization contradicts genuine engagement
            (
                "Kaito/reward platform",
                ["Not optimizing for reward platforms", "Genuine engagement patterns"]
            ),
        ]

        # Check each red flag against contradiction rules
        for red_flag in red_flags:
            red_flag_lower = red_flag.lower()
            for pattern, greens_to_remove in contradictions:
                if pattern.lower() in red_flag_lower:
                    for green_to_remove in greens_to_remove:
                        if green_to_remove in filtered_green:
                            filtered_green.remove(green_to_remove)

        return red_flags, filtered_green

    def _generate_summary(
        self,
        overall_score: float,
        grade: str,
        username: str,
        engagement: EngagementProfile,
        consistency: ConsistencyReport,
        dissonance: DissonanceReport,
        baiting: EngagementBaitReport,
        privilege: PrivilegeReport = None,
        prediction: PredictionReport = None
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

        # Privilege insight (NEW)
        if privilege:
            if privilege.privilege_score < 50:
                insights.append("shows moral high horse tendencies")
            elif privilege.empathy_score >= 80:
                insights.append("shows empathy for newcomers")

        # Prediction insight (NEW)
        if prediction and prediction.total_calls >= 5:
            if prediction.hit_rate >= 60:
                insights.append(f"solid track record ({prediction.hit_rate:.0f}% accuracy)")
            elif prediction.hit_rate < 40:
                insights.append(f"poor prediction accuracy ({prediction.hit_rate:.0f}%)")

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
