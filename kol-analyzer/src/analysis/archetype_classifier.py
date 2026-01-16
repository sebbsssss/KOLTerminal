"""
Archetype Classifier - Classify KOLs into recognizable personas based on behavior patterns.

Instead of just scores, this gives users an intuitive understanding of WHO they're dealing with.

Enhanced with zero-shot classification for more nuanced archetype detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Archetype(Enum):
    """KOL personality archetypes."""
    THE_GURU = "the_guru"
    THE_GRINDER = "the_grinder"
    THE_SHILL = "the_shill"
    THE_ANALYST = "the_analyst"
    THE_ENTERTAINER = "the_entertainer"
    THE_FARMER = "the_farmer"
    THE_INSIDER = "the_insider"
    THE_NEWCOMER = "the_newcomer"
    UNKNOWN = "unknown"


class TrustLevel(Enum):
    """Trust recommendation levels."""
    AVOID = "avoid"
    ENTERTAINMENT_ONLY = "entertainment_only"
    CAUTION = "caution"
    NEUTRAL = "neutral"
    CONSIDER = "consider"
    HIGHER_TRUST = "higher_trust"


@dataclass
class ArchetypeProfile:
    """Complete archetype classification result."""
    primary_archetype: Archetype
    secondary_archetype: Optional[Archetype]
    confidence: float  # 0-100
    trust_level: TrustLevel
    archetype_scores: Dict[str, float]  # Score for each archetype

    # Narrative elements
    one_liner: str  # e.g., "Made it and forgot the struggle"
    detailed_description: str
    key_behaviors: List[str]
    watch_out_for: List[str]
    positive_traits: List[str]

    # Evolution insight
    evolution_stage: str  # "early", "rising", "established", "evolved", "declining"
    evolution_warning: Optional[str]  # e.g., "Shows early signs of guru syndrome"

    # ML-enhanced classification
    ml_available: bool = False
    ml_classification: Optional[Dict[str, float]] = None
    content_themes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            'primary_archetype': self.primary_archetype.value,
            'secondary_archetype': self.secondary_archetype.value if self.secondary_archetype else None,
            'confidence': round(self.confidence, 1),
            'trust_level': self.trust_level.value,
            'archetype_scores': {k: round(v, 1) for k, v in self.archetype_scores.items()},
            'one_liner': self.one_liner,
            'detailed_description': self.detailed_description,
            'key_behaviors': self.key_behaviors,
            'watch_out_for': self.watch_out_for,
            'positive_traits': self.positive_traits,
            'evolution_stage': self.evolution_stage,
            'evolution_warning': self.evolution_warning
        }

        if self.ml_available:
            result['ml_classification'] = self.ml_classification
            result['content_themes'] = self.content_themes

        return result


# Archetype definitions with trait weights
ARCHETYPE_DEFINITIONS = {
    Archetype.THE_GURU: {
        "name": "The Guru",
        "emoji": "🧘",
        "one_liner": "Made it and now lectures from the mountaintop",
        "description": (
            "Once a grinder, now dispenses wisdom from a position of comfort. "
            "Preaches patience while sitting on gains, tells you health is wealth "
            "without acknowledging the grind it takes to get there. Classic "
            "survivorship bias wrapped in wellness language."
        ),
        "trust_level": TrustLevel.CAUTION,
        "traits": {
            "privilege_low": 0.35,  # Low privilege score = high horse behavior
            "empathy_low": 0.25,
            "high_followers": 0.15,
            "patience_preaching": 0.15,
            "consistency_high": 0.10,
        },
        "watch_out_for": [
            "Advice that only works if you're already rich",
            "'Just be patient' from someone who doesn't need the money",
            "Survivorship bias disguised as wisdom",
            "Condescending wellness takes"
        ],
        "positive_traits": [
            "Usually has real experience (just forgot the struggle)",
            "Content is often well-intentioned",
            "Typically not actively malicious"
        ]
    },

    Archetype.THE_GRINDER: {
        "name": "The Grinder",
        "emoji": "⚒️",
        "one_liner": "Still in the trenches, shares the real journey",
        "description": (
            "Acknowledges luck and timing in their success. Posts losses alongside "
            "wins. Remembers what it was like to be new. Shows empathy for those "
            "still struggling. Doesn't pretend to have all the answers."
        ),
        "trust_level": TrustLevel.HIGHER_TRUST,
        "traits": {
            "empathy_high": 0.30,
            "privilege_high": 0.25,  # High privilege score = self-aware
            "transparency_high": 0.20,
            "acknowledges_luck": 0.15,
            "shows_losses": 0.10,
        },
        "watch_out_for": [
            "May still be learning and make mistakes",
            "Smaller following means less social proof",
            "Could evolve into Guru over time"
        ],
        "positive_traits": [
            "Honest about their journey",
            "Relatable content",
            "Likely to admit when wrong",
            "Empathetic to newcomers"
        ]
    },

    Archetype.THE_SHILL: {
        "name": "The Shill",
        "emoji": "📢",
        "one_liner": "Gets paid to pump, rarely discloses",
        "description": (
            "Promotes projects without disclosing compensation. New 'conviction' "
            "every week. Never posts about losses. Coordinated posting with other "
            "accounts. Uses audience as exit liquidity."
        ),
        "trust_level": TrustLevel.AVOID,
        "traits": {
            "transparency_low": 0.30,
            "high_promotion": 0.25,
            "undisclosed_count": 0.20,
            "baiting_high": 0.15,
            "prediction_low": 0.10,
        },
        "watch_out_for": [
            "Undisclosed paid promotions",
            "Exit liquidity for their bags",
            "Coordinated pump schemes",
            "No accountability for failed calls"
        ],
        "positive_traits": [
            "Sometimes early on trends (even if paid)",
            "High engagement can surface info"
        ]
    },

    Archetype.THE_ANALYST: {
        "name": "The Analyst",
        "emoji": "📊",
        "one_liner": "Does the research, tracks the record",
        "description": (
            "Actually looks at data before posting. References on-chain metrics, "
            "fundamentals, and historical patterns. Tracks their own predictions "
            "and revisits old calls. Hedges appropriately and admits uncertainty."
        ),
        "trust_level": TrustLevel.HIGHER_TRUST,
        "traits": {
            "prediction_high": 0.30,
            "consistency_high": 0.25,
            "transparency_high": 0.20,
            "baiting_low": 0.15,
            "technical_content": 0.10,
        },
        "watch_out_for": [
            "Analysis paralysis - may miss momentum plays",
            "Can be wrong despite good process",
            "May be too cautious in bull markets"
        ],
        "positive_traits": [
            "Shows their work",
            "Tracks their record",
            "Admits mistakes",
            "Data-driven approach"
        ]
    },

    Archetype.THE_ENTERTAINER: {
        "name": "The Entertainer",
        "emoji": "🎭",
        "one_liner": "Optimizes for engagement, not truth",
        "description": (
            "Hot takes, rage bait, and controversial opinions drive their content. "
            "High engagement but low signal. Rarely follows up on predictions. "
            "Entertainment value high, alpha value low."
        ),
        "trust_level": TrustLevel.ENTERTAINMENT_ONLY,
        "traits": {
            "baiting_high": 0.35,
            "engagement_high": 0.25,
            "prediction_low": 0.15,
            "consistency_low": 0.15,
            "hot_takes": 0.10,
        },
        "watch_out_for": [
            "Takes designed for engagement, not accuracy",
            "Controversial for the sake of it",
            "No accountability for predictions",
            "Optimizes for algorithm, not truth"
        ],
        "positive_traits": [
            "Can be genuinely funny/entertaining",
            "Good for pulse on CT sentiment",
            "Sometimes contrarian takes are right"
        ]
    },

    Archetype.THE_FARMER: {
        "name": "The Farmer",
        "emoji": "🌾",
        "one_liner": "Playing the points game, not sharing conviction",
        "description": (
            "Content optimized for Kaito points, Galxe quests, and airdrop farming. "
            "Engagement bait heavy. Posts what the algorithm wants, not what they "
            "believe. Gaming metrics, not providing value."
        ),
        "trust_level": TrustLevel.AVOID,
        "traits": {
            "reward_optimization": 0.35,
            "baiting_high": 0.25,
            "engagement_farming": 0.20,
            "low_substance": 0.10,
            "high_frequency": 0.10,
        },
        "watch_out_for": [
            "Content is for points, not conviction",
            "Will promote anything for rewards",
            "No genuine thesis or belief",
            "Gaming every possible metric"
        ],
        "positive_traits": [
            "Shows what's being incentivized in ecosystem",
            "Sometimes surfaces airdrop opportunities"
        ]
    },

    Archetype.THE_INSIDER: {
        "name": "The Insider",
        "emoji": "🔮",
        "one_liner": "Knows things but won't say how",
        "description": (
            "Mysteriously early on projects. Connected to teams but doesn't disclose. "
            "Predictions are 'too good' - either has insider info or is coordinating. "
            "Conflicted interests are the norm."
        ),
        "trust_level": TrustLevel.CAUTION,
        "traits": {
            "early_calls": 0.30,
            "project_connections": 0.25,
            "prediction_high": 0.20,
            "transparency_low": 0.15,
            "undisclosed_advisory": 0.10,
        },
        "watch_out_for": [
            "Undisclosed conflicts of interest",
            "May be using you as exit liquidity",
            "Info asymmetry works against you",
            "'Alpha' may be coordination, not insight"
        ],
        "positive_traits": [
            "Often actually early on things",
            "Connected in the ecosystem",
            "Can surface real opportunities"
        ]
    },

    Archetype.THE_NEWCOMER: {
        "name": "The Newcomer",
        "emoji": "🌱",
        "one_liner": "Still finding their voice, not enough data",
        "description": (
            "New to the space or small following. Not enough history to classify. "
            "Could evolve into any archetype. Worth watching but not enough signal yet."
        ),
        "trust_level": TrustLevel.NEUTRAL,
        "traits": {
            "low_history": 0.40,
            "small_following": 0.30,
            "inconsistent_patterns": 0.20,
            "learning_signals": 0.10,
        },
        "watch_out_for": [
            "Not enough history to judge",
            "Could be anyone at this stage",
            "May be alt account of established player"
        ],
        "positive_traits": [
            "Fresh perspective possible",
            "Not yet corrupted by incentives",
            "May have genuine insights"
        ]
    }
}


class ArchetypeClassifier:
    """
    Classifies KOLs into archetypes based on their analysis scores.

    Enhanced with zero-shot classification for content-based archetype detection.
    """

    # Labels for zero-shot classification
    CONTENT_LABELS = [
        "educational content",
        "market analysis",
        "promotional shilling",
        "entertainment and memes",
        "personal journey sharing",
        "insider alpha leaks",
        "airdrop farming content",
        "philosophical wisdom"
    ]

    ARCHETYPE_LABELS = [
        "guru dispensing wisdom",
        "honest grinder in trenches",
        "paid shill promoter",
        "data-driven analyst",
        "entertainer and memer",
        "points farmer optimizer",
        "connected insider",
        "newcomer learning"
    ]

    def __init__(self, use_ml: bool = True):
        """
        Initialize the classifier.

        Args:
            use_ml: Whether to use ML models for classification
        """
        self.use_ml = use_ml
        self._ml_available = None

    def _check_ml_available(self) -> bool:
        """Check if zero-shot classifier is available."""
        if self._ml_available is None:
            try:
                from .ml_models import is_model_available
                self._ml_available = is_model_available('zero_shot')
            except ImportError:
                self._ml_available = False
        return self._ml_available

    def _classify_content_zero_shot(
        self,
        sample_tweets: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Classify content using zero-shot classification.

        Args:
            sample_tweets: List of sample tweet texts

        Returns:
            Dict with classification results or None
        """
        if not self.use_ml or not self._check_ml_available():
            return None

        if not sample_tweets:
            return None

        try:
            from .ml_models import classify_zero_shot

            # Combine sample tweets for classification
            combined = ' '.join(sample_tweets[:20])  # Limit to 20 tweets
            if len(combined) > 2000:
                combined = combined[:2000]

            # Classify content themes
            content_result = classify_zero_shot(combined, self.CONTENT_LABELS)

            # Classify archetype directly
            archetype_result = classify_zero_shot(combined, self.ARCHETYPE_LABELS)

            if content_result and archetype_result:
                return {
                    'content_themes': content_result,
                    'archetype_classification': archetype_result
                }

        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")

        return None

    def _map_zero_shot_to_archetype(
        self,
        classification: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map zero-shot archetype labels to archetype enum values.

        Args:
            classification: Zero-shot classification results

        Returns:
            Dict mapping archetype values to scores
        """
        mapping = {
            "guru dispensing wisdom": "the_guru",
            "honest grinder in trenches": "the_grinder",
            "paid shill promoter": "the_shill",
            "data-driven analyst": "the_analyst",
            "entertainer and memer": "the_entertainer",
            "points farmer optimizer": "the_farmer",
            "connected insider": "the_insider",
            "newcomer learning": "the_newcomer"
        }

        result = {}
        for label, score in classification.items():
            archetype_key = mapping.get(label)
            if archetype_key:
                result[archetype_key] = score * 100  # Convert to 0-100 scale

        return result

    def classify(
        self,
        engagement_score: float,
        consistency_score: float,
        dissonance_score: float,
        baiting_score: float,
        privilege_score: float,
        prediction_score: float,
        transparency_score: float,
        follower_quality_score: float,
        follower_count: int = 0,
        tweet_count: int = 0,
        account_age_days: int = 0,
        # Optional detailed report data
        privilege_report: dict = None,
        baiting_report: dict = None,
        sponsored_report: dict = None,
        prediction_report: dict = None,
        # NEW: Tweet texts for ML classification
        tweet_texts: List[str] = None
    ) -> ArchetypeProfile:
        """
        Classify a KOL into an archetype based on their scores.

        Args:
            engagement_score: Engagement authenticity score
            consistency_score: Position consistency score
            dissonance_score: Hypocrisy detection score
            baiting_score: Engagement bait score
            privilege_score: Privilege awareness score
            prediction_score: Prediction accuracy score
            transparency_score: Transparency score
            follower_quality_score: Follower quality score
            follower_count: Number of followers
            tweet_count: Number of tweets
            account_age_days: Account age in days
            privilege_report: Detailed privilege report
            baiting_report: Detailed baiting report
            sponsored_report: Detailed sponsored content report
            prediction_report: Detailed prediction report
            tweet_texts: List of tweet texts for ML classification

        Returns:
            ArchetypeProfile with classification results
        """
        # Check for newcomer first
        if tweet_count < 50 or follower_count < 1000:
            return self._create_newcomer_profile(follower_count, tweet_count)

        # Calculate archetype scores from metrics
        archetype_scores = self._calculate_archetype_scores(
            engagement_score, consistency_score, dissonance_score,
            baiting_score, privilege_score, prediction_score,
            transparency_score, follower_quality_score,
            follower_count, baiting_report, sponsored_report
        )

        # Try ML-based classification
        ml_result = None
        ml_archetype_scores = None
        content_themes = []

        if tweet_texts:
            ml_result = self._classify_content_zero_shot(tweet_texts)
            if ml_result:
                ml_archetype_scores = self._map_zero_shot_to_archetype(
                    ml_result.get('archetype_classification', {})
                )
                # Get top content themes
                content_result = ml_result.get('content_themes', {})
                content_themes = sorted(
                    content_result.keys(),
                    key=lambda k: content_result[k],
                    reverse=True
                )[:3]

        # Combine ML and metric-based scores (if ML available)
        if ml_archetype_scores:
            # Weighted combination: 40% ML, 60% metrics
            for key in archetype_scores:
                if key in ml_archetype_scores:
                    archetype_scores[key] = (
                        archetype_scores[key] * 0.6 +
                        ml_archetype_scores[key] * 0.4
                    )

        # Find primary and secondary archetypes
        sorted_archetypes = sorted(
            archetype_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary = sorted_archetypes[0]
        secondary = sorted_archetypes[1] if len(sorted_archetypes) > 1 else None

        primary_archetype = Archetype(primary[0])
        secondary_archetype = Archetype(secondary[0]) if secondary and secondary[1] > 30 else None

        # Get archetype definition
        definition = ARCHETYPE_DEFINITIONS[primary_archetype]

        # Calculate confidence
        confidence = self._calculate_confidence(
            primary[1],
            secondary[1] if secondary else 0,
            tweet_count
        )

        # Determine evolution stage
        evolution_stage, evolution_warning = self._assess_evolution(
            primary_archetype, follower_count, privilege_score,
            account_age_days, archetype_scores
        )

        profile = ArchetypeProfile(
            primary_archetype=primary_archetype,
            secondary_archetype=secondary_archetype,
            confidence=confidence,
            trust_level=definition["trust_level"],
            archetype_scores={k: v for k, v in archetype_scores.items()},
            one_liner=definition["one_liner"],
            detailed_description=definition["description"],
            key_behaviors=self._extract_key_behaviors(
                primary_archetype, privilege_score, baiting_score,
                transparency_score, prediction_score
            ),
            watch_out_for=definition["watch_out_for"],
            positive_traits=definition["positive_traits"],
            evolution_stage=evolution_stage,
            evolution_warning=evolution_warning
        )

        # Add ML classification if available
        if ml_result:
            profile.ml_available = True
            profile.ml_classification = ml_result.get('archetype_classification')
            profile.content_themes = content_themes

        return profile

    def _calculate_archetype_scores(
        self,
        engagement: float,
        consistency: float,
        dissonance: float,
        baiting: float,
        privilege: float,
        prediction: float,
        transparency: float,
        follower_quality: float,
        follower_count: int,
        baiting_report: dict = None,
        sponsored_report: dict = None
    ) -> Dict[str, float]:
        """Calculate how well the KOL matches each archetype."""
        scores = {}

        # THE GURU: High privilege blindness, established following
        guru_score = 0
        if privilege < 50:  # Low privilege score = high horse
            guru_score += (50 - privilege) * 0.6
        if follower_count > 50000:
            guru_score += 20
        if consistency > 70:  # Gurus are usually consistent
            guru_score += 10
        scores["the_guru"] = min(100, guru_score)

        # THE GRINDER: High empathy, transparent, still learning
        grinder_score = 0
        if privilege > 70:  # High privilege score = empathetic
            grinder_score += (privilege - 50) * 0.5
        if transparency > 70:
            grinder_score += 20
        if follower_count < 50000:  # Still growing
            grinder_score += 15
        if baiting < 60:  # Not gaming engagement
            grinder_score += 15
        scores["the_grinder"] = min(100, grinder_score)

        # THE SHILL: Low transparency, high promotion
        shill_score = 0
        if transparency < 50:
            shill_score += (50 - transparency) * 0.5
        if baiting < 50:  # Low baiting score = high bait activity
            shill_score += (50 - baiting) * 0.3
        if sponsored_report:
            undisclosed = sponsored_report.get('undisclosed_count', 0)
            if undisclosed > 5:
                shill_score += 30
        scores["the_shill"] = min(100, shill_score)

        # THE ANALYST: High prediction accuracy, consistent
        analyst_score = 0
        if prediction > 60:
            analyst_score += (prediction - 50) * 0.6
        if consistency > 70:
            analyst_score += 20
        if baiting > 70:  # Not baiting
            analyst_score += 15
        if transparency > 70:
            analyst_score += 15
        scores["the_analyst"] = min(100, analyst_score)

        # THE ENTERTAINER: High engagement bait, low substance
        entertainer_score = 0
        if baiting < 50:
            entertainer_score += (50 - baiting) * 0.5
        if engagement > 70:  # High engagement despite bait
            entertainer_score += 20
        if prediction < 50:  # Poor predictions
            entertainer_score += 15
        if consistency < 60:  # Inconsistent
            entertainer_score += 15
        scores["the_entertainer"] = min(100, entertainer_score)

        # THE FARMER: Reward platform optimization
        farmer_score = 0
        if baiting_report:
            if baiting_report.get('engagement_reward_optimization', False):
                farmer_score += 50
            reward_count = baiting_report.get('bait_type_counts', {}).get('reward_gaming', 0)
            if reward_count > 3:
                farmer_score += 30
        if baiting < 40:
            farmer_score += 20
        scores["the_farmer"] = min(100, farmer_score)

        # THE INSIDER: Early calls, low transparency
        insider_score = 0
        if prediction > 70 and transparency < 50:
            insider_score += 40  # Good calls but opaque
        if sponsored_report:
            if sponsored_report.get('frequently_promoted', []):
                insider_score += 20
        if consistency > 70:
            insider_score += 15
        scores["the_insider"] = min(100, insider_score)

        return scores

    def _calculate_confidence(
        self,
        primary_score: float,
        secondary_score: float,
        tweet_count: int
    ) -> float:
        """Calculate confidence in the classification."""
        # Confidence based on separation between top archetypes
        separation = primary_score - secondary_score
        separation_confidence = min(50, separation)

        # Confidence based on data volume
        if tweet_count >= 200:
            data_confidence = 50
        elif tweet_count >= 100:
            data_confidence = 40
        elif tweet_count >= 50:
            data_confidence = 30
        else:
            data_confidence = 20

        return separation_confidence + data_confidence

    def _assess_evolution(
        self,
        archetype: Archetype,
        follower_count: int,
        privilege_score: float,
        account_age_days: int,
        archetype_scores: Dict[str, float]
    ) -> Tuple[str, Optional[str]]:
        """Assess evolution stage and potential warnings."""
        # Determine stage
        if follower_count < 5000:
            stage = "early"
        elif follower_count < 25000:
            stage = "rising"
        elif follower_count < 100000:
            stage = "established"
        else:
            stage = "influential"

        # Check for evolution warnings
        warning = None

        # Grinder showing early guru signs
        if archetype == Archetype.THE_GRINDER:
            guru_score = archetype_scores.get("the_guru", 0)
            if guru_score > 40 and follower_count > 30000:
                warning = "Showing early signs of guru syndrome - watch for privilege creep"

        # Anyone with rising farmer score
        farmer_score = archetype_scores.get("the_farmer", 0)
        if farmer_score > 50:
            warning = "Heavy reward platform optimization detected"

        # Analyst becoming shill
        if archetype == Archetype.THE_ANALYST:
            shill_score = archetype_scores.get("the_shill", 0)
            if shill_score > 40:
                warning = "Analyst showing shill tendencies - may be monetizing audience"

        return stage, warning

    def _extract_key_behaviors(
        self,
        archetype: Archetype,
        privilege_score: float,
        baiting_score: float,
        transparency_score: float,
        prediction_score: float
    ) -> List[str]:
        """Extract specific behaviors based on scores."""
        behaviors = []

        if privilege_score < 40:
            behaviors.append("Frequently posts 'just be patient' type advice")
            behaviors.append("Shows little empathy for those still struggling")

        if baiting_score < 40:
            behaviors.append("Heavy use of engagement bait tactics")
            behaviors.append("Content optimized for algorithm, not value")

        if transparency_score < 40:
            behaviors.append("Rarely discloses paid promotions")
            behaviors.append("Frequent undisclosed shilling")

        if prediction_score > 70:
            behaviors.append("Strong track record on calls")
            behaviors.append("Follows up on predictions")
        elif prediction_score < 40:
            behaviors.append("Poor prediction accuracy")
            behaviors.append("Rarely revisits old calls")

        return behaviors[:5]

    def _create_newcomer_profile(
        self,
        follower_count: int,
        tweet_count: int
    ) -> ArchetypeProfile:
        """Create profile for accounts with insufficient history."""
        definition = ARCHETYPE_DEFINITIONS[Archetype.THE_NEWCOMER]

        return ArchetypeProfile(
            primary_archetype=Archetype.THE_NEWCOMER,
            secondary_archetype=None,
            confidence=20.0,
            trust_level=TrustLevel.NEUTRAL,
            archetype_scores={"the_newcomer": 100},
            one_liner=definition["one_liner"],
            detailed_description=f"Account has {tweet_count} tweets and {follower_count} followers. "
                                f"Not enough history to classify with confidence.",
            key_behaviors=["Insufficient data for behavior analysis"],
            watch_out_for=definition["watch_out_for"],
            positive_traits=definition["positive_traits"],
            evolution_stage="early",
            evolution_warning=None
        )

    def generate_narrative(
        self,
        profile: ArchetypeProfile,
        username: str = ""
    ) -> str:
        """Generate a human-readable narrative about the KOL."""
        name = f"@{username}" if username else "This account"
        archetype_name = ARCHETYPE_DEFINITIONS[profile.primary_archetype]["name"]
        emoji = ARCHETYPE_DEFINITIONS[profile.primary_archetype]["emoji"]

        narrative = f"{emoji} {name} is **{archetype_name}**\n\n"
        narrative += f"*{profile.one_liner}*\n\n"
        narrative += f"{profile.detailed_description}\n\n"

        if profile.evolution_warning:
            narrative += f"⚠️ **Warning:** {profile.evolution_warning}\n\n"

        narrative += f"**Trust Level:** {profile.trust_level.value.replace('_', ' ').title()}\n"
        narrative += f"**Stage:** {profile.evolution_stage.title()}\n"

        if profile.secondary_archetype:
            secondary_name = ARCHETYPE_DEFINITIONS[profile.secondary_archetype]["name"]
            narrative += f"**Also shows traits of:** {secondary_name}\n"

        return narrative
