"""
Privilege Analyzer - Detect moral high horse, survivorship bias, and privilege blindness.

Detects when successful accounts give condescending advice without acknowledging:
- Their current privileged position
- The struggles of those still grinding
- That their advice only works when you've already made it
- Survivorship bias in their "lessons learned"
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PrivilegeType(Enum):
    """Types of privilege blindness detected."""
    SURVIVORSHIP_BIAS = "survivorship_bias"  # "I made it so can you"
    FORGOTTEN_STRUGGLE = "forgotten_struggle"  # Rich person saying "money isn't everything"
    CONDESCENDING_WELLNESS = "condescending_wellness"  # "Health is wealth" from someone who doesn't need to grind
    EASY_FOR_YOU = "easy_for_you"  # Advice that requires resources/time they have
    HINDSIGHT_HERO = "hindsight_hero"  # "I always knew" when they got lucky
    PATIENCE_PREACHING = "patience_preaching"  # "Just wait" from someone already rich
    RISK_DOWNPLAY = "risk_downplay"  # Telling others to take risks they can afford to lose
    HUMBLE_ORIGIN_FLEX = "humble_origin_flex"  # "I started with nothing" but had advantages


@dataclass
class PrivilegeInstance:
    """A detected instance of privilege blindness."""
    tweet_id: str
    tweet_text: str
    privilege_type: PrivilegeType
    matched_patterns: List[str]
    severity: str  # "mild", "moderate", "severe"
    explanation: str
    timestamp: str

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:150] + '...' if len(self.tweet_text) > 150 else self.tweet_text,
            'privilege_type': self.privilege_type.value,
            'matched_patterns': self.matched_patterns,
            'severity': self.severity,
            'explanation': self.explanation,
            'timestamp': self.timestamp
        }


@dataclass
class PrivilegeReport:
    """Report on privilege blindness and moral high horse behavior."""
    instances: List[PrivilegeInstance] = field(default_factory=list)
    privilege_score: float = 100.0  # 100 = no privilege blindness, 0 = severe
    empathy_score: float = 100.0  # How empathetic vs condescending
    type_breakdown: Dict[str, int] = field(default_factory=dict)
    high_horse_indicators: List[str] = field(default_factory=list)
    empathy_indicators: List[str] = field(default_factory=list)
    context_factors: Dict[str, any] = field(default_factory=dict)  # follower count, account age, etc.

    def to_dict(self) -> dict:
        return {
            'instances': [i.to_dict() for i in self.instances[:15]],
            'privilege_score': round(self.privilege_score, 1),
            'empathy_score': round(self.empathy_score, 1),
            'type_breakdown': self.type_breakdown,
            'high_horse_indicators': self.high_horse_indicators,
            'empathy_indicators': self.empathy_indicators,
            'context_factors': self.context_factors
        }


class PrivilegeAnalyzer:
    """
    Analyzes tweets for privilege blindness and moral high horse behavior.

    Key detection areas:
    1. Survivorship bias - "I made it, you can too" without acknowledging luck/timing
    2. Forgotten struggle - Wealthy people saying "money doesn't buy happiness"
    3. Condescending wellness - "Take care of yourself" when you're grinding to survive
    4. Easy-for-you advice - Requires resources/time/connections they have
    5. Hindsight hero - "I knew it all along" when they got lucky
    6. Patience preaching - "Just HODL" from someone who doesn't need the money
    7. Risk downplaying - Telling others to "go all in" when they can afford losses
    """

    # Survivorship bias patterns - "I made it, so can you"
    SURVIVORSHIP_PATTERNS = [
        r'\b(if i can do it|anyone can|you can too)\b',
        r'\b(i started with nothing|from zero|started broke)\b',
        r'\b(no excuses|stop making excuses)\b',
        r'\b(just (need to|gotta|have to) (grind|hustle|work harder))\b',
        r'\b(success is a choice|choose to be successful)\b',
        r'\b(worked my way up|self.?made)\b',
        r'\b(mindset is everything|just change your mindset)\b',
        r'\b(i did it without|didn\'t have any help)\b',
    ]

    # Forgotten struggle - advice from a position of comfort
    FORGOTTEN_STRUGGLE_PATTERNS = [
        r'\b(money (isn\'t|doesn\'t|won\'t)|money can\'t buy)\b',
        r'\b(it\'s not about the money)\b',
        r'\b(there\'s more to life than (money|gains|profits))\b',
        r'\b(don\'t stress about (money|bills|rent))\b',
        r'\b(material (things|possessions) don\'t matter)\b',
        r'\b(been there done that)\b.*\b(trust me|believe me)\b',
    ]

    # Condescending wellness - "take care of yourself" when you're grinding to survive
    WELLNESS_PREACHING_PATTERNS = [
        r'\b(health is wealth|health over wealth)\b',
        r'\b(take (care of yourself|a break|time off))\b',
        r'\b(mental health (first|matters|is important))\b',
        r'\b(work.?life balance)\b',
        r'\b(don\'t burn yourself out)\b',
        r'\b(rest is productive|rest is important)\b',
        r'\b(touch grass|go outside|take a walk)\b',
        r'\b(log off|step away from (the screen|twitter|crypto))\b',
    ]

    # Easy-for-you advice - requires resources they have
    EASY_FOR_YOU_PATTERNS = [
        r'\b(just (buy|invest|hold|stake|accumulate))\b',
        r'\b(dollar cost average|dca (in|into))\b.*\b(every|weekly|monthly)\b',
        r'\b(don\'t sell|never sell|hold forever)\b',
        r'\b(think long term|long term thinking)\b',
        r'\b(be patient|patience pays|wait it out)\b',
        r'\b(don\'t need the money)\b',
        r'\b(let it ride|forget about it)\b',
        r'\b(time in the market)\b',
    ]

    # Hindsight hero - pretending they always knew
    HINDSIGHT_PATTERNS = [
        r'\b(i (always )?knew (it|this)|called it|told you)\b',
        r'\b(saw (this|it) coming)\b',
        r'\b(was obvious|obviously|clearly)\b.*\b(was going to|would)\b',
        r'\b(if you listened to me)\b',
        r'\b(i tried to warn|warned you)\b',
        r'\b(as i predicted|as i said)\b',
    ]

    # Patience preaching from comfort
    PATIENCE_PREACHING_PATTERNS = [
        r'\b(just (hodl|hold|wait)|be patient)\b',
        r'\b(stop checking (the price|your portfolio))\b',
        r'\b(zoom out|think bigger|bigger picture)\b',
        r'\b(in (5|10|20) years)\b',
        r'\b(generational wealth)\b',
        r'\b(your grandkids will thank you)\b',
        r'\b(we\'re still early|still early)\b',
    ]

    # Risk downplaying - telling others to risk what they can afford to lose
    RISK_DOWNPLAY_PATTERNS = [
        r'\b(go all in|max bid|full send)\b',
        r'\b(risk it (all|for the biscuit))\b',
        r'\b(scared money don\'t make money)\b',
        r'\b(fortune favors the bold|be bold)\b',
        r'\b(bet (big|the farm|everything))\b',
        r'\b(you only live once|yolo)\b',
        r'\b(what\'s the worst that (can|could) happen)\b',
        r'\b(don\'t be a coward|don\'t be scared)\b',
    ]

    # Humble origin flex - "I started with nothing" (but had advantages)
    HUMBLE_ORIGIN_FLEX_PATTERNS = [
        r'\b(started (from|with) (nothing|zero|the bottom))\b',
        r'\b(came from (nothing|poverty|the streets))\b',
        r'\b(didn\'t have (anything|connections|money))\b',
        r'\b(built (this|it) from scratch)\b',
        r'\b(no (silver spoon|handouts|help))\b',
        r'\b(pulled myself up)\b',
    ]

    # Empathy indicators - shows they remember the struggle
    EMPATHY_PATTERNS = [
        r'\b(i (remember|know) (how hard|the struggle|what it\'s like))\b',
        r'\b(not everyone can|not easy for everyone)\b',
        r'\b(i got lucky|was lucky|had help)\b',
        r'\b(privilege|privileged)\b',
        r'\b(took me (years|\d+ years|a long time))\b',
        r'\b(don\'t compare yourself)\b',
        r'\b(everyone\'s (journey|situation) is different)\b',
        r'\b(i had (advantages|support|help))\b',
        r'\b(timing was (right|lucky|fortunate))\b',
        r'\b(not financial advice|nfa|dyor)\b',
        r'\b(only (invest|risk) what you can afford to lose)\b',
    ]

    # Success indicators that provide context
    SUCCESS_INDICATORS = [
        r'\b(\d+k|\d+m|million|hundred thousand) (followers|following)\b',
        r'\b(made it|we made it|finally made it)\b',
        r'\b(retired|financial freedom|financially free)\b',
        r'\b(quit my job|left my (job|9-5|career))\b',
        r'\b(multi.?millionaire|millionaire|wealthy)\b',
        r'\b(lambo|yacht|mansion|penthouse)\b',
        r'\b(paid off|debt free)\b',
    ]

    # Follower thresholds that increase privilege blindness detection sensitivity
    LARGE_ACCOUNT_THRESHOLD = 50000
    MEDIUM_ACCOUNT_THRESHOLD = 10000

    def __init__(self):
        # Compile patterns
        self.survivorship_patterns = [re.compile(p, re.IGNORECASE) for p in self.SURVIVORSHIP_PATTERNS]
        self.forgotten_patterns = [re.compile(p, re.IGNORECASE) for p in self.FORGOTTEN_STRUGGLE_PATTERNS]
        self.wellness_patterns = [re.compile(p, re.IGNORECASE) for p in self.WELLNESS_PREACHING_PATTERNS]
        self.easy_patterns = [re.compile(p, re.IGNORECASE) for p in self.EASY_FOR_YOU_PATTERNS]
        self.hindsight_patterns = [re.compile(p, re.IGNORECASE) for p in self.HINDSIGHT_PATTERNS]
        self.patience_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATIENCE_PREACHING_PATTERNS]
        self.risk_patterns = [re.compile(p, re.IGNORECASE) for p in self.RISK_DOWNPLAY_PATTERNS]
        self.humble_origin_patterns = [re.compile(p, re.IGNORECASE) for p in self.HUMBLE_ORIGIN_FLEX_PATTERNS]
        self.empathy_patterns = [re.compile(p, re.IGNORECASE) for p in self.EMPATHY_PATTERNS]
        self.success_patterns = [re.compile(p, re.IGNORECASE) for p in self.SUCCESS_INDICATORS]

    def analyze(
        self,
        tweets: List[dict],
        follower_count: int = 0,
        account_age_days: int = 0
    ) -> PrivilegeReport:
        """
        Analyze tweets for privilege blindness.

        Args:
            tweets: List of tweet dictionaries
            follower_count: The account's follower count (higher = more scrutiny)
            account_age_days: How old the account is

        Returns:
            PrivilegeReport with analysis results
        """
        if not tweets:
            return PrivilegeReport()

        instances: List[PrivilegeInstance] = []
        type_counts: Dict[str, int] = {t.value: 0 for t in PrivilegeType}
        empathy_count = 0
        success_mentions = 0

        # Calculate context multiplier based on account size
        # Larger accounts held to higher standard
        context_multiplier = 1.0
        if follower_count >= self.LARGE_ACCOUNT_THRESHOLD:
            context_multiplier = 1.5  # More scrutiny for large accounts
        elif follower_count >= self.MEDIUM_ACCOUNT_THRESHOLD:
            context_multiplier = 1.25

        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Check for success indicators
            if any(p.search(text) for p in self.success_patterns):
                success_mentions += 1

            # Check for empathy indicators (positive)
            empathy_matches = [p.pattern for p in self.empathy_patterns if p.search(text)]
            if empathy_matches:
                empathy_count += 1
                continue  # Don't flag tweets that show empathy

            # Check each privilege type
            detections = []

            # Survivorship bias
            matches = [p.pattern for p in self.survivorship_patterns if p.search(text)]
            if matches:
                detections.append((
                    PrivilegeType.SURVIVORSHIP_BIAS,
                    matches,
                    "Implies success is purely about effort, ignoring luck/timing/resources"
                ))

            # Forgotten struggle
            matches = [p.pattern for p in self.forgotten_patterns if p.search(text)]
            if matches:
                detections.append((
                    PrivilegeType.FORGOTTEN_STRUGGLE,
                    matches,
                    "Dismisses money concerns from a position of financial comfort"
                ))

            # Condescending wellness
            matches = [p.pattern for p in self.wellness_patterns if p.search(text)]
            if matches:
                # Only flag if combined with success indicators or from large account
                if follower_count >= self.MEDIUM_ACCOUNT_THRESHOLD or success_mentions > 0:
                    detections.append((
                        PrivilegeType.CONDESCENDING_WELLNESS,
                        matches,
                        "Wellness advice that ignores financial pressures of those still grinding"
                    ))

            # Easy-for-you advice
            matches = [p.pattern for p in self.easy_patterns if p.search(text)]
            if matches:
                detections.append((
                    PrivilegeType.EASY_FOR_YOU,
                    matches,
                    "Advice that requires financial cushion or time horizon they have"
                ))

            # Hindsight hero
            matches = [p.pattern for p in self.hindsight_patterns if p.search(text)]
            if matches:
                detections.append((
                    PrivilegeType.HINDSIGHT_HERO,
                    matches,
                    "Claims to have predicted outcomes that involved significant luck"
                ))

            # Patience preaching
            matches = [p.pattern for p in self.patience_patterns if p.search(text)]
            if matches and follower_count >= self.MEDIUM_ACCOUNT_THRESHOLD:
                detections.append((
                    PrivilegeType.PATIENCE_PREACHING,
                    matches,
                    "Preaches patience from a position where they don't need the money"
                ))

            # Risk downplay
            matches = [p.pattern for p in self.risk_patterns if p.search(text)]
            if matches:
                detections.append((
                    PrivilegeType.RISK_DOWNPLAY,
                    matches,
                    "Encourages risky behavior while having resources to absorb losses"
                ))

            # Humble origin flex
            matches = [p.pattern for p in self.humble_origin_patterns if p.search(text)]
            if matches and follower_count >= self.MEDIUM_ACCOUNT_THRESHOLD:
                detections.append((
                    PrivilegeType.HUMBLE_ORIGIN_FLEX,
                    matches,
                    "Flexes humble origins without acknowledging advantages they had"
                ))

            # Create instances for each detection
            for privilege_type, matched_patterns, explanation in detections:
                severity = self._calculate_severity(
                    privilege_type,
                    len(matched_patterns),
                    follower_count
                )

                instances.append(PrivilegeInstance(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    privilege_type=privilege_type,
                    matched_patterns=matched_patterns[:3],
                    severity=severity,
                    explanation=explanation,
                    timestamp=timestamp
                ))

                type_counts[privilege_type.value] += 1

        # Calculate scores
        total_tweets = len(tweets)
        privilege_violations = len(instances)

        # Privilege score: penalize based on frequency and severity
        base_penalty = sum(
            self._get_severity_weight(i.severity) for i in instances
        ) * context_multiplier

        privilege_score = max(0.0, 100.0 - base_penalty)

        # Empathy score: reward empathy indicators
        empathy_ratio = empathy_count / total_tweets if total_tweets > 0 else 0
        empathy_score = min(100.0, 50.0 + (empathy_ratio * 500))  # Base 50, +50 for empathy

        # If many privilege instances but few empathy, reduce empathy score
        if privilege_violations > 5 and empathy_count < 3:
            empathy_score *= 0.7

        # Generate indicators
        high_horse_indicators = self._generate_high_horse_indicators(
            type_counts, instances, follower_count
        )

        empathy_indicators = self._generate_empathy_indicators(
            empathy_count, total_tweets
        )

        return PrivilegeReport(
            instances=instances,
            privilege_score=privilege_score,
            empathy_score=empathy_score,
            type_breakdown=type_counts,
            high_horse_indicators=high_horse_indicators,
            empathy_indicators=empathy_indicators,
            context_factors={
                'follower_count': follower_count,
                'account_age_days': account_age_days,
                'context_multiplier': context_multiplier,
                'empathy_tweet_count': empathy_count,
                'success_mentions': success_mentions
            }
        )

    def _calculate_severity(
        self,
        privilege_type: PrivilegeType,
        match_count: int,
        follower_count: int
    ) -> str:
        """Calculate severity based on type, matches, and context."""
        base_severity = 1

        # Some types are more severe
        severe_types = [
            PrivilegeType.RISK_DOWNPLAY,
            PrivilegeType.SURVIVORSHIP_BIAS
        ]
        moderate_types = [
            PrivilegeType.FORGOTTEN_STRUGGLE,
            PrivilegeType.EASY_FOR_YOU
        ]

        if privilege_type in severe_types:
            base_severity = 3
        elif privilege_type in moderate_types:
            base_severity = 2

        # More matches = more severe
        base_severity += min(2, match_count - 1)

        # Large accounts held to higher standard
        if follower_count >= self.LARGE_ACCOUNT_THRESHOLD:
            base_severity += 1

        if base_severity >= 4:
            return "severe"
        elif base_severity >= 2:
            return "moderate"
        else:
            return "mild"

    def _get_severity_weight(self, severity: str) -> float:
        """Get penalty weight for severity level."""
        weights = {
            "severe": 8.0,
            "moderate": 4.0,
            "mild": 2.0
        }
        return weights.get(severity, 2.0)

    def _generate_high_horse_indicators(
        self,
        type_counts: Dict[str, int],
        instances: List[PrivilegeInstance],
        follower_count: int
    ) -> List[str]:
        """Generate human-readable high horse indicators."""
        indicators = []

        if type_counts.get('survivorship_bias', 0) > 3:
            indicators.append(
                f"Frequent survivorship bias ({type_counts['survivorship_bias']} instances) - "
                "implies success is purely about effort"
            )

        if type_counts.get('risk_downplay', 0) > 2:
            indicators.append(
                f"Downplays risk ({type_counts['risk_downplay']} instances) - "
                "encourages risky behavior from position of safety"
            )

        if type_counts.get('forgotten_struggle', 0) > 2:
            indicators.append(
                "Dismisses financial concerns - may have forgotten early struggles"
            )

        if type_counts.get('easy_for_you', 0) > 5:
            indicators.append(
                "Gives advice that requires financial cushion they have"
            )

        if type_counts.get('hindsight_hero', 0) > 3:
            indicators.append(
                "Frequently claims to have 'always known' - hindsight bias"
            )

        if type_counts.get('condescending_wellness', 0) > 3 and follower_count > self.MEDIUM_ACCOUNT_THRESHOLD:
            indicators.append(
                "Preaches wellness without acknowledging grind required to reach their position"
            )

        severe_count = sum(1 for i in instances if i.severity == "severe")
        if severe_count > 2:
            indicators.append(
                f"Multiple severe privilege blindness instances ({severe_count})"
            )

        return indicators[:5]

    def _generate_empathy_indicators(
        self,
        empathy_count: int,
        total_tweets: int
    ) -> List[str]:
        """Generate positive empathy indicators."""
        indicators = []

        if total_tweets == 0:
            return indicators

        empathy_ratio = empathy_count / total_tweets

        if empathy_ratio > 0.1:
            indicators.append("Frequently acknowledges luck and privilege")
        elif empathy_ratio > 0.05:
            indicators.append("Sometimes shows awareness of their advantages")

        if empathy_count > 5:
            indicators.append(f"Demonstrates empathy in {empathy_count} tweets")

        return indicators

    def generate_summary(self, report: PrivilegeReport) -> str:
        """Generate a human-readable summary."""
        if not report.instances:
            return "No significant privilege blindness detected. Account shows balanced perspective."

        parts = []

        # Main finding
        total_instances = len(report.instances)
        severe_count = sum(1 for i in report.instances if i.severity == "severe")

        if severe_count > 2:
            parts.append(f"Significant moral high horse behavior detected ({severe_count} severe instances).")
        elif total_instances > 5:
            parts.append(f"Multiple privilege blindness indicators ({total_instances} instances).")
        else:
            parts.append(f"Some privilege blindness detected ({total_instances} instances).")

        # Top issue
        if report.type_breakdown:
            top_type = max(report.type_breakdown.items(), key=lambda x: x[1])
            if top_type[1] > 2:
                type_names = {
                    'survivorship_bias': 'survivorship bias',
                    'forgotten_struggle': 'forgotten struggle mindset',
                    'condescending_wellness': 'condescending wellness advice',
                    'easy_for_you': 'easy-for-them advice',
                    'hindsight_hero': 'hindsight bias',
                    'patience_preaching': 'patience preaching',
                    'risk_downplay': 'risk downplaying',
                    'humble_origin_flex': 'humble origin flexing'
                }
                parts.append(f"Primary issue: {type_names.get(top_type[0], top_type[0])}.")

        # Empathy note
        if report.empathy_indicators:
            parts.append("However, some empathy shown.")
        elif report.empathy_score < 50:
            parts.append("Low empathy for those still struggling.")

        return " ".join(parts)
