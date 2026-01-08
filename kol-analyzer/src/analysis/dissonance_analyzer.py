"""
Dissonance Analyzer - Detect hypocrisy and two-faced behavior in KOL tweets.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class ToneType(Enum):
    """Classification of tweet tone."""
    INSTRUCTIONAL = "instructional"  # Genuinely teaching
    DERISIVE = "derisive"  # Mocking/punching down
    HUMBLE_BRAG = "humble_brag"  # Disguised flexing
    GATEKEEPING = "gatekeeping"  # "You don't belong"
    INCLUSIVE = "inclusive"  # Welcoming
    NEUTRAL = "neutral"


@dataclass
class ToneInstance:
    """A detected tone instance in a tweet."""
    tweet_id: str
    tweet_text: str
    tone: ToneType
    matched_patterns: List[str]
    timestamp: str

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:150] + '...' if len(self.tweet_text) > 150 else self.tweet_text,
            'tone': self.tone.value,
            'matched_patterns': self.matched_patterns,
            'timestamp': self.timestamp
        }


@dataclass
class HypocrisyInstance:
    """A detected instance of hypocrisy."""
    description: str
    historical_tweet_id: str
    historical_text: str
    recent_tweet_id: str
    recent_text: str
    severity: str  # "minor", "moderate", "major"

    def to_dict(self) -> dict:
        return {
            'description': self.description,
            'historical_text': self.historical_text[:100] + '...' if len(self.historical_text) > 100 else self.historical_text,
            'recent_text': self.recent_text[:100] + '...' if len(self.recent_text) > 100 else self.recent_text,
            'severity': self.severity
        }


@dataclass
class DissonanceReport:
    """Report on tone and behavioral consistency."""
    tone_breakdown: Dict[str, int] = field(default_factory=dict)
    tone_instances: List[ToneInstance] = field(default_factory=list)
    hypocrisy_instances: List[HypocrisyInstance] = field(default_factory=list)
    hypocrisy_score: float = 100.0  # 100 = no hypocrisy
    authenticity_score: float = 100.0  # Based on tone
    two_faced_indicators: List[str] = field(default_factory=list)
    power_dynamic_violations: List[str] = field(default_factory=list)
    primary_tone: str = "neutral"
    derisive_percentage: float = 0.0

    def to_dict(self) -> dict:
        return {
            'tone_breakdown': self.tone_breakdown,
            'tone_instances': [ti.to_dict() for ti in self.tone_instances[:10]],  # Limit for output
            'hypocrisy_instances': [hi.to_dict() for hi in self.hypocrisy_instances],
            'hypocrisy_score': round(self.hypocrisy_score, 1),
            'authenticity_score': round(self.authenticity_score, 1),
            'two_faced_indicators': self.two_faced_indicators,
            'power_dynamic_violations': self.power_dynamic_violations,
            'primary_tone': self.primary_tone,
            'derisive_percentage': round(self.derisive_percentage, 1)
        }


class DissonanceAnalyzer:
    """
    Analyzes tweets for hypocrisy and two-faced behavior.

    Detection logic:
    - Classify tweet tones (instructional, derisive, humble_brag, etc.)
    - Detect power dynamic violations (mocking newcomers)
    - Find hypocrisy (criticizing behaviors they've done)
    """

    # Pattern dictionaries
    DERISIVE_PATTERNS = [
        r'\b(ngmi|rekt|dumb money|normies|plebs|noobs|paper hands)\b',
        r'\b(imagine|couldn\'t be me|skill issue|cope|seethe)\b',
        r'\b(have fun staying poor|hfsp)\b',
        r'\b(idiots?|morons?|clowns?)\b',
        r'\b(deserve to lose|deserved it)\b',
        r'\b(weak hands|exit liquidity)\b',
        r'🤡',  # Clown emoji in mocking context
    ]

    INSTRUCTIONAL_PATTERNS = [
        r'\b(here\'s how|let me explain|step by step|guide|tutorial)\b',
        r'\b(for beginners|if you\'re new|starting out)\b',
        r'\b(tip:|pro tip|dyor|nfa)\b',
        r'\b(thread|🧵)\b',
        r'\b(learn|learning|education|educational)\b',
        r'\b(explained|explaining|breakdown)\b',
    ]

    GATEKEEPING_PATTERNS = [
        r'\b(real ones know|og|veteran|been here since)\b',
        r'\b(if you know you know|iykyk)\b',
        r'\b(tourists?|bandwagoners?|late|latecomers?)\b',
        r'\b(not for everyone|only real)\b',
        r'\b(where were you when|back when)\b',
    ]

    INCLUSIVE_PATTERNS = [
        r'\b(welcome|welcoming|join us|happy to help)\b',
        r'\b(everyone can|anyone can|you can too)\b',
        r'\b(no stupid questions|ask me anything|ama)\b',
        r'\b(we\'re all|we all started)\b',
        r'\b(dm.*(open|me)|happy to chat)\b',
    ]

    HUMBLE_BRAG_PATTERNS = [
        r'\b(just got lucky|accidentally|somehow)\b.*(profit|gain|x|money)',
        r'\b(down only|only down)\b.*\d+%.*\b(still up|up on)\b',
        r'\b(people asking|everyone asking)\b.*(how|secret)',
        r'\b(humble|humbly)\b.*(share|admit)',
        r'\b(not flexing|don\'t mean to flex)\b',
        r'\b(small (bag|position))\b.*\d+(x|k|eth|btc)',
    ]

    # Vulnerability patterns (things that indicate past struggles)
    VULNERABILITY_PATTERNS = [
        r'\b(got rekt|i got rekt|was rekt)\b',
        r'\b(lost (everything|it all|my))\b',
        r'\b(fomo\'?d|i fomo\'?d)\b',
        r'\b(made mistakes?|my mistake)\b',
        r'\b(learned.*hard way)\b',
        r'\b(been there|done that)\b',
        r'\b(paper handed|i paper handed)\b',
        r'\b(blew up|account.*blown)\b',
    ]

    # Newcomer targeting patterns
    NEWCOMER_TARGETS = [
        r'\b(newbie|noob|normie|retail|pleb)\b',
        r'\b(new (to|in) crypto)\b',
        r'\b(just (started|joined|entered))\b',
    ]

    def __init__(self):
        # Compile patterns
        self.derisive_patterns = [re.compile(p, re.IGNORECASE) for p in self.DERISIVE_PATTERNS]
        self.instructional_patterns = [re.compile(p, re.IGNORECASE) for p in self.INSTRUCTIONAL_PATTERNS]
        self.gatekeeping_patterns = [re.compile(p, re.IGNORECASE) for p in self.GATEKEEPING_PATTERNS]
        self.inclusive_patterns = [re.compile(p, re.IGNORECASE) for p in self.INCLUSIVE_PATTERNS]
        self.humble_brag_patterns = [re.compile(p, re.IGNORECASE) for p in self.HUMBLE_BRAG_PATTERNS]
        self.vulnerability_patterns = [re.compile(p, re.IGNORECASE) for p in self.VULNERABILITY_PATTERNS]
        self.newcomer_patterns = [re.compile(p, re.IGNORECASE) for p in self.NEWCOMER_TARGETS]

    def analyze(self, tweets: List[dict]) -> DissonanceReport:
        """
        Analyze tweets for dissonance and hypocrisy.

        Args:
            tweets: List of tweet dictionaries

        Returns:
            DissonanceReport with analysis results
        """
        if not tweets:
            return DissonanceReport()

        tone_breakdown: Dict[str, int] = {t.value: 0 for t in ToneType}
        tone_instances: List[ToneInstance] = []
        vulnerability_tweets: List[dict] = []
        derisive_tweets: List[dict] = []
        power_violations: List[str] = []
        two_faced_indicators: List[str] = []

        # Analyze each tweet
        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Classify tone
            tone, matched = self._classify_tone(text)
            tone_breakdown[tone.value] += 1

            if tone != ToneType.NEUTRAL and matched:
                tone_instances.append(ToneInstance(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    tone=tone,
                    matched_patterns=matched,
                    timestamp=timestamp
                ))

            # Track vulnerability mentions
            vuln_matches = self._find_vulnerability_mentions(text)
            if vuln_matches:
                vulnerability_tweets.append({
                    **tweet,
                    'vulnerability_matches': vuln_matches
                })

            # Track derisive tweets
            if tone == ToneType.DERISIVE:
                derisive_tweets.append(tweet)

            # Check for power dynamic violations
            violation = self._check_power_violation(text)
            if violation:
                power_violations.append(violation)

        # Detect hypocrisy
        hypocrisy_instances = self._detect_hypocrisy(
            vulnerability_tweets,
            derisive_tweets
        )

        # Calculate scores
        total_classified = sum(v for k, v in tone_breakdown.items() if k != 'neutral')
        total_tweets = len(tweets)

        derisive_count = tone_breakdown.get('derisive', 0)
        derisive_percentage = (derisive_count / total_tweets * 100) if total_tweets > 0 else 0

        # Authenticity score based on tone mix
        authenticity_score = self._calculate_authenticity_score(tone_breakdown, total_tweets)

        # Hypocrisy score
        hypocrisy_penalty = len(hypocrisy_instances) * 10 + len(power_violations) * 5
        hypocrisy_score = max(0.0, 100.0 - hypocrisy_penalty)

        # Determine primary tone
        primary_tone = max(
            [(k, v) for k, v in tone_breakdown.items() if k != 'neutral'],
            key=lambda x: x[1],
            default=('neutral', 0)
        )[0]

        # Identify two-faced indicators
        if derisive_count > 0 and tone_breakdown.get('instructional', 0) > 0:
            ratio = derisive_count / tone_breakdown['instructional']
            if ratio > 0.5:
                two_faced_indicators.append(
                    f"Mixed messaging: {derisive_count} derisive vs {tone_breakdown['instructional']} instructional tweets"
                )

        if tone_breakdown.get('humble_brag', 0) > total_tweets * 0.1:
            two_faced_indicators.append(
                f"Frequent humble bragging ({tone_breakdown['humble_brag']} instances)"
            )

        if len(hypocrisy_instances) > 0:
            two_faced_indicators.append(
                f"Criticizes behaviors they've admitted to doing ({len(hypocrisy_instances)} instances)"
            )

        return DissonanceReport(
            tone_breakdown=tone_breakdown,
            tone_instances=tone_instances,
            hypocrisy_instances=hypocrisy_instances,
            hypocrisy_score=hypocrisy_score,
            authenticity_score=authenticity_score,
            two_faced_indicators=two_faced_indicators,
            power_dynamic_violations=power_violations,
            primary_tone=primary_tone,
            derisive_percentage=derisive_percentage
        )

    def _classify_tone(self, text: str) -> Tuple[ToneType, List[str]]:
        """Classify the tone of a tweet."""
        matched_patterns = []

        # Check each tone type
        scores = {
            ToneType.DERISIVE: 0,
            ToneType.INSTRUCTIONAL: 0,
            ToneType.GATEKEEPING: 0,
            ToneType.INCLUSIVE: 0,
            ToneType.HUMBLE_BRAG: 0,
        }

        for pattern in self.derisive_patterns:
            if pattern.search(text):
                scores[ToneType.DERISIVE] += 1
                matched_patterns.append(pattern.pattern)

        for pattern in self.instructional_patterns:
            if pattern.search(text):
                scores[ToneType.INSTRUCTIONAL] += 1
                matched_patterns.append(pattern.pattern)

        for pattern in self.gatekeeping_patterns:
            if pattern.search(text):
                scores[ToneType.GATEKEEPING] += 1
                matched_patterns.append(pattern.pattern)

        for pattern in self.inclusive_patterns:
            if pattern.search(text):
                scores[ToneType.INCLUSIVE] += 1
                matched_patterns.append(pattern.pattern)

        for pattern in self.humble_brag_patterns:
            if pattern.search(text):
                scores[ToneType.HUMBLE_BRAG] += 1
                matched_patterns.append(pattern.pattern)

        # Find dominant tone
        max_score = max(scores.values())
        if max_score == 0:
            return ToneType.NEUTRAL, []

        dominant_tone = max(scores, key=scores.get)
        return dominant_tone, matched_patterns[:3]  # Limit patterns returned

    def _find_vulnerability_mentions(self, text: str) -> List[str]:
        """Find mentions of past mistakes/struggles."""
        matches = []
        for pattern in self.vulnerability_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    def _check_power_violation(self, text: str) -> Optional[str]:
        """Check if tweet targets newcomers with derisive language."""
        has_newcomer_ref = any(p.search(text) for p in self.newcomer_patterns)
        has_derisive = any(p.search(text) for p in self.derisive_patterns)

        if has_newcomer_ref and has_derisive:
            return f"Derisive content targeting newcomers: '{text[:80]}...'"

        return None

    def _detect_hypocrisy(
        self,
        vulnerability_tweets: List[dict],
        derisive_tweets: List[dict]
    ) -> List[HypocrisyInstance]:
        """Detect hypocrisy by comparing vulnerability admissions with derisive tweets."""
        hypocrisy_instances = []

        # Map vulnerability types to derisive patterns
        hypocrisy_mappings = [
            {
                'vulnerability': r'(got rekt|was rekt|lost)',
                'criticism': r'(rekt|ngmi|deserve)',
                'description': 'Mocks others for losses despite admitting to losses'
            },
            {
                'vulnerability': r'(fomo\'?d|i fomo\'?d)',
                'criticism': r'(fomo|ngmi|dumb)',
                'description': 'Criticizes FOMO despite admitting to FOMO behavior'
            },
            {
                'vulnerability': r'(paper handed|panic sold)',
                'criticism': r'(paper hands|weak hands)',
                'description': 'Mocks paper hands despite admitting to paper handing'
            },
            {
                'vulnerability': r'(made mistakes?|blew up)',
                'criticism': r'(idiots?|morons?|dumb)',
                'description': 'Calls others idiots despite admitting to mistakes'
            },
        ]

        for mapping in hypocrisy_mappings:
            vuln_pattern = re.compile(mapping['vulnerability'], re.IGNORECASE)
            crit_pattern = re.compile(mapping['criticism'], re.IGNORECASE)

            # Find vulnerability tweets matching this pattern
            matching_vuln = [
                t for t in vulnerability_tweets
                if vuln_pattern.search(t.get('text', ''))
            ]

            # Find derisive tweets matching the criticism pattern
            matching_crit = [
                t for t in derisive_tweets
                if crit_pattern.search(t.get('text', ''))
            ]

            if matching_vuln and matching_crit:
                # Found hypocrisy
                hypocrisy_instances.append(HypocrisyInstance(
                    description=mapping['description'],
                    historical_tweet_id=matching_vuln[0].get('id', ''),
                    historical_text=matching_vuln[0].get('text', ''),
                    recent_tweet_id=matching_crit[0].get('id', ''),
                    recent_text=matching_crit[0].get('text', ''),
                    severity='moderate'
                ))

        return hypocrisy_instances

    def _calculate_authenticity_score(
        self,
        tone_breakdown: Dict[str, int],
        total_tweets: int
    ) -> float:
        """Calculate authenticity score based on tone distribution."""
        if total_tweets == 0:
            return 50.0

        score = 100.0

        # Penalize high derisive content
        derisive_pct = tone_breakdown.get('derisive', 0) / total_tweets
        if derisive_pct > 0.2:
            score -= 25
        elif derisive_pct > 0.1:
            score -= 15
        elif derisive_pct > 0.05:
            score -= 5

        # Penalize high gatekeeping
        gatekeeping_pct = tone_breakdown.get('gatekeeping', 0) / total_tweets
        if gatekeeping_pct > 0.1:
            score -= 15
        elif gatekeeping_pct > 0.05:
            score -= 8

        # Penalize excessive humble bragging
        humble_brag_pct = tone_breakdown.get('humble_brag', 0) / total_tweets
        if humble_brag_pct > 0.15:
            score -= 15
        elif humble_brag_pct > 0.08:
            score -= 8

        # Reward instructional content
        instructional_pct = tone_breakdown.get('instructional', 0) / total_tweets
        if instructional_pct > 0.2:
            score += 10
        elif instructional_pct > 0.1:
            score += 5

        # Reward inclusive content
        inclusive_pct = tone_breakdown.get('inclusive', 0) / total_tweets
        if inclusive_pct > 0.1:
            score += 10

        return max(0.0, min(100.0, score))

    def generate_summary(self, report: DissonanceReport) -> str:
        """Generate a human-readable summary."""
        parts = [f"Primary tone: {report.primary_tone}"]

        if report.derisive_percentage > 10:
            parts.append(f"High derisive content ({report.derisive_percentage:.1f}%)")

        if report.hypocrisy_instances:
            parts.append(f"{len(report.hypocrisy_instances)} hypocrisy instance(s) detected")

        if report.power_dynamic_violations:
            parts.append(f"{len(report.power_dynamic_violations)} power dynamic violation(s)")

        if not report.two_faced_indicators:
            parts.append("No major two-faced behavior detected")

        return ". ".join(parts) + "."
