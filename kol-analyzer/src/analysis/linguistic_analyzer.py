"""
Linguistic Authenticity Analyzer - Analyze language patterns for authenticity signals.

Detects:
- Certainty calibration: Do they use appropriate hedging?
- Complexity mismatch: Fancy words but basic errors?
- Emotional manipulation: Fear/greed triggers
- Authenticity drift: Style changes (ghostwriter detection)
- Copy-paste patterns: Repeated templates
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
import statistics


class CertaintyLevel:
    """Levels of expressed certainty."""
    ABSOLUTE = "absolute"  # "100% guaranteed", "definitely"
    HIGH = "high"  # "very likely", "almost certain"
    MODERATE = "moderate"  # "probably", "I think"
    HEDGED = "hedged"  # "might", "could", "possibly"
    UNCERTAIN = "uncertain"  # "I don't know", "unclear"


@dataclass
class LinguisticReport:
    """Report on linguistic patterns."""
    authenticity_score: float = 50.0  # 0-100
    certainty_calibration: str = "unknown"  # "overconfident", "calibrated", "underconfident"

    # Certainty analysis
    avg_certainty_level: float = 0.0  # 0-100, higher = more certain
    absolute_claims_count: int = 0
    hedged_claims_count: int = 0
    certainty_consistency: float = 0.0  # How consistent is their certainty

    # Complexity analysis
    avg_word_length: float = 0.0
    vocabulary_richness: float = 0.0  # Unique words / total words
    jargon_density: float = 0.0  # Crypto/finance jargon usage
    complexity_score: float = 0.0

    # Manipulation patterns
    fear_triggers: int = 0
    greed_triggers: int = 0
    urgency_triggers: int = 0
    manipulation_score: float = 0.0

    # Authenticity signals
    first_person_ratio: float = 0.0  # "I", "my" usage
    question_ratio: float = 0.0  # Asks vs tells
    emoji_density: float = 0.0
    caps_ratio: float = 0.0  # ALL CAPS usage

    # Template detection
    repeated_phrases: List[Tuple[str, int]] = field(default_factory=list)
    template_score: float = 0.0  # 0-100, higher = more templated

    # Style consistency
    style_variance: float = 0.0  # Low = consistent style
    potential_ghostwriter: bool = False

    patterns_detected: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'authenticity_score': round(self.authenticity_score, 1),
            'certainty_calibration': self.certainty_calibration,
            'avg_certainty_level': round(self.avg_certainty_level, 1),
            'absolute_claims_count': self.absolute_claims_count,
            'hedged_claims_count': self.hedged_claims_count,
            'avg_word_length': round(self.avg_word_length, 2),
            'vocabulary_richness': round(self.vocabulary_richness, 3),
            'jargon_density': round(self.jargon_density, 3),
            'complexity_score': round(self.complexity_score, 1),
            'fear_triggers': self.fear_triggers,
            'greed_triggers': self.greed_triggers,
            'urgency_triggers': self.urgency_triggers,
            'manipulation_score': round(self.manipulation_score, 1),
            'first_person_ratio': round(self.first_person_ratio, 3),
            'question_ratio': round(self.question_ratio, 3),
            'emoji_density': round(self.emoji_density, 3),
            'caps_ratio': round(self.caps_ratio, 3),
            'repeated_phrases': self.repeated_phrases[:10],
            'template_score': round(self.template_score, 1),
            'style_variance': round(self.style_variance, 2),
            'potential_ghostwriter': self.potential_ghostwriter,
            'patterns_detected': self.patterns_detected,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class LinguisticAnalyzer:
    """
    Analyzes language patterns for authenticity and manipulation signals.
    """

    # Absolute certainty markers
    ABSOLUTE_CERTAINTY = [
        r'\b(100%|guaranteed|definitely|certainly|absolutely|for sure)\b',
        r'\b(will (moon|pump|dump)|going to (moon|pump|10x))\b',
        r'\b(can\'t lose|impossible to|no way|zero chance)\b',
        r'\b(trust me|mark my words|remember this)\b',
        r'\b(this is it|the one|next big thing)\b',
    ]

    # High certainty markers
    HIGH_CERTAINTY = [
        r'\b(very likely|almost certain|pretty sure|highly probable)\b',
        r'\b(expect(ing)?|anticipate|confident)\b',
        r'\b(should|will probably|most likely)\b',
    ]

    # Hedged language
    HEDGED_LANGUAGE = [
        r'\b(might|could|may|possibly|perhaps|maybe)\b',
        r'\b(i think|i believe|in my opinion|imo|seems like)\b',
        r'\b(not sure|unclear|hard to say|depends)\b',
        r'\b(if|assuming|provided that|unless)\b',
        r'\b(dyor|nfa|not financial advice)\b',
    ]

    # Fear triggers (FUD)
    FEAR_TRIGGERS = [
        r'\b(crash|collapse|dump|rug|scam|ponzi)\b',
        r'\b(warning|danger|risk|careful|watch out)\b',
        r'\b(liquidat|margin call|blow up|rekt)\b',
        r'\b(sell now|get out|exit|before it\'s too late)\b',
        r'\b(dead|dying|over|finished|done)\b',
    ]

    # Greed triggers
    GREED_TRIGGERS = [
        r'\b(100x|1000x|10x|massive gains|huge returns)\b',
        r'\b(easy money|free money|guaranteed profit)\b',
        r'\b(millionaire|life changing|generational)\b',
        r'\b(don\'t miss|last chance|early)\b',
        r'\b(moon|lambo|yacht|retire)\b',
    ]

    # Urgency triggers
    URGENCY_TRIGGERS = [
        r'\b(now|today|right now|immediately|asap)\b',
        r'\b(hurry|quick|fast|before|last chance)\b',
        r'\b(limited|only|exclusive|ending soon)\b',
        r'\b(don\'t wait|act now|time sensitive)\b',
    ]

    # Crypto jargon
    CRYPTO_JARGON = [
        r'\b(alpha|degen|ape|fomo|fud|hodl|wagmi|ngmi)\b',
        r'\b(bullish|bearish|moon|pump|dump|rekt)\b',
        r'\b(dex|cex|defi|nft|dao|tvl|mcap)\b',
        r'\b(whale|shrimp|diamond hands|paper hands)\b',
        r'\b(yield|apy|apr|liquidity|slippage)\b',
        r'\b(layer 1|layer 2|l1|l2|rollup|bridge)\b',
    ]

    # Common templates/copypasta patterns
    TEMPLATE_PATTERNS = [
        r'(thread|🧵)',
        r'(\d+/\d+)',  # 1/10 thread format
        r'(here\'s why|let me explain|breakdown)',
        r'(gm|gn|wagmi)',
        r'(like.*retweet|rt.*follow)',
    ]

    def __init__(self):
        self.absolute_patterns = [re.compile(p, re.IGNORECASE) for p in self.ABSOLUTE_CERTAINTY]
        self.high_cert_patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_CERTAINTY]
        self.hedge_patterns = [re.compile(p, re.IGNORECASE) for p in self.HEDGED_LANGUAGE]
        self.fear_patterns = [re.compile(p, re.IGNORECASE) for p in self.FEAR_TRIGGERS]
        self.greed_patterns = [re.compile(p, re.IGNORECASE) for p in self.GREED_TRIGGERS]
        self.urgency_patterns = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_TRIGGERS]
        self.jargon_patterns = [re.compile(p, re.IGNORECASE) for p in self.CRYPTO_JARGON]
        self.template_patterns = [re.compile(p, re.IGNORECASE) for p in self.TEMPLATE_PATTERNS]
        self.emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')

    def analyze(self, tweets: List[dict]) -> LinguisticReport:
        """Analyze linguistic patterns across tweets."""
        if not tweets:
            return LinguisticReport()

        all_text = []
        tweet_metrics = []
        phrase_counter: Counter = Counter()

        for tweet in tweets:
            text = tweet.get('text', '')
            if not text:
                continue

            all_text.append(text)
            metrics = self._analyze_single_tweet(text)
            tweet_metrics.append(metrics)

            # Extract phrases for template detection
            phrases = self._extract_phrases(text)
            phrase_counter.update(phrases)

        if not tweet_metrics:
            return LinguisticReport()

        # Aggregate metrics
        combined_text = ' '.join(all_text)

        # Certainty analysis
        absolute_count = sum(m['absolute_certainty'] for m in tweet_metrics)
        high_count = sum(m['high_certainty'] for m in tweet_metrics)
        hedged_count = sum(m['hedged'] for m in tweet_metrics)

        total_claims = absolute_count + high_count + hedged_count
        avg_certainty = 0.0
        if total_claims > 0:
            avg_certainty = (absolute_count * 100 + high_count * 70 + hedged_count * 30) / total_claims

        certainty_calibration = self._determine_calibration(absolute_count, hedged_count, total_claims)

        # Complexity analysis
        all_words = re.findall(r'\b\w+\b', combined_text.lower())
        avg_word_length = sum(len(w) for w in all_words) / len(all_words) if all_words else 0
        vocabulary_richness = len(set(all_words)) / len(all_words) if all_words else 0

        jargon_count = sum(1 for p in self.jargon_patterns for _ in p.finditer(combined_text))
        jargon_density = jargon_count / len(all_words) if all_words else 0

        complexity_score = self._calculate_complexity(avg_word_length, vocabulary_richness, jargon_density)

        # Manipulation analysis
        fear_count = sum(m['fear_triggers'] for m in tweet_metrics)
        greed_count = sum(m['greed_triggers'] for m in tweet_metrics)
        urgency_count = sum(m['urgency_triggers'] for m in tweet_metrics)
        manipulation_score = self._calculate_manipulation_score(
            fear_count, greed_count, urgency_count, len(tweets)
        )

        # Style metrics
        first_person = len(re.findall(r'\b(i|my|me|myself)\b', combined_text, re.IGNORECASE))
        first_person_ratio = first_person / len(all_words) if all_words else 0

        questions = len(re.findall(r'\?', combined_text))
        question_ratio = questions / len(tweets)

        emoji_count = len(self.emoji_pattern.findall(combined_text))
        emoji_density = emoji_count / len(tweets)

        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', combined_text))
        caps_ratio = caps_words / len(all_words) if all_words else 0

        # Template detection
        repeated_phrases = [(phrase, count) for phrase, count in phrase_counter.most_common(20) if count >= 3]
        template_score = self._calculate_template_score(repeated_phrases, len(tweets))

        # Style variance (for ghostwriter detection)
        style_variance, potential_ghostwriter = self._analyze_style_consistency(tweet_metrics)

        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity_score(
            certainty_calibration,
            manipulation_score,
            template_score,
            style_variance,
            first_person_ratio
        )

        # Generate patterns, flags
        patterns = self._detect_patterns(
            avg_certainty, manipulation_score, template_score,
            fear_count, greed_count, potential_ghostwriter
        )

        red_flags, green_flags = self._generate_flags(
            absolute_count, manipulation_score, template_score,
            certainty_calibration, style_variance
        )

        return LinguisticReport(
            authenticity_score=authenticity_score,
            certainty_calibration=certainty_calibration,
            avg_certainty_level=avg_certainty,
            absolute_claims_count=absolute_count,
            hedged_claims_count=hedged_count,
            avg_word_length=avg_word_length,
            vocabulary_richness=vocabulary_richness,
            jargon_density=jargon_density,
            complexity_score=complexity_score,
            fear_triggers=fear_count,
            greed_triggers=greed_count,
            urgency_triggers=urgency_count,
            manipulation_score=manipulation_score,
            first_person_ratio=first_person_ratio,
            question_ratio=question_ratio,
            emoji_density=emoji_density,
            caps_ratio=caps_ratio,
            repeated_phrases=repeated_phrases,
            template_score=template_score,
            style_variance=style_variance,
            potential_ghostwriter=potential_ghostwriter,
            patterns_detected=patterns,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _analyze_single_tweet(self, text: str) -> Dict:
        """Analyze a single tweet."""
        return {
            'absolute_certainty': sum(1 for p in self.absolute_patterns if p.search(text)),
            'high_certainty': sum(1 for p in self.high_cert_patterns if p.search(text)),
            'hedged': sum(1 for p in self.hedge_patterns if p.search(text)),
            'fear_triggers': sum(1 for p in self.fear_patterns if p.search(text)),
            'greed_triggers': sum(1 for p in self.greed_patterns if p.search(text)),
            'urgency_triggers': sum(1 for p in self.urgency_patterns if p.search(text)),
            'word_count': len(text.split()),
            'avg_word_len': sum(len(w) for w in text.split()) / max(1, len(text.split())),
            'caps_ratio': len(re.findall(r'[A-Z]', text)) / max(1, len(text)),
        }

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract 3-4 word phrases for template detection."""
        words = text.lower().split()
        phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if len(phrase) > 10:  # Meaningful phrase
                phrases.append(phrase)
        return phrases

    def _determine_calibration(
        self,
        absolute: int,
        hedged: int,
        total: int
    ) -> str:
        """Determine certainty calibration."""
        if total < 5:
            return "insufficient_data"

        absolute_ratio = absolute / total
        hedged_ratio = hedged / total

        if absolute_ratio > 0.4:
            return "overconfident"
        elif hedged_ratio > 0.5:
            return "well_calibrated"
        elif absolute_ratio > 0.2:
            return "somewhat_overconfident"
        else:
            return "calibrated"

    def _calculate_complexity(
        self,
        avg_word_len: float,
        vocab_richness: float,
        jargon_density: float
    ) -> float:
        """Calculate language complexity score."""
        # Score based on multiple factors
        word_len_score = min(100, (avg_word_len - 3) * 20)  # Longer words = higher
        vocab_score = vocab_richness * 100
        jargon_score = min(100, jargon_density * 500)  # Moderate jargon is fine

        return (word_len_score + vocab_score + jargon_score) / 3

    def _calculate_manipulation_score(
        self,
        fear: int,
        greed: int,
        urgency: int,
        tweet_count: int
    ) -> float:
        """Calculate manipulation score."""
        if tweet_count == 0:
            return 0

        # Normalize by tweet count
        fear_per = fear / tweet_count
        greed_per = greed / tweet_count
        urgency_per = urgency / tweet_count

        # Weight: greed and urgency are more manipulative
        score = (fear_per * 15 + greed_per * 25 + urgency_per * 20) * 10
        return min(100, score)

    def _calculate_template_score(
        self,
        repeated: List[Tuple[str, int]],
        tweet_count: int
    ) -> float:
        """Calculate template usage score."""
        if tweet_count < 10:
            return 0

        # High repetition = templated content
        total_repeats = sum(count for _, count in repeated)
        repeat_ratio = total_repeats / tweet_count

        return min(100, repeat_ratio * 50)

    def _analyze_style_consistency(
        self,
        metrics: List[Dict]
    ) -> Tuple[float, bool]:
        """Analyze style consistency across tweets."""
        if len(metrics) < 10:
            return 0.0, False

        # Calculate variance in style metrics
        word_counts = [m['word_count'] for m in metrics]
        word_lens = [m['avg_word_len'] for m in metrics]
        caps_ratios = [m['caps_ratio'] for m in metrics]

        try:
            wc_cv = statistics.stdev(word_counts) / max(1, statistics.mean(word_counts))
            wl_cv = statistics.stdev(word_lens) / max(1, statistics.mean(word_lens))
            caps_cv = statistics.stdev(caps_ratios) / max(0.01, statistics.mean(caps_ratios))

            style_variance = (wc_cv + wl_cv + caps_cv) / 3

            # Very high variance could indicate multiple writers
            potential_ghostwriter = style_variance > 0.8

            return style_variance, potential_ghostwriter
        except statistics.StatisticsError:
            return 0.0, False

    def _calculate_authenticity_score(
        self,
        calibration: str,
        manipulation: float,
        template: float,
        variance: float,
        first_person: float
    ) -> float:
        """Calculate overall authenticity score."""
        score = 100.0

        # Calibration penalty
        if calibration == "overconfident":
            score -= 25
        elif calibration == "somewhat_overconfident":
            score -= 15

        # Manipulation penalty
        score -= manipulation * 0.4

        # Template penalty
        score -= template * 0.2

        # High variance is concerning (but not definitive)
        if variance > 0.5:
            score -= 10

        # First person usage is slightly positive (personal voice)
        if first_person > 0.03:
            score += 5

        return max(0, min(100, score))

    def _detect_patterns(
        self,
        certainty: float,
        manipulation: float,
        template: float,
        fear: int,
        greed: int,
        ghostwriter: bool
    ) -> List[str]:
        """Detect and describe patterns."""
        patterns = []

        if certainty > 70:
            patterns.append("Frequently uses absolute certainty language")

        if manipulation > 50:
            patterns.append("High use of emotional manipulation triggers")

        if fear > greed * 2:
            patterns.append("Primarily uses fear-based messaging (FUD)")
        elif greed > fear * 2:
            patterns.append("Primarily uses greed-based messaging (FOMO)")

        if template > 40:
            patterns.append("Repetitive/templated content structure")

        if ghostwriter:
            patterns.append("Inconsistent writing style (possible ghostwriter)")

        return patterns

    def _generate_flags(
        self,
        absolute: int,
        manipulation: float,
        template: float,
        calibration: str,
        variance: float
    ) -> Tuple[List[str], List[str]]:
        """Generate red and green flags."""
        red_flags = []
        green_flags = []

        if absolute > 10:
            red_flags.append(f"Excessive use of absolute claims ({absolute} instances)")

        if calibration == "overconfident":
            red_flags.append("Overconfident language - rarely hedges predictions")

        if manipulation > 60:
            red_flags.append("Heavy use of emotional manipulation (fear/greed/urgency)")

        if template > 50:
            red_flags.append("Highly templated content - possible automation")

        # Green flags
        if calibration == "well_calibrated":
            green_flags.append("Well-calibrated certainty - uses appropriate hedging")

        if manipulation < 20:
            green_flags.append("Low manipulation tactics - informative style")

        if template < 20:
            green_flags.append("Original content - not templated")

        return red_flags[:4], green_flags[:3]

    def generate_summary(self, report: LinguisticReport) -> str:
        """Generate human-readable summary."""
        parts = []

        if report.certainty_calibration == "overconfident":
            parts.append("Frequently overconfident in predictions.")
        elif report.certainty_calibration == "well_calibrated":
            parts.append("Uses appropriate hedging language.")

        if report.manipulation_score > 50:
            parts.append("Heavy use of emotional triggers.")

        if report.potential_ghostwriter:
            parts.append("Writing style inconsistencies detected.")

        if not parts:
            parts.append("Linguistic patterns appear normal.")

        return " ".join(parts)
