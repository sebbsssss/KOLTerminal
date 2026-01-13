"""
Asshole Meter Analyzer

Measures how toxic, rude, or unpleasant a KOL is in their communications.
This analyzes personality/behavior rather than credibility.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import Counter


@dataclass
class AssholeAnalysis:
    """Results from asshole analysis."""
    asshole_score: float = 50.0  # 0 = saint, 100 = toxic
    toxicity_level: str = "mid"  # saint, chill, mid, prickly, toxic
    toxicity_emoji: str = "😐"

    # Sub-scores
    insult_score: float = 0.0
    condescension_score: float = 0.0
    ego_score: float = 0.0
    empathy_score: float = 100.0  # Higher = more empathetic (inverted for final)
    dismissiveness_score: float = 0.0

    # Detected patterns
    toxic_phrases: List[str] = field(default_factory=list)
    ego_phrases: List[str] = field(default_factory=list)
    helpful_phrases: List[str] = field(default_factory=list)

    # Summary
    personality_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asshole_score": self.asshole_score,
            "toxicity_level": self.toxicity_level,
            "toxicity_emoji": self.toxicity_emoji,
            "insult_score": self.insult_score,
            "condescension_score": self.condescension_score,
            "ego_score": self.ego_score,
            "empathy_score": self.empathy_score,
            "dismissiveness_score": self.dismissiveness_score,
            "toxic_phrases": self.toxic_phrases[:10],
            "ego_phrases": self.ego_phrases[:10],
            "helpful_phrases": self.helpful_phrases[:10],
            "personality_summary": self.personality_summary
        }


class AssholeAnalyzer:
    """Analyzes how much of an asshole a KOL is."""

    # Toxic phrases and insults
    INSULT_PATTERNS = [
        r'\b(idiot|stupid|dumb|moron|retard|loser|clown|fool|pathetic)\b',
        r'\b(ngmi|hfsp|stay poor|have fun staying poor|cope|seethe|mald)\b',
        r'\b(absolute (trash|garbage|shit)|brain ?dead|smooth ?brain)\b',
        r'\b(you.re (wrong|stupid|dumb)|shut up|stfu|gtfo)\b',
        r'\b(lmao (imagine|you)|imagine (not|being|thinking))\b',
        r'\b(skill issue|cry about it|die mad|stay mad)\b',
        r'\b(peasants?|poors?|broke (boys?|people))\b',
    ]

    # Condescending patterns
    CONDESCENSION_PATTERNS = [
        r'\b(obviously|clearly you|if you (knew|understood)|learn to)\b',
        r'\b(basic (concept|understanding)|not that hard|simple (concept|math))\b',
        r'\b(do your (own )?research|figure it out|not my (job|problem))\b',
        r'\b(imagine not knowing|how (do|can) you not|everyone knows)\b',
        r'\b(let me (explain|educate)|since you (clearly|obviously))\b',
        r'\b(its not (that|rocket) science|common sense|basic logic)\b',
        r'\b(youre (clearly|obviously) (new|wrong|confused))\b',
    ]

    # Ego/bragging patterns
    EGO_PATTERNS = [
        r'\b(i (called|predicted|told you)|my (call|prediction) was)\b',
        r'\b(i was right|proved me right|vindicat(ed|ion))\b',
        r'\b(as i (said|predicted|warned)|like i said)\b',
        r'\b(follow(ed)? my (advice|call)|listen(ed)? to me)\b',
        r'\b(another (win|W)|easy (money|win|call))\b',
        r'\b(you.re welcome|thank me later|told you so)\b',
        r'\b(i (never|dont) miss|always (right|winning))\b',
        r'\b(\d+x|\d+ ?x|made \d+k|up \d+%)\b',  # Gain bragging
    ]

    # Low empathy patterns (mocking losses, dismissing concerns)
    LOW_EMPATHY_PATTERNS = [
        r'\b(rekt|get rekt|shoulda|shouldve|your (fault|problem))\b',
        r'\b(sucks to (be you|suck)|not my (problem|fault))\b',
        r'\b(you (deserve|asked for) (it|this)|play stupid games)\b',
        r'\b(cry(ing)?|tears|cope|copium|coping)\b',
        r'\b(paper hands|weak hands|sold too (early|soon))\b',
        r'\b(if you (lost|got rekt)|thats on you)\b',
        r'\b(unlucky|bad luck|skill issue|deserved)\b',
    ]

    # Dismissive patterns
    DISMISSIVE_PATTERNS = [
        r'\b(dont care|idgaf|idc|who (asked|cares)|nobody (asked|cares))\b',
        r'\b(whatever|cool story|ok and\??|so\??|and\??)\b',
        r'\b(not reading (that|this)|tldr|too long)\b',
        r'\b(blocked|muted|bye|next|moving on)\b',
        r'\b(ratio|L|take the L|hold this L)\b',
        r'\b(irrelevant|doesnt matter|who are you)\b',
    ]

    # Helpful/positive patterns (reduces asshole score)
    HELPFUL_PATTERNS = [
        r'\b(happy to (help|explain)|let me (help|know))\b',
        r'\b(good (question|point)|thats (fair|valid))\b',
        r'\b(i (could be|might be|may be) wrong)\b',
        r'\b(thanks for|appreciate|grateful)\b',
        r'\b(heres (how|what|why)|hope this helps)\b',
        r'\b(dm me|reach out|ask me anything|ama)\b',
        r'\b(congrat(s|ulations)|well done|nice (job|work))\b',
        r'\b(sorry (to hear|for|about)|that sucks|feel for you)\b',
        r'\b(not financial advice|nfa|dyor|do your (own )?research)\b',
        r'\b(be careful|stay safe|manage (risk|your position))\b',
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.insult_re = [re.compile(p, re.IGNORECASE) for p in self.INSULT_PATTERNS]
        self.condescension_re = [re.compile(p, re.IGNORECASE) for p in self.CONDESCENSION_PATTERNS]
        self.ego_re = [re.compile(p, re.IGNORECASE) for p in self.EGO_PATTERNS]
        self.low_empathy_re = [re.compile(p, re.IGNORECASE) for p in self.LOW_EMPATHY_PATTERNS]
        self.dismissive_re = [re.compile(p, re.IGNORECASE) for p in self.DISMISSIVE_PATTERNS]
        self.helpful_re = [re.compile(p, re.IGNORECASE) for p in self.HELPFUL_PATTERNS]

    def analyze(self, tweets: List[Dict[str, Any]], username: str = "") -> AssholeAnalysis:
        """Analyze tweets for asshole behavior."""
        result = AssholeAnalysis()

        if not tweets:
            result.personality_summary = "Not enough data to assess personality."
            return result

        # Count pattern matches
        insult_count = 0
        condescension_count = 0
        ego_count = 0
        low_empathy_count = 0
        dismissive_count = 0
        helpful_count = 0

        toxic_examples = []
        ego_examples = []
        helpful_examples = []

        for tweet in tweets:
            text = tweet.get('text', '')
            if not text:
                continue

            # Clean text
            text_clean = text.lower()

            # Check insults
            for pattern in self.insult_re:
                matches = pattern.findall(text_clean)
                if matches:
                    insult_count += len(matches)
                    toxic_examples.append(text[:100])

            # Check condescension
            for pattern in self.condescension_re:
                matches = pattern.findall(text_clean)
                if matches:
                    condescension_count += len(matches)
                    if text[:100] not in toxic_examples:
                        toxic_examples.append(text[:100])

            # Check ego
            for pattern in self.ego_re:
                matches = pattern.findall(text_clean)
                if matches:
                    ego_count += len(matches)
                    ego_examples.append(text[:100])

            # Check low empathy
            for pattern in self.low_empathy_re:
                matches = pattern.findall(text_clean)
                if matches:
                    low_empathy_count += len(matches)

            # Check dismissiveness
            for pattern in self.dismissive_re:
                matches = pattern.findall(text_clean)
                if matches:
                    dismissive_count += len(matches)

            # Check helpful (reduces score)
            for pattern in self.helpful_re:
                matches = pattern.findall(text_clean)
                if matches:
                    helpful_count += len(matches)
                    helpful_examples.append(text[:100])

        # Calculate sub-scores (normalized to 0-100)
        total_tweets = len(tweets)

        # Insult score: heavily weighted
        result.insult_score = min(100, (insult_count / total_tweets) * 500)

        # Condescension score
        result.condescension_score = min(100, (condescension_count / total_tweets) * 300)

        # Ego score
        result.ego_score = min(100, (ego_count / total_tweets) * 200)

        # Empathy score (inverse - more helpful = more empathetic)
        empathy_negative = (low_empathy_count / total_tweets) * 400
        empathy_positive = (helpful_count / total_tweets) * 200
        result.empathy_score = max(0, min(100, 70 + empathy_positive - empathy_negative))

        # Dismissiveness score
        result.dismissiveness_score = min(100, (dismissive_count / total_tweets) * 300)

        # Calculate overall asshole score
        # Weight: insults (30%), condescension (20%), ego (15%), low empathy (20%), dismissive (15%)
        raw_score = (
            result.insult_score * 0.30 +
            result.condescension_score * 0.20 +
            result.ego_score * 0.15 +
            (100 - result.empathy_score) * 0.20 +  # Invert empathy
            result.dismissiveness_score * 0.15
        )

        # Bonus reduction for helpful behavior
        helpful_bonus = min(20, (helpful_count / total_tweets) * 100)
        raw_score = max(0, raw_score - helpful_bonus)

        result.asshole_score = min(100, max(0, raw_score))

        # Determine toxicity level
        if result.asshole_score < 20:
            result.toxicity_level = "saint"
            result.toxicity_emoji = "😇"
        elif result.asshole_score < 40:
            result.toxicity_level = "chill"
            result.toxicity_emoji = "😊"
        elif result.asshole_score < 60:
            result.toxicity_level = "mid"
            result.toxicity_emoji = "😐"
        elif result.asshole_score < 80:
            result.toxicity_level = "prickly"
            result.toxicity_emoji = "😤"
        else:
            result.toxicity_level = "toxic"
            result.toxicity_emoji = "🤬"

        # Store examples (deduplicated)
        result.toxic_phrases = list(set(toxic_examples))[:10]
        result.ego_phrases = list(set(ego_examples))[:10]
        result.helpful_phrases = list(set(helpful_examples))[:10]

        # Generate personality summary
        result.personality_summary = self._generate_summary(result, username)

        return result

    def _generate_summary(self, result: AssholeAnalysis, username: str) -> str:
        """Generate a personality summary."""
        summaries = {
            "saint": f"@{username} is genuinely helpful and patient. Rare to see someone this nice on CT.",
            "chill": f"@{username} is generally chill and approachable. Occasional snark but nothing serious.",
            "mid": f"@{username} has average social behavior. Mix of helpful and dismissive moments.",
            "prickly": f"@{username} can be quick to attack and has thin skin. Approach with caution.",
            "toxic": f"@{username} shows consistently toxic behavior. High levels of condescension and dismissiveness."
        }

        base = summaries.get(result.toxicity_level, "Unable to assess personality.")

        # Add specific observations
        observations = []

        if result.insult_score > 50:
            observations.append("frequently uses insults")
        if result.condescension_score > 50:
            observations.append("talks down to others")
        if result.ego_score > 60:
            observations.append("brags about wins often")
        if result.empathy_score < 40:
            observations.append("shows little empathy for losses")
        if result.dismissiveness_score > 50:
            observations.append("dismisses criticism quickly")
        if result.empathy_score > 70 and result.asshole_score < 40:
            observations.append("shows genuine care for followers")

        if observations:
            base += f" Notable: {', '.join(observations)}."

        return base
