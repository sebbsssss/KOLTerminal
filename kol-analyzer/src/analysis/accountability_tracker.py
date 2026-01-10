"""
Accountability Tracker - Track whether KOLs own their predictions and mistakes.

Detects:
- Loss acknowledgment: Do they post about losses, not just wins?
- Prediction follow-up: Do they revisit old calls?
- Correction behavior: Do they correct misinformation?
- Win cherry-picking: Only posting wins, hiding losses
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict


@dataclass
class PredictionMention:
    """A mention of a past prediction or trade result."""
    tweet_id: str
    tweet_text: str
    timestamp: str
    mention_type: str  # "win", "loss", "followup", "correction", "deflection"
    token_mentioned: Optional[str]
    admits_mistake: bool
    self_aware: bool  # Shows awareness of past statement

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:120] + '...' if len(self.tweet_text) > 120 else self.tweet_text,
            'timestamp': self.timestamp,
            'mention_type': self.mention_type,
            'token_mentioned': self.token_mentioned,
            'admits_mistake': self.admits_mistake,
            'self_aware': self.self_aware
        }


@dataclass
class AccountabilityReport:
    """Report on accountability patterns."""
    accountability_score: float = 50.0  # 0-100

    # Loss acknowledgment
    wins_posted: int = 0
    losses_posted: int = 0
    win_loss_ratio: float = 0.0
    admits_losses: bool = False

    # Follow-up behavior
    followup_count: int = 0
    correction_count: int = 0
    deflection_count: int = 0  # Blames others, market, etc.

    # Accountability signals
    takes_responsibility: bool = False
    blames_others: bool = False
    deletes_bad_calls: bool = False  # Can't detect directly, inferred

    # Specific instances
    mentions: List[PredictionMention] = field(default_factory=list)

    # Analysis
    cherry_picks_wins: bool = False
    accountability_pattern: str = ""  # "accountable", "deflector", "silent", "cherry_picker"

    patterns_detected: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'accountability_score': round(self.accountability_score, 1),
            'wins_posted': self.wins_posted,
            'losses_posted': self.losses_posted,
            'win_loss_ratio': round(self.win_loss_ratio, 2),
            'admits_losses': self.admits_losses,
            'followup_count': self.followup_count,
            'correction_count': self.correction_count,
            'deflection_count': self.deflection_count,
            'takes_responsibility': self.takes_responsibility,
            'blames_others': self.blames_others,
            'cherry_picks_wins': self.cherry_picks_wins,
            'accountability_pattern': self.accountability_pattern,
            'mentions': [m.to_dict() for m in self.mentions[:15]],
            'patterns_detected': self.patterns_detected,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class AccountabilityTracker:
    """
    Tracks whether KOLs take accountability for their predictions.

    Key questions:
    1. Do they post about losses or only wins?
    2. Do they revisit old predictions?
    3. Do they correct mistakes or deflect blame?
    4. Do they cherry-pick their track record?
    """

    # Win mentions
    WIN_PATTERNS = [
        r'\b(nailed it|called it|told you|was right)\b',
        r'\b(profit|gains|made|up) (\d+%|\d+x)\b',
        r'\b(green|winning|win|won|crushing)\b',
        r'\b(easy money|free money|printing)\b',
        r'\b(to the moon|mooning|pumping)\b',
        r'\b(lfg|let\'s go|we did it)\b',
    ]

    # Loss mentions
    LOSS_PATTERNS = [
        r'\b(took a(n)? (loss|l)|lost|down|red)\b',
        r'\b(rekt|liquidated|stopped out|cut)\b',
        r'\b(wrong|mistake|shouldn\'t have)\b',
        r'\b(bad call|bad trade|missed)\b',
        r'\b(my fault|i messed up|lesson learned)\b',
        r'\b(paper hands|sold too early|sold too late)\b',
    ]

    # Follow-up language
    FOLLOWUP_PATTERNS = [
        r'\b(as i (said|mentioned|predicted))\b',
        r'\b(remember when i|i said (this|that))\b',
        r'\b(follow up|update on|revisiting)\b',
        r'\b(here\'s (how|what)|result)\b',
        r'\b(months ago|weeks ago|earlier)\b',
    ]

    # Correction/admission language
    CORRECTION_PATTERNS = [
        r'\b(i was wrong|was wrong about)\b',
        r'\b(need to correct|correction)\b',
        r'\b(changed my mind|reconsidering)\b',
        r'\b(admit|admitting|acknowledged)\b',
        r'\b(update|revised|correcting)\b',
        r'\b(apolog|sorry for)\b',
    ]

    # Deflection/blame language
    DEFLECTION_PATTERNS = [
        r'\b(market\'s fault|manipulation|coordinated)\b',
        r'\b(couldn\'t have known|no one could)\b',
        r'\b(whales|bots|algorithms)\b',
        r'\b(unlucky|bad timing|unexpected)\b',
        r'\b(they|them|the team) (rugged|dumped|lied)\b',
        r'\b(not my fault|wasn\'t me)\b',
    ]

    # Responsibility language
    RESPONSIBILITY_PATTERNS = [
        r'\b(my (fault|mistake|error))\b',
        r'\b(i (should|shouldn\'t) have)\b',
        r'\b(learned|lesson|next time)\b',
        r'\b(accountability|responsible)\b',
        r'\b(own it|taking the l)\b',
    ]

    # Token extraction
    TOKEN_PATTERN = r'\$([A-Z]{2,10})\b'

    def __init__(self):
        self.win_patterns = [re.compile(p, re.IGNORECASE) for p in self.WIN_PATTERNS]
        self.loss_patterns = [re.compile(p, re.IGNORECASE) for p in self.LOSS_PATTERNS]
        self.followup_patterns = [re.compile(p, re.IGNORECASE) for p in self.FOLLOWUP_PATTERNS]
        self.correction_patterns = [re.compile(p, re.IGNORECASE) for p in self.CORRECTION_PATTERNS]
        self.deflection_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEFLECTION_PATTERNS]
        self.responsibility_patterns = [re.compile(p, re.IGNORECASE) for p in self.RESPONSIBILITY_PATTERNS]
        self.token_pattern = re.compile(self.TOKEN_PATTERN)

    def analyze(self, tweets: List[dict]) -> AccountabilityReport:
        """Analyze accountability patterns in tweets."""
        if not tweets:
            return AccountabilityReport()

        mentions: List[PredictionMention] = []
        wins = 0
        losses = 0
        followups = 0
        corrections = 0
        deflections = 0
        responsibilities = 0

        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            if not text:
                continue

            # Extract tokens mentioned
            tokens = self.token_pattern.findall(text)
            token = tokens[0] if tokens else None

            # Check for win mentions
            is_win = any(p.search(text) for p in self.win_patterns)
            if is_win:
                wins += 1
                mentions.append(PredictionMention(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    mention_type="win",
                    token_mentioned=token,
                    admits_mistake=False,
                    self_aware=True
                ))

            # Check for loss mentions
            is_loss = any(p.search(text) for p in self.loss_patterns)
            if is_loss:
                losses += 1
                admits = any(p.search(text) for p in self.responsibility_patterns)
                mentions.append(PredictionMention(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    mention_type="loss",
                    token_mentioned=token,
                    admits_mistake=admits,
                    self_aware=True
                ))

            # Check for follow-ups
            is_followup = any(p.search(text) for p in self.followup_patterns)
            if is_followup:
                followups += 1
                mentions.append(PredictionMention(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    mention_type="followup",
                    token_mentioned=token,
                    admits_mistake=False,
                    self_aware=True
                ))

            # Check for corrections
            is_correction = any(p.search(text) for p in self.correction_patterns)
            if is_correction:
                corrections += 1
                mentions.append(PredictionMention(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    mention_type="correction",
                    token_mentioned=token,
                    admits_mistake=True,
                    self_aware=True
                ))

            # Check for deflections
            is_deflection = any(p.search(text) for p in self.deflection_patterns)
            if is_deflection and (is_loss or is_correction):
                deflections += 1
                # Update the last mention to be a deflection
                if mentions and mentions[-1].tweet_id == tweet_id:
                    mentions[-1] = PredictionMention(
                        tweet_id=tweet_id,
                        tweet_text=text,
                        timestamp=timestamp,
                        mention_type="deflection",
                        token_mentioned=token,
                        admits_mistake=False,
                        self_aware=False
                    )

            # Check for taking responsibility
            if any(p.search(text) for p in self.responsibility_patterns):
                responsibilities += 1

        # Calculate metrics
        total_results = wins + losses
        win_loss_ratio = wins / losses if losses > 0 else float('inf') if wins > 0 else 0

        # Determine patterns
        admits_losses = losses > 0
        takes_responsibility = responsibilities > deflections
        blames_others = deflections > responsibilities
        cherry_picks = wins > 5 and losses == 0 and total_results > 0

        # Determine accountability pattern
        accountability_pattern = self._determine_pattern(
            wins, losses, followups, corrections, deflections,
            takes_responsibility, cherry_picks
        )

        # Calculate score
        accountability_score = self._calculate_score(
            wins, losses, followups, corrections, deflections,
            takes_responsibility, cherry_picks, len(tweets)
        )

        # Generate analysis
        patterns_detected = self._detect_patterns(
            wins, losses, followups, corrections, deflections,
            cherry_picks, accountability_pattern
        )

        red_flags, green_flags = self._generate_flags(
            wins, losses, corrections, deflections,
            cherry_picks, takes_responsibility, accountability_pattern
        )

        return AccountabilityReport(
            accountability_score=accountability_score,
            wins_posted=wins,
            losses_posted=losses,
            win_loss_ratio=win_loss_ratio,
            admits_losses=admits_losses,
            followup_count=followups,
            correction_count=corrections,
            deflection_count=deflections,
            takes_responsibility=takes_responsibility,
            blames_others=blames_others,
            cherry_picks_wins=cherry_picks,
            accountability_pattern=accountability_pattern,
            mentions=mentions,
            patterns_detected=patterns_detected,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _determine_pattern(
        self,
        wins: int,
        losses: int,
        followups: int,
        corrections: int,
        deflections: int,
        responsible: bool,
        cherry_picks: bool
    ) -> str:
        """Determine primary accountability pattern."""
        if cherry_picks:
            return "cherry_picker"

        if corrections > 2 and responsible:
            return "accountable"

        if deflections > corrections and deflections > 2:
            return "deflector"

        if wins > 3 and losses == 0 and followups == 0:
            return "silent_on_losses"

        if followups > 3 and (corrections > 0 or losses > 0):
            return "transparent"

        return "neutral"

    def _calculate_score(
        self,
        wins: int,
        losses: int,
        followups: int,
        corrections: int,
        deflections: int,
        responsible: bool,
        cherry_picks: bool,
        total_tweets: int
    ) -> float:
        """Calculate accountability score."""
        score = 50.0  # Start neutral

        # Posting losses is good
        if losses > 0:
            score += min(20, losses * 5)

        # Follow-ups are good
        score += min(15, followups * 3)

        # Corrections are very good
        score += min(20, corrections * 10)

        # Deflections are bad
        score -= min(25, deflections * 5)

        # Cherry-picking is very bad
        if cherry_picks:
            score -= 30

        # Taking responsibility is good
        if responsible:
            score += 15

        # Extreme win/loss ratio is suspicious
        if wins > 0 and losses == 0 and wins > 5:
            score -= 20  # Unrealistic

        return max(0, min(100, score))

    def _detect_patterns(
        self,
        wins: int,
        losses: int,
        followups: int,
        corrections: int,
        deflections: int,
        cherry_picks: bool,
        pattern: str
    ) -> List[str]:
        """Detect and describe patterns."""
        patterns = []

        if cherry_picks:
            patterns.append(f"Only posts wins ({wins} wins, 0 losses) - cherry-picking")

        if wins > 0 and losses > 0:
            ratio = wins / losses
            if ratio > 10:
                patterns.append(f"Extreme win/loss ratio ({ratio:.1f}:1)")
            elif 0.5 < ratio < 2:
                patterns.append("Balanced win/loss posting - honest about results")

        if followups > 3:
            patterns.append(f"Frequently revisits old predictions ({followups} follow-ups)")

        if corrections > 2:
            patterns.append(f"Corrects mistakes publicly ({corrections} corrections)")

        if deflections > corrections:
            patterns.append("More likely to blame others than take responsibility")

        if pattern == "accountable":
            patterns.append("Shows strong accountability for predictions")
        elif pattern == "deflector":
            patterns.append("Tends to deflect blame when wrong")
        elif pattern == "silent_on_losses":
            patterns.append("Goes silent on losing trades")

        return patterns

    def _generate_flags(
        self,
        wins: int,
        losses: int,
        corrections: int,
        deflections: int,
        cherry_picks: bool,
        responsible: bool,
        pattern: str
    ) -> Tuple[List[str], List[str]]:
        """Generate red and green flags."""
        red_flags = []
        green_flags = []

        if cherry_picks:
            red_flags.append(f"Cherry-picks wins only ({wins} wins posted, 0 losses)")

        if deflections > 3:
            red_flags.append(f"Frequently deflects blame ({deflections} instances)")

        if wins > 10 and losses == 0:
            red_flags.append("Unrealistic track record - likely hiding losses")

        if pattern == "deflector":
            red_flags.append("Blames others when predictions fail")

        # Green flags
        if losses > 0 and responsible:
            green_flags.append("Admits losses and takes responsibility")

        if corrections > 2:
            green_flags.append("Publicly corrects mistakes")

        if pattern == "accountable":
            green_flags.append("Strong accountability - revisits and owns predictions")

        if pattern == "transparent":
            green_flags.append("Transparent about trading results")

        if 0.5 < (wins / max(1, losses)) < 3 and (wins + losses) > 5:
            green_flags.append("Realistic win/loss ratio - honest reporting")

        return red_flags[:4], green_flags[:3]

    def generate_summary(self, report: AccountabilityReport) -> str:
        """Generate human-readable summary."""
        parts = []

        if report.cherry_picks_wins:
            parts.append("Only posts about wins - likely hiding losses.")
        elif report.admits_losses and report.takes_responsibility:
            parts.append("Takes accountability for both wins and losses.")
        elif report.blames_others:
            parts.append("Tends to blame others when wrong.")

        if report.correction_count > 0:
            parts.append(f"Has publicly corrected mistakes ({report.correction_count} times).")

        if report.followup_count > 3:
            parts.append("Regularly follows up on predictions.")

        if not parts:
            parts.append("Insufficient data on accountability patterns.")

        return " ".join(parts)
