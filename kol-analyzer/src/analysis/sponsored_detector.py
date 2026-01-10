"""
Sponsored Content Detector - Detect paid promotions and undisclosed sponsorships.

Identifies:
- Disclosed sponsorships (#ad, #sponsored, etc.)
- Undisclosed promotions (shill patterns without disclosure)
- Affiliate links and referral codes
- Project team affiliations
- Coordinated promotion patterns
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter


class SponsoredType:
    """Types of sponsored content."""
    DISCLOSED_AD = "disclosed_ad"  # Properly disclosed #ad
    UNDISCLOSED_SHILL = "undisclosed_shill"  # Promotion without disclosure
    AFFILIATE_LINK = "affiliate_link"  # Referral/affiliate content
    PROJECT_TEAM = "project_team"  # Working for the project
    AMBASSADOR = "ambassador"  # Official ambassador role
    AIRDROP_PROMO = "airdrop_promo"  # Promoting for airdrop
    COORDINATED = "coordinated"  # Part of coordinated campaign


@dataclass
class SponsoredInstance:
    """A detected sponsored content instance."""
    tweet_id: str
    tweet_text: str
    timestamp: str
    sponsored_type: str
    indicators: List[str]
    project_mentioned: Optional[str]
    severity: str  # "disclosed" (good), "suspicious", "undisclosed" (bad)
    is_properly_disclosed: bool

    def to_dict(self) -> dict:
        return {
            'tweet_id': self.tweet_id,
            'tweet_text': self.tweet_text[:150] + '...' if len(self.tweet_text) > 150 else self.tweet_text,
            'timestamp': self.timestamp,
            'sponsored_type': self.sponsored_type,
            'indicators': self.indicators,
            'project_mentioned': self.project_mentioned,
            'severity': self.severity,
            'is_properly_disclosed': self.is_properly_disclosed
        }


@dataclass
class SponsoredReport:
    """Report on sponsored content detection."""
    instances: List[SponsoredInstance] = field(default_factory=list)
    transparency_score: float = 100.0  # 100 = fully transparent, 0 = never discloses
    total_promotional: int = 0
    disclosed_count: int = 0
    undisclosed_count: int = 0
    affiliate_count: int = 0
    disclosure_rate: float = 100.0  # % of promotions that are disclosed
    frequently_promoted: List[Tuple[str, int]] = field(default_factory=list)  # Projects promoted
    promotion_patterns: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'instances': [i.to_dict() for i in self.instances[:20]],
            'transparency_score': round(self.transparency_score, 1),
            'total_promotional': self.total_promotional,
            'disclosed_count': self.disclosed_count,
            'undisclosed_count': self.undisclosed_count,
            'affiliate_count': self.affiliate_count,
            'disclosure_rate': round(self.disclosure_rate, 1),
            'frequently_promoted': self.frequently_promoted[:10],
            'promotion_patterns': self.promotion_patterns,
            'red_flags': self.red_flags,
            'green_flags': self.green_flags
        }


class SponsoredDetector:
    """
    Detects sponsored content and evaluates disclosure transparency.

    Detection methodology:
    1. Look for disclosure hashtags (#ad, #sponsored, etc.)
    2. Detect promotional language patterns
    3. Find affiliate/referral indicators
    4. Identify project team/ambassador roles
    5. Detect coordinated promotion patterns (same project, timing)
    """

    # Proper disclosure patterns (good)
    DISCLOSURE_PATTERNS = [
        r'#(ad|advertisement|sponsored|paidpartnership)',
        r'\b(sponsored|paid partnership|paid promotion)\b',
        r'\b(this is (an )?ad|advertising)\b',
        r'\b(ambassador|brand ambassador)\b',
        r'\b(working with|partnered with|in partnership with)\b',
        r'\b(disclosure|disclosing|i am paid)\b',
    ]

    # Promotional language (needs disclosure if frequent)
    PROMO_PATTERNS = [
        r'\b(check out|go check|must check)\b',
        r'\b(use (my|this) (code|link|referral))\b',
        r'\b(sign up (with|using)|register (with|using))\b',
        r'\b(don\'t miss|limited time|exclusive)\b',
        r'\b(best (dex|exchange|platform|wallet))\b',
        r'\b(i (use|recommend|suggest|endorse))\b',
        r'\b(game.?changer|revolutionary|innovative)\b',
        r'\b(join (the|my)|be (early|first))\b',
    ]

    # Strong shill indicators
    SHILL_PATTERNS = [
        r'\b(100x|1000x|10x potential)\b.*\$(token|project)',
        r'\b(next (eth|sol|btc)|the next)\b',
        r'\b(hidden gem|alpha (call|leak))\b',
        r'\b(before it (moons|pumps|explodes))\b',
        r'\b(early entry|get in early)\b',
        r'\b(not financial advice|nfa)\b.*\$(token)',  # NFA + specific token = suspicious
        r'\b(aping|aped|going all in)\b',
        r'\b(easiest (money|x|gains))\b',
    ]

    # Affiliate/referral indicators
    AFFILIATE_PATTERNS = [
        r'\?ref=',
        r'\?r=',
        r'\?code=',
        r'referral[_-]?code',
        r'/r/[a-zA-Z0-9]+',
        r'\b(my link|my code|my referral)\b',
        r'\b(use code|promo code|discount code)\b',
        r'\b(affiliate|partner link)\b',
        r'bit\.ly|t\.co|tinyurl|shorturl',  # URL shorteners often used for affiliate
    ]

    # Project team indicators
    TEAM_PATTERNS = [
        r'\b(i work (for|at|with)|working (for|at|with))\b',
        r'\b(our (team|project|token|protocol))\b',
        r'\b(we (are|\'re) (building|launching|releasing))\b',
        r'\b(join (us|our)|our community)\b',
        r'\b(i\'m (the|a) (founder|co-?founder|ceo|cto|dev|developer))\b',
        r'\b(building|built) this\b',
    ]

    # Ambassador role indicators
    AMBASSADOR_PATTERNS = [
        r'\b(ambassador|brand rep|representative)\b',
        r'\b(proud to (announce|partner|work))\b',
        r'\b(official (partner|supporter))\b',
        r'\b(sponsored by|brought to you by)\b',
    ]

    # Airdrop promotion patterns
    AIRDROP_PATTERNS = [
        r'\b(airdrop|air drop)\b',
        r'\b(farm(ing)?|farming points)\b',
        r'\b(testnet|incentivized testnet)\b',
        r'\b(tasks?|quests?|missions?)\b.*\b(complete|finish|do)\b',
        r'\b(galxe|zealy|crew3|layer3|intract)\b',
        r'\b(points|rewards?)\b.*\b(earn|collect|accumulate)\b',
    ]

    # Project name extraction
    PROJECT_PATTERN = r'@([A-Za-z0-9_]+)|(?:\$([A-Z]{2,10}))|(?:#([A-Za-z0-9]+))'

    def __init__(self):
        self.disclosure_patterns = [re.compile(p, re.IGNORECASE) for p in self.DISCLOSURE_PATTERNS]
        self.promo_patterns = [re.compile(p, re.IGNORECASE) for p in self.PROMO_PATTERNS]
        self.shill_patterns = [re.compile(p, re.IGNORECASE) for p in self.SHILL_PATTERNS]
        self.affiliate_patterns = [re.compile(p, re.IGNORECASE) for p in self.AFFILIATE_PATTERNS]
        self.team_patterns = [re.compile(p, re.IGNORECASE) for p in self.TEAM_PATTERNS]
        self.ambassador_patterns = [re.compile(p, re.IGNORECASE) for p in self.AMBASSADOR_PATTERNS]
        self.airdrop_patterns = [re.compile(p, re.IGNORECASE) for p in self.AIRDROP_PATTERNS]
        self.project_pattern = re.compile(self.PROJECT_PATTERN)

    def analyze(self, tweets: List[dict]) -> SponsoredReport:
        """
        Analyze tweets for sponsored content.

        Args:
            tweets: List of tweet dictionaries

        Returns:
            SponsoredReport with analysis
        """
        if not tweets:
            return SponsoredReport()

        instances: List[SponsoredInstance] = []
        project_counts: Counter = Counter()
        project_timestamps: Dict[str, List[str]] = {}  # For coordinated detection

        for tweet in tweets:
            text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            timestamp = tweet.get('timestamp', '')

            # Extract mentioned projects
            projects = self._extract_projects(text)

            # Check for disclosure
            has_disclosure = any(p.search(text) for p in self.disclosure_patterns)

            # Check various sponsored types
            detection = self._detect_sponsored_type(text, has_disclosure)

            if detection:
                sponsored_type, indicators, is_disclosed = detection

                # Determine primary project
                primary_project = projects[0] if projects else None

                # Track project mentions
                for project in projects:
                    project_counts[project] += 1
                    if project not in project_timestamps:
                        project_timestamps[project] = []
                    project_timestamps[project].append(timestamp)

                # Determine severity
                if is_disclosed:
                    severity = "disclosed"
                elif sponsored_type in [SponsoredType.UNDISCLOSED_SHILL, SponsoredType.COORDINATED]:
                    severity = "undisclosed"
                else:
                    severity = "suspicious"

                instances.append(SponsoredInstance(
                    tweet_id=tweet_id,
                    tweet_text=text,
                    timestamp=timestamp,
                    sponsored_type=sponsored_type,
                    indicators=indicators,
                    project_mentioned=primary_project,
                    severity=severity,
                    is_properly_disclosed=is_disclosed
                ))

        # Detect coordinated promotion patterns
        coordinated = self._detect_coordinated_patterns(project_timestamps)

        # Calculate statistics
        total_promotional = len(instances)
        disclosed_count = sum(1 for i in instances if i.is_properly_disclosed)
        undisclosed_count = sum(1 for i in instances if i.severity == "undisclosed")
        affiliate_count = sum(1 for i in instances if i.sponsored_type == SponsoredType.AFFILIATE_LINK)

        disclosure_rate = (disclosed_count / total_promotional * 100) if total_promotional > 0 else 100.0

        # Frequently promoted projects
        frequently_promoted = project_counts.most_common(10)

        # Generate flags
        red_flags = self._generate_red_flags(
            instances, undisclosed_count, disclosure_rate, coordinated
        )
        green_flags = self._generate_green_flags(
            instances, disclosed_count, disclosure_rate
        )

        # Promotion patterns
        promotion_patterns = self._identify_patterns(instances, coordinated)

        # Calculate transparency score
        transparency_score = self._calculate_transparency_score(
            disclosure_rate, undisclosed_count, total_promotional, len(coordinated)
        )

        return SponsoredReport(
            instances=instances,
            transparency_score=transparency_score,
            total_promotional=total_promotional,
            disclosed_count=disclosed_count,
            undisclosed_count=undisclosed_count,
            affiliate_count=affiliate_count,
            disclosure_rate=disclosure_rate,
            frequently_promoted=frequently_promoted,
            promotion_patterns=promotion_patterns,
            red_flags=red_flags,
            green_flags=green_flags
        )

    def _extract_projects(self, text: str) -> List[str]:
        """Extract project names from text."""
        projects = []
        matches = self.project_pattern.findall(text)

        for match in matches:
            # match is a tuple of (handle, ticker, hashtag)
            project = match[0] or match[1] or match[2]
            if project and len(project) > 1:
                projects.append(project.upper())

        return list(set(projects))

    def _detect_sponsored_type(
        self,
        text: str,
        has_disclosure: bool
    ) -> Optional[Tuple[str, List[str], bool]]:
        """
        Detect if content is sponsored and what type.

        Returns: (type, indicators, is_disclosed) or None
        """
        indicators = []

        # Check for affiliate links (always promotional)
        affiliate_matches = [p.pattern for p in self.affiliate_patterns if p.search(text)]
        if affiliate_matches:
            indicators.extend(affiliate_matches[:2])
            return (SponsoredType.AFFILIATE_LINK, indicators, has_disclosure)

        # Check for team membership
        team_matches = [p.pattern for p in self.team_patterns if p.search(text)]
        if team_matches:
            indicators.extend(team_matches[:2])
            return (SponsoredType.PROJECT_TEAM, indicators, True)  # Team = disclosed by nature

        # Check for ambassador role
        ambassador_matches = [p.pattern for p in self.ambassador_patterns if p.search(text)]
        if ambassador_matches:
            indicators.extend(ambassador_matches[:2])
            return (SponsoredType.AMBASSADOR, indicators, has_disclosure)

        # Check for airdrop promotion
        airdrop_matches = [p.pattern for p in self.airdrop_patterns if p.search(text)]
        if airdrop_matches:
            indicators.extend(airdrop_matches[:2])
            return (SponsoredType.AIRDROP_PROMO, indicators, has_disclosure)

        # Check for strong shill patterns
        shill_matches = [p.pattern for p in self.shill_patterns if p.search(text)]
        if shill_matches:
            indicators.extend(shill_matches[:2])
            if has_disclosure:
                return (SponsoredType.DISCLOSED_AD, indicators, True)
            else:
                return (SponsoredType.UNDISCLOSED_SHILL, indicators, False)

        # Check for general promotional content
        promo_matches = [p.pattern for p in self.promo_patterns if p.search(text)]
        if len(promo_matches) >= 2:  # Multiple promo indicators
            indicators.extend(promo_matches[:2])
            if has_disclosure:
                return (SponsoredType.DISCLOSED_AD, indicators, True)
            else:
                return (SponsoredType.UNDISCLOSED_SHILL, indicators, False)

        # If has disclosure but no other matches, it's a disclosed ad
        if has_disclosure:
            return (SponsoredType.DISCLOSED_AD, ["explicit disclosure"], True)

        return None

    def _detect_coordinated_patterns(
        self,
        project_timestamps: Dict[str, List[str]]
    ) -> List[str]:
        """Detect coordinated promotion patterns."""
        coordinated_projects = []

        for project, timestamps in project_timestamps.items():
            if len(timestamps) < 3:
                continue

            # Check for burst patterns (many mentions in short time)
            try:
                dates = sorted([
                    datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    for ts in timestamps
                ])

                # Count mentions in 24-hour windows
                for i, date in enumerate(dates):
                    window_count = sum(
                        1 for d in dates
                        if 0 <= (d - date).total_seconds() <= 86400
                    )
                    if window_count >= 5:  # 5+ mentions in 24 hours
                        coordinated_projects.append(project)
                        break

            except (ValueError, TypeError):
                continue

        return list(set(coordinated_projects))

    def _generate_red_flags(
        self,
        instances: List[SponsoredInstance],
        undisclosed_count: int,
        disclosure_rate: float,
        coordinated: List[str]
    ) -> List[str]:
        """Generate red flags for sponsored content."""
        flags = []

        if undisclosed_count > 5:
            flags.append(f"High undisclosed promotional content ({undisclosed_count} instances)")

        if disclosure_rate < 50 and len(instances) > 5:
            flags.append(f"Low disclosure rate ({disclosure_rate:.0f}%) on promotional content")

        if coordinated:
            flags.append(f"Coordinated promotion patterns for: {', '.join(coordinated[:3])}")

        shill_count = sum(1 for i in instances if i.sponsored_type == SponsoredType.UNDISCLOSED_SHILL)
        if shill_count > 3:
            flags.append(f"Frequent undisclosed shilling ({shill_count} instances)")

        affiliate_undisclosed = sum(
            1 for i in instances
            if i.sponsored_type == SponsoredType.AFFILIATE_LINK and not i.is_properly_disclosed
        )
        if affiliate_undisclosed > 2:
            flags.append("Undisclosed affiliate links detected")

        return flags[:5]

    def _generate_green_flags(
        self,
        instances: List[SponsoredInstance],
        disclosed_count: int,
        disclosure_rate: float
    ) -> List[str]:
        """Generate green flags for transparency."""
        flags = []

        if disclosure_rate >= 90 and len(instances) > 3:
            flags.append("Excellent disclosure practices")

        if disclosure_rate >= 75 and len(instances) > 5:
            flags.append("Good transparency on sponsored content")

        if disclosed_count > 0 and disclosed_count == len(instances):
            flags.append("All promotional content properly disclosed")

        if not instances:
            flags.append("No promotional content detected")

        return flags[:3]

    def _identify_patterns(
        self,
        instances: List[SponsoredInstance],
        coordinated: List[str]
    ) -> List[str]:
        """Identify promotional patterns."""
        patterns = []

        type_counts = Counter(i.sponsored_type for i in instances)

        if type_counts.get(SponsoredType.AFFILIATE_LINK, 0) > 3:
            patterns.append("Frequent affiliate marketing")

        if type_counts.get(SponsoredType.AIRDROP_PROMO, 0) > 5:
            patterns.append("Heavy airdrop promotion")

        if type_counts.get(SponsoredType.UNDISCLOSED_SHILL, 0) > type_counts.get(SponsoredType.DISCLOSED_AD, 0):
            patterns.append("Prefers undisclosed promotions")

        if coordinated:
            patterns.append("Shows coordinated campaign behavior")

        if type_counts.get(SponsoredType.PROJECT_TEAM, 0) > 0:
            patterns.append("Project team member")

        return patterns[:4]

    def _calculate_transparency_score(
        self,
        disclosure_rate: float,
        undisclosed_count: int,
        total_promotional: int,
        coordinated_count: int
    ) -> float:
        """Calculate transparency score."""
        if total_promotional == 0:
            return 100.0  # No promotional content = fully transparent

        # Base from disclosure rate
        score = disclosure_rate

        # Penalty for undisclosed content
        undisclosed_penalty = min(30, undisclosed_count * 5)
        score -= undisclosed_penalty

        # Penalty for coordinated campaigns
        coordinated_penalty = min(20, coordinated_count * 10)
        score -= coordinated_penalty

        return max(0.0, min(100.0, score))

    def generate_summary(self, report: SponsoredReport) -> str:
        """Generate human-readable summary."""
        if report.total_promotional == 0:
            return "No promotional content detected. Account appears organic."

        parts = []

        # Main finding
        if report.disclosure_rate >= 80:
            parts.append("Good transparency on sponsored content.")
        elif report.disclosure_rate >= 50:
            parts.append("Mixed disclosure practices on promotions.")
        else:
            parts.append("Poor disclosure of sponsored content.")

        # Stats
        parts.append(
            f"{report.disclosed_count}/{report.total_promotional} promotions disclosed "
            f"({report.disclosure_rate:.0f}%)."
        )

        # Top promoted
        if report.frequently_promoted:
            top = report.frequently_promoted[0]
            parts.append(f"Most promoted: {top[0]} ({top[1]} mentions).")

        return " ".join(parts)
