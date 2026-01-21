"""
Suspicious Activity Analyzer - Detect suspicious money movement patterns.

Detection patterns:
1. Mixer/Tumbler Usage - Interactions with Tornado Cash, etc.
2. Wash Trading - Circular transactions between related wallets
3. Exit Liquidity - Large coordinated sells
4. Fresh Wallet Funding - Receiving from newly created wallets
5. Coordinated Pump - Simultaneous buys with related wallets
6. Rapid Token Rotation - Quick buy-sell cycles across many tokens
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


class SuspiciousPatternType(Enum):
    """Types of suspicious patterns detected."""
    MIXER_USAGE = "mixer_usage"
    WASH_TRADING = "wash_trading"
    EXIT_LIQUIDITY = "exit_liquidity"
    FRESH_WALLET_FUNDING = "fresh_wallet_funding"
    COORDINATED_PUMP = "coordinated_pump"
    RAPID_TOKEN_ROTATION = "rapid_token_rotation"
    UNUSUAL_TIMING = "unusual_timing"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SuspiciousEvidence:
    """Evidence supporting a suspicious pattern detection."""
    transaction_hashes: List[str] = field(default_factory=list)
    amounts_usd: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    counterparties: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "transaction_hashes": self.transaction_hashes[:5],
            "amounts_usd": self.amounts_usd[:5],
            "timestamps": self.timestamps[:5],
            "counterparties": self.counterparties[:5],
            "description": self.description
        }


@dataclass
class SuspiciousPattern:
    """A detected suspicious pattern."""
    pattern_type: SuspiciousPatternType
    risk_level: RiskLevel
    confidence: float  # 0-100
    evidence: SuspiciousEvidence
    summary: str

    def to_dict(self) -> dict:
        return {
            "pattern_type": self.pattern_type.value,
            "risk_level": self.risk_level.value,
            "confidence": round(self.confidence, 1),
            "evidence": self.evidence.to_dict(),
            "summary": self.summary
        }


@dataclass
class SuspiciousActivityReport:
    """Complete suspicious activity analysis report."""
    address: str
    chain: str

    # Overall assessment
    overall_risk_level: RiskLevel = RiskLevel.LOW
    overall_risk_score: float = 0.0  # 0-100

    # Detected patterns
    patterns_detected: List[SuspiciousPattern] = field(default_factory=list)
    pattern_count: int = 0

    # Summary statistics
    total_transactions_analyzed: int = 0
    suspicious_transaction_count: int = 0
    suspicious_volume_usd: float = 0.0

    # Counterparty analysis
    mixer_interactions: int = 0
    fresh_wallet_funding_count: int = 0
    related_wallet_volume_usd: float = 0.0

    # Flags for display
    red_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Analysis metadata
    analysis_period_days: int = 90
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "chain": self.chain,
            "overall_risk_level": self.overall_risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 1),
            "patterns_detected": [p.to_dict() for p in self.patterns_detected],
            "pattern_count": self.pattern_count,
            "total_transactions_analyzed": self.total_transactions_analyzed,
            "suspicious_transaction_count": self.suspicious_transaction_count,
            "suspicious_volume_usd": round(self.suspicious_volume_usd, 2),
            "mixer_interactions": self.mixer_interactions,
            "fresh_wallet_funding_count": self.fresh_wallet_funding_count,
            "related_wallet_volume_usd": round(self.related_wallet_volume_usd, 2),
            "red_flags": self.red_flags,
            "warnings": self.warnings,
            "analysis_period_days": self.analysis_period_days,
            "success": self.success,
            "error": self.error
        }


class SuspiciousActivityAnalyzer:
    """
    Analyzes transaction patterns for suspicious activity.

    Non-blocking design - failures return empty results, never break flow.
    """

    # Known mixer/tumbler labels and keywords
    MIXER_LABELS = {
        "tornado cash", "tornado", "mixer", "tumbler",
        "blender", "chipmixer", "wasabi", "samourai",
        "railgun", "aztec", "cyclone"
    }

    # Suspicious counterparty categories
    SUSPICIOUS_CATEGORIES = {
        "mixer", "tumbler", "privacy", "gambling"
    }

    # Labels indicating fresh/new wallets
    FRESH_WALLET_LABELS = {
        "fresh wallet", "new wallet", "fresh", "newly created"
    }

    def __init__(self, nansen_client=None):
        """
        Initialize the analyzer.

        Args:
            nansen_client: Optional NansenClient instance for API calls
        """
        self.nansen_client = nansen_client

    async def analyze(
        self,
        address: str,
        chain: str = "ethereum",
        transactions: List[Any] = None,
        related_wallets: List[Any] = None,
        analysis_days: int = 90
    ) -> SuspiciousActivityReport:
        """
        Perform comprehensive suspicious activity analysis.

        Args:
            address: Wallet address to analyze
            chain: Blockchain
            transactions: Pre-fetched transactions (Transaction objects)
            related_wallets: Pre-fetched related wallets (RelatedWallet objects)
            analysis_days: Days of history to analyze

        Returns:
            SuspiciousActivityReport with findings
        """
        report = SuspiciousActivityReport(
            address=address,
            chain=chain,
            analysis_period_days=analysis_days
        )

        try:
            # Fetch transactions if not provided
            if transactions is None and self.nansen_client:
                tx_history = await self.nansen_client.get_address_transactions(
                    address, chain, days=analysis_days
                )
                transactions = tx_history.transactions if tx_history.success else []

            if not transactions:
                report.warnings.append("No transaction data available for analysis")
                return report

            report.total_transactions_analyzed = len(transactions)

            # Run detection heuristics
            patterns = []

            # 1. Mixer/Tumbler Detection
            mixer_pattern = self._detect_mixer_usage(transactions)
            if mixer_pattern:
                patterns.append(mixer_pattern)
                report.mixer_interactions = len(mixer_pattern.evidence.transaction_hashes)

            # 2. Fresh Wallet Funding Detection
            fresh_pattern = self._detect_fresh_wallet_funding(transactions)
            if fresh_pattern:
                patterns.append(fresh_pattern)
                report.fresh_wallet_funding_count = len(fresh_pattern.evidence.transaction_hashes)

            # 3. Wash Trading Detection
            wash_pattern = self._detect_wash_trading(transactions, related_wallets)
            if wash_pattern:
                patterns.append(wash_pattern)

            # 4. Exit Liquidity Detection
            exit_pattern = self._detect_exit_liquidity(transactions)
            if exit_pattern:
                patterns.append(exit_pattern)

            # 5. Coordinated Pump Detection
            pump_pattern = self._detect_coordinated_pump(transactions)
            if pump_pattern:
                patterns.append(pump_pattern)

            # 6. Rapid Token Rotation (pump & dump indicator)
            rotation_pattern = self._detect_rapid_token_rotation(transactions)
            if rotation_pattern:
                patterns.append(rotation_pattern)

            # 7. Unusual Large Transactions
            unusual_pattern = self._detect_unusual_large_transactions(transactions)
            if unusual_pattern:
                patterns.append(unusual_pattern)

            # Compile results
            report.patterns_detected = patterns
            report.pattern_count = len(patterns)

            # Calculate overall risk score and level
            report.overall_risk_score = self._calculate_risk_score(patterns)
            report.overall_risk_level = self._determine_risk_level(report.overall_risk_score)

            # Calculate suspicious volume
            report.suspicious_volume_usd = sum(
                sum(p.evidence.amounts_usd) for p in patterns
            )
            report.suspicious_transaction_count = sum(
                len(p.evidence.transaction_hashes) for p in patterns
            )

            # Generate red flags and warnings
            report.red_flags = self._generate_red_flags(patterns)
            report.warnings = self._generate_warnings(report)

        except Exception as e:
            report.success = False
            report.error = str(e)

        return report

    def _detect_mixer_usage(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect interactions with known mixers/tumblers."""
        mixer_txs = []

        for tx in transactions:
            counterparty = (getattr(tx, "counterparty_label", None) or "").lower()
            category = (getattr(tx, "counterparty_category", None) or "").lower()
            to_addr = (getattr(tx, "to_address", None) or "").lower()

            # Check for mixer labels or categories
            is_mixer = (
                any(m in counterparty for m in self.MIXER_LABELS) or
                category in self.SUSPICIOUS_CATEGORIES or
                "tornado" in to_addr or
                "mixer" in counterparty
            )

            if is_mixer:
                mixer_txs.append(tx)

        if not mixer_txs:
            return None

        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in mixer_txs)

        # Determine risk level based on volume and frequency
        if total_volume > 100000 or len(mixer_txs) > 5:
            risk_level = RiskLevel.CRITICAL
            confidence = 95.0
        elif total_volume > 10000 or len(mixer_txs) > 2:
            risk_level = RiskLevel.HIGH
            confidence = 85.0
        else:
            risk_level = RiskLevel.MEDIUM
            confidence = 70.0

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.MIXER_USAGE,
            risk_level=risk_level,
            confidence=confidence,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in mixer_txs],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in mixer_txs],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in mixer_txs],
                counterparties=[getattr(tx, "counterparty_label", None) or getattr(tx, "to_address", "") for tx in mixer_txs],
                description=f"Detected {len(mixer_txs)} interactions with mixing services totaling ${total_volume:,.2f}"
            ),
            summary=f"Mixer/tumbler usage ({len(mixer_txs)} txs, ${total_volume:,.0f})"
        )

    def _detect_fresh_wallet_funding(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect receiving funds from newly created wallets."""
        fresh_funding_txs = []

        for tx in transactions:
            counterparty = (getattr(tx, "counterparty_label", None) or "").lower()

            # Check if counterparty is labeled as fresh/new wallet
            if any(label in counterparty for label in self.FRESH_WALLET_LABELS):
                fresh_funding_txs.append(tx)

        if not fresh_funding_txs:
            return None

        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in fresh_funding_txs)

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.FRESH_WALLET_FUNDING,
            risk_level=RiskLevel.HIGH if total_volume > 50000 else RiskLevel.MEDIUM,
            confidence=60.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in fresh_funding_txs],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in fresh_funding_txs],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in fresh_funding_txs],
                counterparties=[getattr(tx, "from_address", "") for tx in fresh_funding_txs],
                description=f"Received ${total_volume:,.2f} from {len(fresh_funding_txs)} fresh wallets"
            ),
            summary=f"Fresh wallet funding ({len(fresh_funding_txs)} txs, ${total_volume:,.0f})"
        )

    def _detect_wash_trading(
        self,
        transactions: List[Any],
        related_wallets: List[Any] = None
    ) -> Optional[SuspiciousPattern]:
        """Detect circular transactions between related wallets."""
        if not related_wallets:
            return None

        related_addresses: Set[str] = {
            getattr(w, "address", "").lower() for w in related_wallets
        }

        if not related_addresses:
            return None

        circular_txs = []

        for tx in transactions:
            to_addr = (getattr(tx, "to_address", None) or "").lower()
            from_addr = (getattr(tx, "from_address", None) or "").lower()

            # Check for transactions to/from related wallets
            if to_addr in related_addresses or from_addr in related_addresses:
                circular_txs.append(tx)

        if len(circular_txs) < 3:
            return None

        # Look for same-token back-and-forth patterns
        token_volumes: Dict[str, float] = defaultdict(float)
        for tx in circular_txs:
            token = getattr(tx, "token_symbol", "UNKNOWN")
            token_volumes[token] += getattr(tx, "amount_usd", 0)

        total_volume = sum(token_volumes.values())

        if total_volume < 10000:
            return None

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.WASH_TRADING,
            risk_level=RiskLevel.HIGH if total_volume > 100000 else RiskLevel.MEDIUM,
            confidence=75.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in circular_txs[:10]],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in circular_txs[:10]],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in circular_txs[:10]],
                counterparties=list(related_addresses)[:5],
                description=f"Detected {len(circular_txs)} transactions with related wallets totaling ${total_volume:,.2f}"
            ),
            summary=f"Possible wash trading ({len(circular_txs)} txs with related wallets)"
        )

    def _detect_exit_liquidity(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect coordinated selling patterns (providing exit liquidity)."""
        # Group sells by token
        token_sells: Dict[str, List[Any]] = defaultdict(list)

        for tx in transactions:
            method = (getattr(tx, "method", "") or "").lower()
            amount_usd = getattr(tx, "amount_usd", 0)

            if method in ("sell", "swap_sell", "sold") and amount_usd > 1000:
                token = getattr(tx, "token_symbol", "UNKNOWN")
                token_sells[token].append(tx)

        exit_patterns = []

        for token, sells in token_sells.items():
            if len(sells) < 2:
                continue

            total_sell_volume = sum(getattr(s, "amount_usd", 0) for s in sells)

            # Large total sells of a single token could indicate exit liquidity
            if total_sell_volume > 50000:
                exit_patterns.extend(sells)

        if not exit_patterns:
            return None

        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in exit_patterns)

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.EXIT_LIQUIDITY,
            risk_level=RiskLevel.HIGH if total_volume > 100000 else RiskLevel.MEDIUM,
            confidence=65.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in exit_patterns[:10]],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in exit_patterns[:10]],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in exit_patterns[:10]],
                counterparties=[getattr(tx, "token_symbol", "") for tx in exit_patterns[:10]],
                description=f"Large coordinated sells totaling ${total_volume:,.2f}"
            ),
            summary=f"Exit liquidity pattern (${total_volume:,.0f} in concentrated sells)"
        )

    def _detect_coordinated_pump(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect concentrated buying activity."""
        buy_txs = []

        for tx in transactions:
            method = (getattr(tx, "method", "") or "").lower()
            if method in ("buy", "swap_buy", "bought", "swap"):
                buy_txs.append(tx)

        if len(buy_txs) < 5:
            return None

        # Group by 30-minute windows
        from collections import Counter
        windows: Counter = Counter()

        for tx in buy_txs:
            try:
                ts_str = getattr(tx, "block_timestamp", "")
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    window = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
                    windows[window] += 1
            except (ValueError, AttributeError):
                pass

        if not windows:
            return None

        max_window_count = max(windows.values())

        if max_window_count < 3:
            return None

        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in buy_txs)

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.COORDINATED_PUMP,
            risk_level=RiskLevel.MEDIUM,
            confidence=55.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in buy_txs[:10]],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in buy_txs[:10]],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in buy_txs[:10]],
                counterparties=[getattr(tx, "token_symbol", "") for tx in buy_txs[:10]],
                description=f"Concentrated buying ({max_window_count} buys in same 30-min window)"
            ),
            summary=f"Coordinated pump pattern ({max_window_count} clustered buys)"
        )

    def _detect_rapid_token_rotation(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect rapid buy-sell cycles across many tokens."""
        tokens_bought: Set[str] = set()
        tokens_sold: Set[str] = set()

        for tx in transactions:
            method = (getattr(tx, "method", "") or "").lower()
            token = getattr(tx, "token_symbol", "")

            if not token:
                continue

            if method in ("buy", "swap_buy", "bought"):
                tokens_bought.add(token)
            elif method in ("sell", "swap_sell", "sold", "swap"):
                tokens_sold.add(token)

        # Tokens that were both bought and sold
        rotated_tokens = tokens_bought & tokens_sold

        if len(rotated_tokens) < 10:
            return None

        rotation_txs = [
            tx for tx in transactions
            if getattr(tx, "token_symbol", "") in rotated_tokens
        ]
        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in rotation_txs)

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.RAPID_TOKEN_ROTATION,
            risk_level=RiskLevel.MEDIUM,
            confidence=60.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in rotation_txs[:10]],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in rotation_txs[:10]],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in rotation_txs[:10]],
                counterparties=list(rotated_tokens)[:10],
                description=f"Rotated through {len(rotated_tokens)} tokens rapidly"
            ),
            summary=f"High token rotation ({len(rotated_tokens)} tokens churned)"
        )

    def _detect_unusual_large_transactions(self, transactions: List[Any]) -> Optional[SuspiciousPattern]:
        """Detect unusually large transactions that may warrant attention."""
        large_txs = []

        for tx in transactions:
            amount_usd = getattr(tx, "amount_usd", 0)
            if amount_usd > 100000:  # Transactions over $100k
                large_txs.append(tx)

        if len(large_txs) < 2:
            return None

        total_volume = sum(getattr(tx, "amount_usd", 0) for tx in large_txs)

        return SuspiciousPattern(
            pattern_type=SuspiciousPatternType.UNUSUAL_TIMING,
            risk_level=RiskLevel.LOW,
            confidence=40.0,
            evidence=SuspiciousEvidence(
                transaction_hashes=[getattr(tx, "tx_hash", "") for tx in large_txs[:10]],
                amounts_usd=[getattr(tx, "amount_usd", 0) for tx in large_txs[:10]],
                timestamps=[getattr(tx, "block_timestamp", "") for tx in large_txs[:10]],
                counterparties=[getattr(tx, "token_symbol", "") for tx in large_txs[:10]],
                description=f"Multiple large transactions (>{len(large_txs)} over $100k)"
            ),
            summary=f"Large transactions ({len(large_txs)} txs over $100k)"
        )

    def _calculate_risk_score(self, patterns: List[SuspiciousPattern]) -> float:
        """Calculate overall risk score from detected patterns."""
        if not patterns:
            return 0.0

        # Weight patterns by risk level
        risk_weights = {
            RiskLevel.LOW: 10,
            RiskLevel.MEDIUM: 25,
            RiskLevel.HIGH: 50,
            RiskLevel.CRITICAL: 80
        }

        # Pattern type severity multipliers
        type_multipliers = {
            SuspiciousPatternType.MIXER_USAGE: 1.5,
            SuspiciousPatternType.WASH_TRADING: 1.3,
            SuspiciousPatternType.EXIT_LIQUIDITY: 1.2,
            SuspiciousPatternType.FRESH_WALLET_FUNDING: 1.0,
            SuspiciousPatternType.COORDINATED_PUMP: 1.1,
            SuspiciousPatternType.RAPID_TOKEN_ROTATION: 0.8,
            SuspiciousPatternType.UNUSUAL_TIMING: 0.6
        }

        total_score = 0.0

        for pattern in patterns:
            base_score = risk_weights.get(pattern.risk_level, 25)
            multiplier = type_multipliers.get(pattern.pattern_type, 1.0)
            confidence_factor = pattern.confidence / 100

            total_score += base_score * multiplier * confidence_factor

        # Normalize to 0-100
        return min(100.0, total_score)

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine overall risk level from score."""
        if score >= 70:
            return RiskLevel.CRITICAL
        elif score >= 50:
            return RiskLevel.HIGH
        elif score >= 25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_red_flags(self, patterns: List[SuspiciousPattern]) -> List[str]:
        """Generate human-readable red flags from patterns."""
        flags = []

        for pattern in patterns:
            if pattern.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                flags.append(f"[{pattern.risk_level.value.upper()}] {pattern.summary}")

        return flags[:10]  # Limit to top 10

    def _generate_warnings(self, report: SuspiciousActivityReport) -> List[str]:
        """Generate warning messages based on analysis."""
        warnings = list(report.warnings)  # Start with existing warnings

        if report.mixer_interactions > 0:
            warnings.append(f"Interacted with {report.mixer_interactions} mixing service(s)")

        if report.fresh_wallet_funding_count > 3:
            warnings.append("Multiple funding sources from fresh wallets")

        if report.suspicious_volume_usd > 100000:
            warnings.append(f"Over ${report.suspicious_volume_usd:,.0f} in suspicious transactions")

        if report.pattern_count > 3:
            warnings.append(f"Multiple suspicious patterns detected ({report.pattern_count})")

        return warnings[:10]  # Limit to 10
