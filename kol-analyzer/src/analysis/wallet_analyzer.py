"""Wallet analyzer module for KOL credibility assessment using Nansen data."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


@dataclass
class WalletAnalysisResult:
    """Result of wallet analysis for a KOL."""
    # Identification
    username: str
    entity_matches: List[str] = field(default_factory=list)
    wallets_found: int = 0

    # Performance metrics
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    avg_roi: float = 0.0

    # Smart money status
    is_smart_money: bool = False
    smart_money_labels: List[str] = field(default_factory=list)

    # Risk assessment
    risk_score: float = 50.0  # 0-100, higher = more risky
    risk_flags: List[str] = field(default_factory=list)
    trust_signals: List[str] = field(default_factory=list)

    # Holdings analysis
    top_holdings: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_diversity: float = 0.0
    memecoin_exposure: float = 0.0

    # Trading patterns
    trade_frequency: str = "unknown"  # low, medium, high
    avg_trade_size: float = 0.0

    # Credibility impact
    credibility_modifier: float = 0.0  # -20 to +20
    analysis_summary: str = ""

    success: bool = True
    error: Optional[str] = None


class WalletAnalyzer:
    """
    Analyzes KOL wallets using Nansen data to assess credibility.

    Factors considered:
    - Smart money status and labels
    - Trading performance (PnL, ROI)
    - Risk indicators (rug involvement, exit liquidity)
    - Portfolio composition
    - Trading patterns
    """

    # Smart money labels that boost credibility
    POSITIVE_LABELS = {
        "Smart Trader": 10,
        "30D Smart Trader": 8,
        "90D Smart Trader": 7,
        "180D Smart Trader": 6,
        "Fund": 15,
        "Sector Specialist": 5,
        "AI Specialist": 5,
        "DeFi Specialist": 5,
        "Memecoin Specialist": 3,
    }

    # Labels that reduce credibility
    NEGATIVE_LABELS = {
        "Exit Liquidity": -15,
        "Rug Pull": -20,
        "Scammer": -25,
        "Wash Trader": -10,
        "Former Smart Trader": -5,
    }

    # Memecoin indicators
    MEMECOIN_PATTERNS = [
        r'\b(doge|shib|pepe|wojak|bonk|floki|inu|moon|elon|safe)\b',
        r'\b(meme|pump|rocket|lambo|wagmi)\b',
    ]

    def __init__(self):
        self.memecoin_regex = re.compile(
            '|'.join(self.MEMECOIN_PATTERNS),
            re.IGNORECASE
        )

    def analyze(
        self,
        username: str,
        nansen_data: Dict[str, Any]
    ) -> WalletAnalysisResult:
        """
        Analyze wallet data and produce credibility assessment.

        Args:
            username: KOL's username
            nansen_data: Data from NansenClient.analyze_kol_wallets()

        Returns:
            WalletAnalysisResult with analysis
        """
        result = WalletAnalysisResult(username=username)

        if not nansen_data.get("success", False):
            result.success = False
            result.error = nansen_data.get("error", "No wallet data available")
            return result

        # Extract basic data
        result.entity_matches = nansen_data.get("entity_matches", [])
        result.wallets_found = len(nansen_data.get("wallets_analyzed", []))
        result.total_realized_pnl = nansen_data.get("total_realized_pnl", 0)
        result.total_unrealized_pnl = nansen_data.get("total_unrealized_pnl", 0)
        result.is_smart_money = nansen_data.get("is_smart_money", False)
        result.smart_money_labels = nansen_data.get("smart_money_labels", [])
        result.top_holdings = nansen_data.get("top_holdings", [])[:10]

        # Analyze wallets
        wallets = nansen_data.get("wallets_analyzed", [])
        if wallets:
            result = self._analyze_wallets(result, wallets)

        # Calculate risk score
        result = self._calculate_risk_score(result, nansen_data)

        # Calculate credibility modifier
        result = self._calculate_credibility_modifier(result)

        # Generate summary
        result.analysis_summary = self._generate_summary(result)

        return result

    def _analyze_wallets(
        self,
        result: WalletAnalysisResult,
        wallets: List[Dict[str, Any]]
    ) -> WalletAnalysisResult:
        """Analyze individual wallet data."""

        total_trades = 0
        total_wins = 0
        total_roi = 0
        roi_count = 0
        trade_sizes = []

        all_labels = []

        for wallet in wallets:
            # Collect labels
            for label_info in wallet.get("labels", []):
                all_labels.append(label_info.get("label", ""))

            # Risk flags from wallet
            result.risk_flags.extend(wallet.get("risk_flags", []))

            # Smart money labels
            result.smart_money_labels.extend(wallet.get("smart_money_labels", []))

        # Calculate win rate from top holdings
        for holding in result.top_holdings:
            roi = holding.get("roi_percent", 0)
            if roi != 0:
                total_roi += roi
                roi_count += 1
                if roi > 0:
                    total_wins += 1
                total_trades += 1

            trade_size = holding.get("holding_usd", 0) + abs(holding.get("pnl_realized", 0))
            if trade_size > 0:
                trade_sizes.append(trade_size)

        # Calculate metrics
        if total_trades > 0:
            result.win_rate = (total_wins / total_trades) * 100

        if roi_count > 0:
            result.avg_roi = total_roi / roi_count

        if trade_sizes:
            result.avg_trade_size = sum(trade_sizes) / len(trade_sizes)

            # Determine trade frequency
            if len(trade_sizes) > 20:
                result.trade_frequency = "high"
            elif len(trade_sizes) > 5:
                result.trade_frequency = "medium"
            else:
                result.trade_frequency = "low"

        # Analyze portfolio diversity
        if result.top_holdings:
            unique_tokens = len(set(h.get("token", "") for h in result.top_holdings))
            result.portfolio_diversity = min(unique_tokens / 10, 1.0) * 100

        # Calculate memecoin exposure
        memecoin_count = 0
        for holding in result.top_holdings:
            token = holding.get("token", "")
            if self.memecoin_regex.search(token):
                memecoin_count += 1

        if result.top_holdings:
            result.memecoin_exposure = (memecoin_count / len(result.top_holdings)) * 100

        # Deduplicate
        result.risk_flags = list(set(result.risk_flags))
        result.smart_money_labels = list(set(result.smart_money_labels))

        return result

    def _calculate_risk_score(
        self,
        result: WalletAnalysisResult,
        nansen_data: Dict[str, Any]
    ) -> WalletAnalysisResult:
        """Calculate risk score based on wallet analysis."""

        risk_score = 50  # Start neutral

        # Adjust based on labels
        for label in result.smart_money_labels:
            if label in self.POSITIVE_LABELS:
                risk_score -= self.POSITIVE_LABELS[label]
                result.trust_signals.append(f"Has '{label}' status")

        for flag in result.risk_flags:
            for neg_label, penalty in self.NEGATIVE_LABELS.items():
                if neg_label.lower() in flag.lower():
                    risk_score -= penalty  # Penalty is negative, so this adds to risk
                    break

        # Adjust based on PnL
        total_pnl = result.total_realized_pnl + result.total_unrealized_pnl
        if total_pnl > 100000:
            risk_score -= 10
            result.trust_signals.append(f"Strong PnL: ${total_pnl:,.0f}")
        elif total_pnl > 10000:
            risk_score -= 5
            result.trust_signals.append(f"Positive PnL: ${total_pnl:,.0f}")
        elif total_pnl < -50000:
            risk_score += 10
            result.risk_flags.append(f"Large losses: ${total_pnl:,.0f}")
        elif total_pnl < -10000:
            risk_score += 5

        # Adjust based on win rate
        if result.win_rate > 70:
            risk_score -= 5
            result.trust_signals.append(f"High win rate: {result.win_rate:.0f}%")
        elif result.win_rate < 30 and result.win_rate > 0:
            risk_score += 5
            result.risk_flags.append(f"Low win rate: {result.win_rate:.0f}%")

        # Adjust based on memecoin exposure
        if result.memecoin_exposure > 80:
            risk_score += 10
            result.risk_flags.append("Heavy memecoin exposure")
        elif result.memecoin_exposure > 50:
            risk_score += 5

        # Smart money status
        if result.is_smart_money:
            risk_score -= 15
            if "Smart Trader" not in str(result.trust_signals):
                result.trust_signals.append("Verified Smart Money")

        # Clamp to 0-100
        result.risk_score = max(0, min(100, risk_score))

        return result

    def _calculate_credibility_modifier(
        self,
        result: WalletAnalysisResult
    ) -> WalletAnalysisResult:
        """
        Calculate how wallet analysis should modify KOL credibility score.

        Returns a modifier between -20 and +20.
        """
        modifier = 0

        # Smart money bonus
        if result.is_smart_money:
            modifier += 10

            # Additional bonus for specific labels
            for label in result.smart_money_labels:
                if label == "Fund":
                    modifier += 5
                elif "Smart Trader" in label:
                    modifier += 3

        # Performance bonus/penalty
        if result.avg_roi > 100:
            modifier += 5
        elif result.avg_roi > 50:
            modifier += 3
        elif result.avg_roi < -50:
            modifier -= 5

        # Risk penalty
        if result.risk_score > 70:
            modifier -= 10
        elif result.risk_score > 50:
            modifier -= 5
        elif result.risk_score < 30:
            modifier += 5

        # Flag-based adjustments
        for flag in result.risk_flags:
            if "scam" in flag.lower() or "rug" in flag.lower():
                modifier -= 10
                break
            elif "exit liquidity" in flag.lower():
                modifier -= 5

        # Clamp to -20 to +20
        result.credibility_modifier = max(-20, min(20, modifier))

        return result

    def _generate_summary(self, result: WalletAnalysisResult) -> str:
        """Generate human-readable summary of wallet analysis."""

        parts = []

        # Smart money status
        if result.is_smart_money:
            labels = ", ".join(result.smart_money_labels[:3]) if result.smart_money_labels else "verified"
            parts.append(f"Nansen Smart Money ({labels})")

        # Entity matches
        if result.entity_matches:
            parts.append(f"Known entity: {result.entity_matches[0]}")

        # PnL summary
        total_pnl = result.total_realized_pnl + result.total_unrealized_pnl
        if total_pnl > 0:
            parts.append(f"Profitable trader (${total_pnl:,.0f} total PnL)")
        elif total_pnl < -10000:
            parts.append(f"Net losses (${total_pnl:,.0f})")

        # Win rate
        if result.win_rate > 0:
            if result.win_rate > 60:
                parts.append(f"Strong {result.win_rate:.0f}% win rate")
            elif result.win_rate < 40:
                parts.append(f"Low {result.win_rate:.0f}% win rate")

        # Risk flags
        critical_flags = [f for f in result.risk_flags if any(
            word in f.lower() for word in ["scam", "rug", "exit liquidity"]
        )]
        if critical_flags:
            parts.append(f"WARNING: {critical_flags[0]}")

        # Memecoin exposure
        if result.memecoin_exposure > 70:
            parts.append("Primarily trades memecoins")

        if not parts:
            if result.wallets_found > 0:
                parts.append(f"Analyzed {result.wallets_found} wallet(s), no significant findings")
            else:
                parts.append("No wallet data available")

        return ". ".join(parts)

    def get_credibility_impact(
        self,
        result: WalletAnalysisResult
    ) -> Dict[str, Any]:
        """
        Get structured data for integrating into credibility score.

        Returns dict with:
        - modifier: float (-20 to +20) to add to credibility score
        - confidence: float (0-1) how confident we are in wallet data
        - factors: list of factor dicts
        """
        confidence = 0.5

        if result.wallets_found > 0:
            confidence = min(0.9, 0.5 + (result.wallets_found * 0.1))

        if result.is_smart_money:
            confidence = min(1.0, confidence + 0.2)

        factors = []

        if result.is_smart_money:
            factors.append({
                "name": "Smart Money Status",
                "impact": "positive",
                "value": ", ".join(result.smart_money_labels[:3]) or "verified"
            })

        if result.total_realized_pnl > 10000:
            factors.append({
                "name": "Trading Performance",
                "impact": "positive",
                "value": f"${result.total_realized_pnl:,.0f} realized PnL"
            })
        elif result.total_realized_pnl < -10000:
            factors.append({
                "name": "Trading Performance",
                "impact": "negative",
                "value": f"${result.total_realized_pnl:,.0f} realized PnL"
            })

        for flag in result.risk_flags[:3]:
            factors.append({
                "name": "Risk Flag",
                "impact": "negative",
                "value": flag
            })

        for signal in result.trust_signals[:3]:
            factors.append({
                "name": "Trust Signal",
                "impact": "positive",
                "value": signal
            })

        return {
            "modifier": result.credibility_modifier,
            "confidence": confidence,
            "factors": factors,
            "summary": result.analysis_summary
        }
