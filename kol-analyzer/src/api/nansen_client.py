"""Nansen API client for token screening, wallet analysis, and smart money tracking."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


@dataclass
class WalletLabel:
    """Label associated with a wallet address."""
    label: str
    category: str  # behavioral, defi, social, smart_money, others
    definition: str
    fullname: Optional[str] = None
    sm_earned_date: Optional[str] = None


@dataclass
class WalletPnL:
    """PnL data for a token position."""
    token_symbol: str
    token_address: str
    chain: str
    pnl_usd_realized: float
    pnl_usd_unrealized: float
    roi_percent_realized: float
    roi_percent_unrealized: float
    bought_amount: float
    sold_amount: float
    holding_amount: float
    cost_basis_usd: float
    holding_usd: float
    num_buys: int
    num_sells: int


@dataclass
class RelatedWallet:
    """Related wallet connection."""
    address: str
    label: Optional[str]
    relation: str
    transaction_hash: str
    block_timestamp: str
    chain: str


@dataclass
class SmartMoneyTrade:
    """Smart money DEX trade."""
    wallet_address: str
    wallet_label: Optional[str]
    token_symbol: str
    token_address: str
    chain: str
    trade_type: str  # buy or sell
    amount_usd: float
    token_amount: float
    price_usd: float
    timestamp: str


@dataclass
class WalletProfile:
    """Complete wallet profile with labels, PnL, and related data."""
    address: str
    chain: str
    entity_name: Optional[str] = None
    labels: List[WalletLabel] = field(default_factory=list)
    pnl_data: List[WalletPnL] = field(default_factory=list)
    related_wallets: List[RelatedWallet] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    is_smart_money: bool = False
    smart_money_labels: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class EntitySearchResult:
    """Entity search result."""
    entity_names: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


@dataclass
class TokenData:
    """Token data from Nansen token screener."""
    symbol: str
    name: str
    chain: str
    contract_address: str
    price_usd: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    buy_volume: Optional[float] = None
    sell_volume: Optional[float] = None
    net_flow: Optional[float] = None
    smart_money_holders: int = 0
    token_age_days: int = 0
    price_change_24h: Optional[float] = None
    logo_url: Optional[str] = None


@dataclass
class NansenResponse:
    """Response from Nansen API."""
    tokens: List[TokenData] = field(default_factory=list)
    total_count: int = 0
    page: int = 1
    per_page: int = 100
    success: bool = True
    error: Optional[str] = None


@dataclass
class Transaction:
    """Single transaction from Nansen address/transactions endpoint."""
    tx_hash: str
    block_timestamp: str
    from_address: str
    to_address: str
    token_symbol: str
    token_address: str
    amount: float
    amount_usd: float
    method: str  # transfer, swap, buy, sell, etc.
    counterparty_label: Optional[str] = None
    counterparty_category: Optional[str] = None
    chain: str = "ethereum"


@dataclass
class TransactionHistory:
    """Transaction history response from Nansen."""
    address: str
    chain: str
    transactions: List[Transaction] = field(default_factory=list)
    total_count: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class PnLSummary:
    """Aggregate PnL summary from Nansen pnl-summary endpoint."""
    address: str
    chain: str
    total_realized_pnl: float = 0.0
    realized_pnl_percent: float = 0.0
    win_rate: float = 0.0
    traded_token_count: int = 0
    traded_times: int = 0
    top_5_tokens: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


class NansenClient:
    """
    Nansen API client for smart money analysis.

    Provides access to:
    - Token screener with smart money filters
    - Chain-specific token data
    - Volume and flow analysis
    """

    BASE_URL = "https://api.nansen.ai/api/v1"

    SUPPORTED_CHAINS = [
        "ethereum", "solana", "base", "arbitrum", "polygon",
        "optimism", "avalanche", "bsc", "fantom"
    ]

    SUPPORTED_TIMEFRAMES = ["1h", "4h", "12h", "24h", "7d", "30d"]

    ORDER_FIELDS = [
        "buy_volume", "sell_volume", "net_flow", "market_cap",
        "volume", "smart_money_holders", "price_change"
    ]

    def __init__(self, api_key: str = None):
        """
        Initialize the Nansen client.

        Args:
            api_key: Nansen API key. If not provided, reads from NANSEN_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("NANSEN_API_KEY", "")
        self._session = None

        if not self.api_key:
            print("  [Warning] No Nansen API key configured")

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import httpx
                self._session = httpx.AsyncClient(
                    headers={
                        "Content-Type": "application/json",
                        "apiKey": self.api_key
                    },
                    timeout=30.0
                )
            except ImportError:
                raise ImportError("httpx is required for NansenClient")
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.aclose()
            self._session = None

    async def get_token_screener(
        self,
        chains: List[str] = None,
        timeframe: str = "24h",
        only_smart_money: bool = True,
        token_age_min: int = 1,
        token_age_max: int = 365,
        order_by: str = "buy_volume",
        order_direction: str = "DESC",
        page: int = 1,
        per_page: int = 100
    ) -> NansenResponse:
        """
        Get tokens from the Nansen token screener.

        Args:
            chains: List of chains to filter (default: ethereum, solana, base)
            timeframe: Time period for metrics (1h, 4h, 12h, 24h, 7d, 30d)
            only_smart_money: Filter for smart money activity
            token_age_min: Minimum token age in days
            token_age_max: Maximum token age in days
            order_by: Field to order by (buy_volume, sell_volume, net_flow, etc.)
            order_direction: Sort direction (ASC or DESC)
            page: Page number for pagination
            per_page: Results per page (max 100)

        Returns:
            NansenResponse with token data
        """
        if not self.is_configured:
            return NansenResponse(
                success=False,
                error="Nansen API key not configured"
            )

        # Default chains
        if chains is None:
            chains = ["ethereum", "solana", "base"]

        # Validate inputs
        chains = [c.lower() for c in chains if c.lower() in self.SUPPORTED_CHAINS]
        if not chains:
            chains = ["ethereum", "solana", "base"]

        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            timeframe = "24h"

        if order_by not in self.ORDER_FIELDS:
            order_by = "buy_volume"

        per_page = min(max(1, per_page), 100)

        # Build request payload
        payload = {
            "chains": chains,
            "timeframe": timeframe,
            "filters": {
                "only_smart_money": only_smart_money,
                "token_age_days": {
                    "min": token_age_min,
                    "max": token_age_max
                }
            },
            "order_by": [
                {
                    "field": order_by,
                    "direction": order_direction.upper()
                }
            ],
            "pagination": {
                "page": page,
                "per_page": per_page
            }
        }

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/token-screener",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_token_screener_response(data, page, per_page)
            elif response.status_code == 401:
                return NansenResponse(
                    success=False,
                    error="Invalid API key"
                )
            elif response.status_code == 403:
                return NansenResponse(
                    success=False,
                    error="Access forbidden - check API key permissions"
                )
            elif response.status_code == 429:
                return NansenResponse(
                    success=False,
                    error="Rate limit exceeded"
                )
            else:
                return NansenResponse(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}"
                )

        except Exception as e:
            return NansenResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )

    def _parse_token_screener_response(
        self,
        data: Dict[str, Any],
        page: int,
        per_page: int
    ) -> NansenResponse:
        """Parse the token screener API response."""
        tokens = []

        # Handle different response formats
        token_list = data.get("data", data.get("tokens", data.get("results", [])))

        if isinstance(token_list, list):
            for item in token_list:
                try:
                    token = TokenData(
                        symbol=item.get("symbol", item.get("token_symbol", "UNKNOWN")),
                        name=item.get("name", item.get("token_name", "")),
                        chain=item.get("chain", item.get("blockchain", "unknown")),
                        contract_address=item.get("contract_address", item.get("address", "")),
                        price_usd=float(item.get("price_usd", item.get("price", 0)) or 0),
                        market_cap=self._safe_float(item.get("market_cap")),
                        volume_24h=self._safe_float(item.get("volume_24h", item.get("volume"))),
                        buy_volume=self._safe_float(item.get("buy_volume")),
                        sell_volume=self._safe_float(item.get("sell_volume")),
                        net_flow=self._safe_float(item.get("net_flow")),
                        smart_money_holders=int(item.get("smart_money_holders", 0) or 0),
                        token_age_days=int(item.get("token_age_days", item.get("age_days", 0)) or 0),
                        price_change_24h=self._safe_float(item.get("price_change_24h", item.get("price_change"))),
                        logo_url=item.get("logo_url", item.get("logo"))
                    )
                    tokens.append(token)
                except Exception as e:
                    print(f"  [Warning] Failed to parse token: {e}")
                    continue

        total_count = data.get("total_count", data.get("total", len(tokens)))

        return NansenResponse(
            tokens=tokens,
            total_count=total_count,
            page=page,
            per_page=per_page,
            success=True
        )

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def get_sync_token_screener(
        self,
        chains: List[str] = None,
        timeframe: str = "24h",
        only_smart_money: bool = True,
        token_age_min: int = 1,
        token_age_max: int = 365,
        order_by: str = "buy_volume",
        order_direction: str = "DESC",
        page: int = 1,
        per_page: int = 100
    ) -> NansenResponse:
        """
        Synchronous version of get_token_screener using requests.

        Use this when not in an async context.
        """
        if not self.is_configured:
            return NansenResponse(
                success=False,
                error="Nansen API key not configured"
            )

        try:
            import requests
        except ImportError:
            return NansenResponse(
                success=False,
                error="requests library required for sync operations"
            )

        # Default chains
        if chains is None:
            chains = ["ethereum", "solana", "base"]

        # Validate inputs
        chains = [c.lower() for c in chains if c.lower() in self.SUPPORTED_CHAINS]
        if not chains:
            chains = ["ethereum", "solana", "base"]

        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            timeframe = "24h"

        if order_by not in self.ORDER_FIELDS:
            order_by = "buy_volume"

        per_page = min(max(1, per_page), 100)

        # Build request payload
        payload = {
            "chains": chains,
            "timeframe": timeframe,
            "filters": {
                "only_smart_money": only_smart_money,
                "token_age_days": {
                    "min": token_age_min,
                    "max": token_age_max
                }
            },
            "order_by": [
                {
                    "field": order_by,
                    "direction": order_direction.upper()
                }
            ],
            "pagination": {
                "page": page,
                "per_page": per_page
            }
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/token-screener",
                headers={
                    "Content-Type": "application/json",
                    "apiKey": self.api_key
                },
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_token_screener_response(data, page, per_page)
            elif response.status_code == 401:
                return NansenResponse(
                    success=False,
                    error="Invalid API key"
                )
            elif response.status_code == 403:
                return NansenResponse(
                    success=False,
                    error="Access forbidden - check API key permissions"
                )
            elif response.status_code == 429:
                return NansenResponse(
                    success=False,
                    error="Rate limit exceeded"
                )
            else:
                return NansenResponse(
                    success=False,
                    error=f"API error: {response.status_code} - {response.text}"
                )

        except Exception as e:
            return NansenResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )

    # =========================================================================
    # WALLET & ENTITY ANALYSIS ENDPOINTS
    # =========================================================================

    async def search_entity(self, query: str) -> EntitySearchResult:
        """
        Search for entity names (KOLs, funds, etc.) in Nansen database.

        Args:
            query: Search query (min 2 characters, case-insensitive)

        Returns:
            EntitySearchResult with matching entity names
        """
        if not self.is_configured:
            return EntitySearchResult(success=False, error="API key not configured")

        if len(query) < 2:
            return EntitySearchResult(success=False, error="Query must be at least 2 characters")

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/search/entity-name",
                json={"search_query": query}
            )

            if response.status_code == 200:
                data = response.json()
                entities = [item.get("entity_name", "") for item in data.get("data", [])]
                return EntitySearchResult(entity_names=entities, success=True)
            else:
                return EntitySearchResult(
                    success=False,
                    error=f"API error: {response.status_code}"
                )

        except Exception as e:
            return EntitySearchResult(success=False, error=str(e))

    async def get_address_labels(
        self,
        address: str,
        chain: str = "ethereum"
    ) -> List[WalletLabel]:
        """
        Get all labels for a wallet address.

        Args:
            address: Wallet address
            chain: Blockchain (ethereum, solana, etc.)

        Returns:
            List of WalletLabel objects
        """
        if not self.is_configured:
            return []

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/profiler/address/labels",
                json={
                    "address": address,
                    "chain": chain.lower()
                }
            )

            if response.status_code == 200:
                data = response.json()
                labels = []
                label_list = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(label_list, list):
                    for item in label_list:
                        labels.append(WalletLabel(
                            label=item.get("label", ""),
                            category=item.get("category", "others"),
                            definition=item.get("definition", ""),
                            fullname=item.get("fullname"),
                            sm_earned_date=item.get("smEarnedDate")
                        ))
                return labels
            return []

        except Exception as e:
            print(f"  [Warning] Failed to get address labels: {e}")
            return []

    async def get_address_pnl(
        self,
        address: str,
        chain: str = "ethereum",
        limit: int = 50
    ) -> List[WalletPnL]:
        """
        Get PnL data for a wallet address.

        Args:
            address: Wallet address
            chain: Blockchain
            limit: Max results

        Returns:
            List of WalletPnL objects
        """
        if not self.is_configured:
            return []

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/profiler/address/pnl",
                json={
                    "address": address,
                    "chain": chain.lower(),
                    "pagination": {"page": 1, "per_page": min(limit, 100)},
                    "order_by": [{"field": "pnl_usd_realised", "direction": "DESC"}]
                }
            )

            if response.status_code == 200:
                data = response.json()
                pnl_list = []
                for item in data.get("data", []):
                    pnl_list.append(WalletPnL(
                        token_symbol=item.get("token_symbol", ""),
                        token_address=item.get("token_address", ""),
                        chain=chain,
                        pnl_usd_realized=float(item.get("pnl_usd_realised", 0) or 0),
                        pnl_usd_unrealized=float(item.get("pnl_usd_unrealised", 0) or 0),
                        roi_percent_realized=float(item.get("roi_percent_realised", 0) or 0),
                        roi_percent_unrealized=float(item.get("roi_percent_unrealised", 0) or 0),
                        bought_amount=float(item.get("bought_amount", 0) or 0),
                        sold_amount=float(item.get("sold_amount", 0) or 0),
                        holding_amount=float(item.get("holding_amount", 0) or 0),
                        cost_basis_usd=float(item.get("cost_basis_usd", 0) or 0),
                        holding_usd=float(item.get("holding_usd", 0) or 0),
                        num_buys=int(item.get("nof_buys", 0) or 0),
                        num_sells=int(item.get("nof_sells", 0) or 0)
                    ))
                return pnl_list
            return []

        except Exception as e:
            print(f"  [Warning] Failed to get address PnL: {e}")
            return []

    async def get_related_wallets(
        self,
        address: str,
        chain: str = "ethereum",
        limit: int = 20
    ) -> List[RelatedWallet]:
        """
        Get wallets related to an address.

        Args:
            address: Wallet address
            chain: Blockchain
            limit: Max results

        Returns:
            List of RelatedWallet objects
        """
        if not self.is_configured:
            return []

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/profiler/address/related-wallets",
                json={
                    "address": address,
                    "chain": chain.lower(),
                    "pagination": {"page": 1, "per_page": min(limit, 100)}
                }
            )

            if response.status_code == 200:
                data = response.json()
                wallets = []
                for item in data.get("data", []):
                    wallets.append(RelatedWallet(
                        address=item.get("address", ""),
                        label=item.get("address_label"),
                        relation=item.get("relation", ""),
                        transaction_hash=item.get("transaction_hash", ""),
                        block_timestamp=item.get("block_timestamp", ""),
                        chain=item.get("chain", chain)
                    ))
                return wallets
            return []

        except Exception as e:
            print(f"  [Warning] Failed to get related wallets: {e}")
            return []

    async def get_smart_money_trades(
        self,
        chains: List[str] = None,
        token_address: str = None,
        min_value_usd: float = 1000,
        limit: int = 50
    ) -> List[SmartMoneyTrade]:
        """
        Get recent smart money DEX trades.

        Args:
            chains: List of chains to filter
            token_address: Optional token to filter
            min_value_usd: Minimum trade value
            limit: Max results

        Returns:
            List of SmartMoneyTrade objects
        """
        if not self.is_configured:
            return []

        if chains is None:
            chains = ["ethereum", "solana", "base"]

        try:
            session = await self._get_session()
            payload = {
                "chains": [c.lower() for c in chains],
                "filters": {
                    "include_smart_money_labels": ["Fund", "Smart Trader", "30D Smart Trader"],
                    "value_usd": {"min": min_value_usd}
                },
                "pagination": {"page": 1, "per_page": min(limit, 100)},
                "order_by": [{"field": "value_usd", "direction": "DESC"}]
            }

            if token_address:
                payload["filters"]["token_address"] = token_address

            response = await session.post(
                f"{self.BASE_URL}/smart-money/dex-trades",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                trades = []
                for item in data.get("data", []):
                    trades.append(SmartMoneyTrade(
                        wallet_address=item.get("wallet_address", item.get("address", "")),
                        wallet_label=item.get("wallet_label", item.get("label")),
                        token_symbol=item.get("token_symbol", ""),
                        token_address=item.get("token_address", ""),
                        chain=item.get("chain", ""),
                        trade_type=item.get("trade_type", item.get("type", "")),
                        amount_usd=float(item.get("value_usd", item.get("amount_usd", 0)) or 0),
                        token_amount=float(item.get("token_amount", 0) or 0),
                        price_usd=float(item.get("price_usd", item.get("token_price", 0)) or 0),
                        timestamp=item.get("timestamp", item.get("block_timestamp", ""))
                    ))
                return trades
            return []

        except Exception as e:
            print(f"  [Warning] Failed to get smart money trades: {e}")
            return []

    async def get_wallet_profile(
        self,
        address: str,
        chain: str = "ethereum",
        entity_name: str = None
    ) -> WalletProfile:
        """
        Get comprehensive wallet profile including labels, PnL, and related wallets.

        Args:
            address: Wallet address
            chain: Blockchain
            entity_name: Optional entity name for context

        Returns:
            WalletProfile with all available data
        """
        if not self.is_configured:
            return WalletProfile(
                address=address,
                chain=chain,
                success=False,
                error="API key not configured"
            )

        try:
            import asyncio
            labels_task = self.get_address_labels(address, chain)
            pnl_task = self.get_address_pnl(address, chain)
            related_task = self.get_related_wallets(address, chain)

            labels, pnl_data, related = await asyncio.gather(
                labels_task, pnl_task, related_task
            )

            total_realized = sum(p.pnl_usd_realized for p in pnl_data)
            total_unrealized = sum(p.pnl_usd_unrealized for p in pnl_data)

            smart_money_categories = ["smart_money", "Fund", "Smart Trader"]
            sm_labels = [
                l.label for l in labels
                if l.category == "smart_money" or l.label in smart_money_categories
            ]
            is_smart_money = len(sm_labels) > 0

            risk_flags = []
            risky_labels = ["Exit Liquidity", "Rug Pull", "Scammer", "Wash Trader"]
            for label in labels:
                if label.label in risky_labels:
                    risk_flags.append(f"{label.label}: {label.definition}")

            big_losses = [p for p in pnl_data if p.pnl_usd_realized < -10000]
            if len(big_losses) > 3:
                risk_flags.append(f"Multiple large losses ({len(big_losses)} tokens with >$10K loss)")

            return WalletProfile(
                address=address,
                chain=chain,
                entity_name=entity_name,
                labels=labels,
                pnl_data=pnl_data,
                related_wallets=related,
                total_realized_pnl=total_realized,
                total_unrealized_pnl=total_unrealized,
                is_smart_money=is_smart_money,
                smart_money_labels=sm_labels,
                risk_flags=risk_flags,
                success=True
            )

        except Exception as e:
            return WalletProfile(
                address=address,
                chain=chain,
                success=False,
                error=str(e)
            )

    async def analyze_kol_wallets(
        self,
        kol_name: str,
        known_addresses: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze wallets associated with a KOL.

        Args:
            kol_name: KOL's name or username to search
            known_addresses: Optional list of {"address": "0x...", "chain": "ethereum"}

        Returns:
            Dict with wallet analysis findings
        """
        findings = {
            "kol_name": kol_name,
            "entity_matches": [],
            "wallets_analyzed": [],
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "is_smart_money": False,
            "smart_money_labels": [],
            "risk_flags": [],
            "top_holdings": [],
            "success": True,
            "error": None
        }

        if not self.is_configured:
            findings["success"] = False
            findings["error"] = "Nansen API not configured"
            return findings

        try:
            entity_result = await self.search_entity(kol_name)
            if entity_result.success and entity_result.entity_names:
                findings["entity_matches"] = entity_result.entity_names[:5]

            if known_addresses:
                for addr_info in known_addresses[:5]:
                    address = addr_info.get("address", "")
                    chain = addr_info.get("chain", "ethereum")

                    if not address:
                        continue

                    profile = await self.get_wallet_profile(address, chain, kol_name)

                    if profile.success:
                        findings["wallets_analyzed"].append({
                            "address": address,
                            "chain": chain,
                            "labels": [{"label": l.label, "category": l.category} for l in profile.labels],
                            "realized_pnl": profile.total_realized_pnl,
                            "unrealized_pnl": profile.total_unrealized_pnl,
                            "is_smart_money": profile.is_smart_money,
                            "smart_money_labels": profile.smart_money_labels,
                            "risk_flags": profile.risk_flags,
                            "related_count": len(profile.related_wallets)
                        })

                        findings["total_realized_pnl"] += profile.total_realized_pnl
                        findings["total_unrealized_pnl"] += profile.total_unrealized_pnl

                        if profile.is_smart_money:
                            findings["is_smart_money"] = True
                            findings["smart_money_labels"].extend(profile.smart_money_labels)

                        findings["risk_flags"].extend(profile.risk_flags)

                        for pnl in profile.pnl_data[:3]:
                            if pnl.holding_usd > 1000:
                                findings["top_holdings"].append({
                                    "token": pnl.token_symbol,
                                    "holding_usd": pnl.holding_usd,
                                    "pnl_realized": pnl.pnl_usd_realized,
                                    "roi_percent": pnl.roi_percent_realized
                                })

            findings["smart_money_labels"] = list(set(findings["smart_money_labels"]))
            findings["risk_flags"] = list(set(findings["risk_flags"]))

            return findings

        except Exception as e:
            findings["success"] = False
            findings["error"] = str(e)
            return findings

    # =========================================================================
    # TRANSACTION & SUSPICIOUS ACTIVITY ENDPOINTS
    # =========================================================================

    async def get_address_transactions(
        self,
        address: str,
        chain: str = "ethereum",
        days: int = 90,
        hide_spam: bool = True,
        min_volume_usd: float = 100,
        limit: int = 500
    ) -> TransactionHistory:
        """
        Get transaction history for a wallet address.

        Args:
            address: Wallet address
            chain: Blockchain
            days: Number of days to look back
            hide_spam: Filter out spam tokens
            min_volume_usd: Minimum transaction value in USD
            limit: Maximum transactions to retrieve

        Returns:
            TransactionHistory with transaction list
        """
        if not self.is_configured:
            return TransactionHistory(
                address=address,
                chain=chain,
                success=False,
                error="API key not configured"
            )

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/profiler/address/transactions",
                json={
                    "address": address,
                    "chain": chain.lower(),
                    "date": {
                        "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": end_date.strftime("%Y-%m-%dT23:59:59Z")
                    },
                    "hide_spam_token": hide_spam,
                    "filters": {
                        "volume_usd": {"min": min_volume_usd}
                    },
                    "pagination": {"page": 1, "per_page": min(limit, 500)}
                }
            )

            if response.status_code == 200:
                data = response.json()
                transactions = []
                for item in data.get("data", []):
                    # Parse transaction method/type
                    method = item.get("method", item.get("source_type", "unknown"))
                    if not method:
                        method = "transfer"

                    transactions.append(Transaction(
                        tx_hash=item.get("transaction_hash", item.get("tx_hash", "")),
                        block_timestamp=item.get("block_timestamp", ""),
                        from_address=item.get("from_address", item.get("from", "")),
                        to_address=item.get("to_address", item.get("to", "")),
                        token_symbol=item.get("token_symbol", item.get("symbol", "")),
                        token_address=item.get("token_address", item.get("contract_address", "")),
                        amount=float(item.get("amount", item.get("token_amount", 0)) or 0),
                        amount_usd=float(item.get("value_usd", item.get("amount_usd", 0)) or 0),
                        method=method,
                        counterparty_label=item.get("counterparty_label", item.get("counterparty_name")),
                        counterparty_category=item.get("counterparty_category"),
                        chain=chain
                    ))

                return TransactionHistory(
                    address=address,
                    chain=chain,
                    transactions=transactions,
                    total_count=data.get("total_count", len(transactions)),
                    success=True
                )

            return TransactionHistory(
                address=address,
                chain=chain,
                success=False,
                error=f"API error: {response.status_code}"
            )

        except Exception as e:
            return TransactionHistory(
                address=address,
                chain=chain,
                success=False,
                error=str(e)
            )

    async def get_pnl_summary(
        self,
        address: str,
        chain: str = "ethereum"
    ) -> PnLSummary:
        """
        Get aggregate PnL summary for a wallet.

        Args:
            address: Wallet address
            chain: Blockchain

        Returns:
            PnLSummary with aggregate statistics
        """
        if not self.is_configured:
            return PnLSummary(
                address=address,
                chain=chain,
                success=False,
                error="API key not configured"
            )

        try:
            session = await self._get_session()
            response = await session.post(
                f"{self.BASE_URL}/profiler/address/pnl-summary",
                json={
                    "address": address,
                    "chain": chain.lower()
                }
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("data", data)

                # Handle nested response structure
                if isinstance(result, dict):
                    return PnLSummary(
                        address=address,
                        chain=chain,
                        total_realized_pnl=float(result.get("realized_pnl_usd", result.get("total_realized_pnl", 0)) or 0),
                        realized_pnl_percent=float(result.get("realized_pnl_percent", 0) or 0),
                        win_rate=float(result.get("win_rate", 0) or 0),
                        traded_token_count=int(result.get("traded_token_count", 0) or 0),
                        traded_times=int(result.get("traded_times", 0) or 0),
                        top_5_tokens=result.get("top5_tokens", result.get("top_5_tokens", [])),
                        success=True
                    )

            return PnLSummary(
                address=address,
                chain=chain,
                success=False,
                error=f"API error: {response.status_code}"
            )

        except Exception as e:
            return PnLSummary(
                address=address,
                chain=chain,
                success=False,
                error=str(e)
            )
