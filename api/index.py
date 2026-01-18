"""Vercel Serverless Function Entry Point for KOL Analyzer API."""

import sys
import os
from pathlib import Path

# Set up paths BEFORE any other imports
root_dir = Path(__file__).parent.parent.absolute()
api_dir = Path(__file__).parent.absolute()  # The api/ folder
kol_analyzer_dir = root_dir / "kol-analyzer"
src_dir = kol_analyzer_dir / "src"

# Add paths to sys.path
for path in [str(api_dir), str(src_dir), str(kol_analyzer_dir), str(root_dir)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Set PYTHONPATH environment variable as well
os.environ["PYTHONPATH"] = f"{src_dir}:{kol_analyzer_dir}:{root_dir}"

from typing import List, Optional
from datetime import datetime
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Create FastAPI app first (before potentially failing imports)
app = FastAPI(
    title="KOL Credibility Analyzer API",
    description="Analyze crypto Twitter KOLs for credibility",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": traceback.format_exc() if os.environ.get("DEBUG") else None
        }
    )

# Try to import analysis components with detailed error handling
engine = None
TwitterCrawler = None
SupabaseDatabase = None
db = None

import_errors = []

try:
    from analysis.credibility_engine import CredibilityEngine
    engine = CredibilityEngine()
except Exception as e:
    import_errors.append(f"CredibilityEngine: {e}")

try:
    from scraper.twitter_crawler import TwitterCrawler as TC
    TwitterCrawler = TC
except Exception as e:
    import_errors.append(f"TwitterCrawler: {e}")

try:
    from storage.supabase_client import SupabaseDatabase as SDB
    SupabaseDatabase = SDB

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        db = SupabaseDatabase(url=supabase_url, key=supabase_key)
except Exception as e:
    import_errors.append(f"SupabaseDatabase: {e}")

# Import Nansen client (direct import to avoid triggering api/__init__.py)
NansenClient = None
nansen_client = None
try:
    import importlib.util
    nansen_module_path = src_dir / "api" / "nansen_client.py"
    if nansen_module_path.exists():
        spec = importlib.util.spec_from_file_location("nansen_client", nansen_module_path)
        nansen_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nansen_module)
        NansenClient = nansen_module.NansenClient
        nansen_api_key = os.environ.get("NANSEN_API_KEY")
        if nansen_api_key:
            nansen_client = NansenClient(api_key=nansen_api_key)
    else:
        import_errors.append(f"NansenClient: nansen_client.py not found at {nansen_module_path}")
except Exception as e:
    import_errors.append(f"NansenClient: {e}")

# Import Wallet Analyzer
WalletAnalyzer = None
try:
    wallet_analyzer_path = src_dir / "analysis" / "wallet_analyzer.py"
    if wallet_analyzer_path.exists():
        spec = importlib.util.spec_from_file_location("wallet_analyzer", wallet_analyzer_path)
        wallet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wallet_module)
        WalletAnalyzer = wallet_module.WalletAnalyzer
except Exception as e:
    import_errors.append(f"WalletAnalyzer: {e}")

# Import and setup admin router
try:
    from admin import router as admin_router, set_db as admin_set_db, set_admin_password
    app.include_router(admin_router)
    if db:
        admin_set_db(db)
    # Set admin password from environment variable
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if admin_password:
        set_admin_password(admin_password)
except Exception as e:
    import_errors.append(f"AdminRouter: {e}")


# Pydantic models
class WalletAddress(BaseModel):
    address: str = Field(..., description="Wallet address (0x... or Solana address)")
    chain: str = Field(default="ethereum", description="Blockchain (ethereum, solana, base, etc.)")


class AnalyzeRequest(BaseModel):
    username: str = Field(..., description="Twitter username to analyze")
    max_tweets: int = Field(1000, description="Maximum tweets to analyze", ge=10, le=2000)
    force_refresh: bool = Field(False, description="Force re-analysis even if cached")
    wallet_addresses: Optional[List[WalletAddress]] = Field(
        default=None,
        description="Optional wallet addresses to analyze with Nansen"
    )


class AnalysisResponse(BaseModel):
    username: str
    display_name: Optional[str] = None
    profile_image_url: Optional[str] = None
    follower_count: Optional[int] = None
    overall_score: float
    grade: str
    confidence: float
    assessment: str
    engagement_score: float
    consistency_score: float
    dissonance_score: float
    baiting_score: float
    red_flags: List[str]
    green_flags: List[str]
    summary: str
    tweets_analyzed: int
    tweet_count: int = 0  # Total tweets in DB for this user
    analyzed_at: str
    demo_mode: bool = False
    # Asshole Meter
    asshole_score: float = 50.0
    toxicity_level: str = "mid"
    toxicity_emoji: str = "😐"
    # Contradiction/BS Detection
    bs_score: float = 0.0
    contradiction_count: int = 0
    contradictions: List[dict] = []
    # Wallet/On-chain Analysis (Nansen)
    wallet_analysis: Optional[dict] = None


class StatsResponse(BaseModel):
    kols_analyzed: int
    total_analyses: int
    total_tweets: int
    average_score: float


# Nansen API Models
class NansenTokenScreenerRequest(BaseModel):
    chains: List[str] = Field(
        default=["ethereum", "solana", "base"],
        description="Chains to filter (ethereum, solana, base, arbitrum, polygon, optimism, avalanche, bsc, fantom)"
    )
    timeframe: str = Field(
        default="24h",
        description="Time period for metrics (1h, 4h, 12h, 24h, 7d, 30d)"
    )
    only_smart_money: bool = Field(
        default=True,
        description="Filter for smart money activity only"
    )
    token_age_min: int = Field(default=1, ge=0, description="Minimum token age in days")
    token_age_max: int = Field(default=365, ge=1, description="Maximum token age in days")
    order_by: str = Field(
        default="buy_volume",
        description="Field to order by (buy_volume, sell_volume, net_flow, market_cap, volume, smart_money_holders, price_change)"
    )
    order_direction: str = Field(default="DESC", description="Sort direction (ASC or DESC)")
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=100, ge=1, le=100, description="Results per page (max 100)")


class NansenTokenResponse(BaseModel):
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


class NansenScreenerResponse(BaseModel):
    success: bool
    tokens: List[NansenTokenResponse] = []
    total_count: int = 0
    page: int = 1
    per_page: int = 100
    error: Optional[str] = None


# Wallet Analysis Models
class WalletAnalysisRequest(BaseModel):
    username: str = Field(..., description="KOL username for context")
    wallet_addresses: List[WalletAddress] = Field(..., description="Wallet addresses to analyze")


class WalletAnalysisResponse(BaseModel):
    success: bool
    username: str
    entity_matches: List[str] = []
    wallets_analyzed: int = 0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    is_smart_money: bool = False
    smart_money_labels: List[str] = []
    risk_score: float = 50.0
    risk_flags: List[str] = []
    trust_signals: List[str] = []
    top_holdings: List[dict] = []
    credibility_modifier: float = 0.0
    analysis_summary: str = ""
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the web UI."""
    static_file = kol_analyzer_dir / "static" / "index.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text(), status_code=200)
    return HTMLResponse(
        content=f"""
        <html>
        <head><title>KOL Analyzer</title></head>
        <body>
            <h1>KOL Credibility Analyzer API</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for documentation.</p>
            <p>Health check: <a href="/health">/health</a></p>
        </body>
        </html>
        """,
        status_code=200
    )


@app.get("/admin-panel", response_class=HTMLResponse, include_in_schema=False)
async def admin_page():
    """Serve the admin panel UI."""
    static_file = kol_analyzer_dir / "static" / "admin.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text(), status_code=200)
    return HTMLResponse(
        content="<html><body><h1>Admin panel not found</h1></body></html>",
        status_code=404
    )


@app.get("/docs.html", response_class=HTMLResponse, include_in_schema=False)
async def docs_page():
    """Serve the documentation page."""
    static_file = kol_analyzer_dir / "static" / "docs.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text(), status_code=200)
    return HTMLResponse(
        content="<html><body><h1>Documentation not found</h1></body></html>",
        status_code=404
    )


@app.get("/health")
async def health_check():
    """Health check endpoint with diagnostic info."""
    return {
        "status": "healthy" if engine else "degraded",
        "service": "KOL Credibility Analyzer",
        "version": "1.0.0",
        "components": {
            "engine": engine is not None,
            "crawler": TwitterCrawler is not None,
            "database": db is not None,
            "nansen": nansen_client is not None and nansen_client.is_configured
        },
        "import_errors": import_errors if import_errors else None,
        "paths": {
            "root": str(root_dir),
            "src": str(src_dir),
            "exists": {
                "src": src_dir.exists(),
                "analysis": (src_dir / "analysis").exists(),
                "scraper": (src_dir / "scraper").exists()
            }
        }
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    if db:
        try:
            stats = db.get_stats()
            return StatsResponse(**stats)
        except Exception as e:
            return StatsResponse(kols_analyzed=0, total_analyses=0, total_tweets=0, average_score=0)
    return StatsResponse(kols_analyzed=0, total_analyses=0, total_tweets=0, average_score=0)


# =============================================================================
# NANSEN API ENDPOINTS
# =============================================================================

@app.post("/nansen/token-screener", response_model=NansenScreenerResponse)
async def nansen_token_screener(request: NansenTokenScreenerRequest):
    """
    Get tokens from Nansen token screener with smart money filters.

    This endpoint provides access to Nansen's token screening data, allowing you to:
    - Filter tokens by blockchain (Ethereum, Solana, Base, etc.)
    - Filter by smart money activity
    - Sort by various metrics (buy volume, sell volume, net flow, etc.)
    - Paginate through results
    """
    if not nansen_client or not nansen_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Nansen API not configured. Set NANSEN_API_KEY environment variable."
        )

    try:
        result = await nansen_client.get_token_screener(
            chains=request.chains,
            timeframe=request.timeframe,
            only_smart_money=request.only_smart_money,
            token_age_min=request.token_age_min,
            token_age_max=request.token_age_max,
            order_by=request.order_by,
            order_direction=request.order_direction,
            page=request.page,
            per_page=request.per_page
        )

        if not result.success:
            return NansenScreenerResponse(
                success=False,
                error=result.error
            )

        return NansenScreenerResponse(
            success=True,
            tokens=[
                NansenTokenResponse(
                    symbol=t.symbol,
                    name=t.name,
                    chain=t.chain,
                    contract_address=t.contract_address,
                    price_usd=t.price_usd,
                    market_cap=t.market_cap,
                    volume_24h=t.volume_24h,
                    buy_volume=t.buy_volume,
                    sell_volume=t.sell_volume,
                    net_flow=t.net_flow,
                    smart_money_holders=t.smart_money_holders,
                    token_age_days=t.token_age_days,
                    price_change_24h=t.price_change_24h,
                    logo_url=t.logo_url
                )
                for t in result.tokens
            ],
            total_count=result.total_count,
            page=result.page,
            per_page=result.per_page
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Nansen API request failed: {str(e)}"
        )


@app.get("/nansen/token-screener", response_model=NansenScreenerResponse)
async def nansen_token_screener_get(
    chains: str = "ethereum,solana,base",
    timeframe: str = "24h",
    only_smart_money: bool = True,
    token_age_min: int = 1,
    token_age_max: int = 365,
    order_by: str = "buy_volume",
    order_direction: str = "DESC",
    page: int = 1,
    per_page: int = 100
):
    """
    Get tokens from Nansen token screener (GET method for convenience).

    Pass chains as comma-separated string (e.g., "ethereum,solana,base")
    """
    chain_list = [c.strip() for c in chains.split(",") if c.strip()]

    request = NansenTokenScreenerRequest(
        chains=chain_list,
        timeframe=timeframe,
        only_smart_money=only_smart_money,
        token_age_min=token_age_min,
        token_age_max=token_age_max,
        order_by=order_by,
        order_direction=order_direction,
        page=page,
        per_page=per_page
    )
    return await nansen_token_screener(request)


# =============================================================================
# WALLET ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/nansen/wallet-analysis", response_model=WalletAnalysisResponse)
async def analyze_wallet(request: WalletAnalysisRequest):
    """
    Analyze wallet addresses associated with a KOL using Nansen data.

    This endpoint:
    - Searches for the KOL entity in Nansen's database
    - Retrieves labels, PnL, and trading performance for provided wallets
    - Identifies smart money status and risk flags
    - Returns a credibility modifier to adjust KOL scores
    """
    if not nansen_client or not nansen_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Nansen API not configured. Set NANSEN_API_KEY environment variable."
        )

    try:
        # Convert wallet addresses to dict format
        addresses = [
            {"address": w.address, "chain": w.chain}
            for w in request.wallet_addresses
        ]

        # Get wallet data from Nansen
        nansen_data = await nansen_client.analyze_kol_wallets(
            kol_name=request.username,
            known_addresses=addresses
        )

        # Analyze with WalletAnalyzer if available
        if WalletAnalyzer:
            analyzer = WalletAnalyzer()
            result = analyzer.analyze(request.username, nansen_data)

            return WalletAnalysisResponse(
                success=result.success,
                username=request.username,
                entity_matches=result.entity_matches,
                wallets_analyzed=result.wallets_found,
                total_realized_pnl=result.total_realized_pnl,
                total_unrealized_pnl=result.total_unrealized_pnl,
                win_rate=result.win_rate,
                is_smart_money=result.is_smart_money,
                smart_money_labels=result.smart_money_labels,
                risk_score=result.risk_score,
                risk_flags=result.risk_flags,
                trust_signals=result.trust_signals,
                top_holdings=result.top_holdings,
                credibility_modifier=result.credibility_modifier,
                analysis_summary=result.analysis_summary,
                error=result.error
            )
        else:
            # Return raw Nansen data if WalletAnalyzer not available
            return WalletAnalysisResponse(
                success=nansen_data.get("success", False),
                username=request.username,
                entity_matches=nansen_data.get("entity_matches", []),
                wallets_analyzed=len(nansen_data.get("wallets_analyzed", [])),
                total_realized_pnl=nansen_data.get("total_realized_pnl", 0),
                total_unrealized_pnl=nansen_data.get("total_unrealized_pnl", 0),
                is_smart_money=nansen_data.get("is_smart_money", False),
                smart_money_labels=nansen_data.get("smart_money_labels", []),
                risk_flags=nansen_data.get("risk_flags", []),
                top_holdings=nansen_data.get("top_holdings", []),
                error=nansen_data.get("error")
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Wallet analysis failed: {str(e)}"
        )


@app.get("/nansen/entity-search")
async def search_entity(query: str):
    """
    Search for entity names in Nansen's database.

    Use this to find if a KOL or fund is tracked by Nansen.
    """
    if not nansen_client or not nansen_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Nansen API not configured"
        )

    try:
        result = await nansen_client.search_entity(query)
        return {
            "success": result.success,
            "query": query,
            "entity_matches": result.entity_names,
            "error": result.error
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nansen/smart-money-trades")
async def get_smart_money_trades(
    chains: str = "ethereum,solana,base",
    min_value_usd: float = 1000,
    limit: int = 50
):
    """
    Get recent smart money DEX trades.

    Useful for seeing what smart traders are buying/selling.
    """
    if not nansen_client or not nansen_client.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Nansen API not configured"
        )

    try:
        chain_list = [c.strip() for c in chains.split(",") if c.strip()]
        trades = await nansen_client.get_smart_money_trades(
            chains=chain_list,
            min_value_usd=min_value_usd,
            limit=limit
        )

        return {
            "success": True,
            "trades": [
                {
                    "wallet_address": t.wallet_address,
                    "wallet_label": t.wallet_label,
                    "token_symbol": t.token_symbol,
                    "chain": t.chain,
                    "trade_type": t.trade_type,
                    "amount_usd": t.amount_usd,
                    "timestamp": t.timestamp
                }
                for t in trades
            ],
            "count": len(trades)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze")
async def analyze_kol_get(username: str, max_tweets: int = 200, force_refresh: bool = False):
    """Analyze a KOL's credibility (GET method for convenience)."""
    request = AnalyzeRequest(username=username, max_tweets=max_tweets, force_refresh=force_refresh)
    return await analyze_kol_post(request)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_kol_post(request: AnalyzeRequest):
    """Analyze a KOL's credibility."""
    if not engine:
        raise HTTPException(
            status_code=503,
            detail=f"Analysis engine not available. Import errors: {import_errors}"
        )

    username = request.username.lstrip('@').lower()

    # Check for cached analysis first (unless force_refresh)
    if db and not request.force_refresh:
        try:
            cached = db.get_latest_analysis(username)
            if cached:
                # Check if we have more cached tweets than were analyzed
                kol = db.get_kol(username)
                if kol:
                    cached_tweet_count = 0
                    try:
                        count_result = db.client.table("tweets").select("id", count="exact").eq("kol_id", kol["id"]).execute()
                        cached_tweet_count = count_result.count or 0
                    except:
                        pass

                    # If we have significantly more tweets (>20% more), skip cached analysis
                    analyzed_count = cached.get('tweets_analyzed', 0)
                    if cached_tweet_count > analyzed_count * 1.2 and cached_tweet_count >= 20:
                        print(f"Skipping cached analysis: have {cached_tweet_count} tweets but only analyzed {analyzed_count}")
                    else:
                        # Return cached analysis
                        return AnalysisResponse(
                            username=username,
                            display_name=kol.get('display_name') if kol else None,
                            profile_image_url=kol.get('profile_image_url') if kol else None,
                            follower_count=kol.get('follower_count') if kol else None,
                            overall_score=cached['overall_score'],
                            grade=cached['grade'],
                            confidence=cached['confidence'],
                            assessment=cached['assessment'],
                            engagement_score=cached['engagement_score'],
                            consistency_score=cached['consistency_score'],
                            dissonance_score=cached['dissonance_score'],
                            baiting_score=cached['baiting_score'],
                            red_flags=cached['red_flags'],
                            green_flags=cached['green_flags'],
                            summary=cached['summary'],
                            tweets_analyzed=cached['tweets_analyzed'],
                            analyzed_at=cached['created_at'],
                            demo_mode=False,
                            asshole_score=cached.get('asshole_score', 50.0),
                            toxicity_level=cached.get('toxicity_level', 'mid'),
                            toxicity_emoji=cached.get('toxicity_emoji', '😐'),
                            bs_score=cached.get('bs_score', 0.0),
                            contradiction_count=cached.get('contradiction_count', 0),
                            contradictions=cached.get('contradictions', [])
                        )
        except Exception as e:
            print(f"Cache lookup failed: {e}")

    # Check if we have cached tweets in Supabase
    cached_tweets = []
    kol = None
    if db:
        try:
            kol = db.get_kol(username)
            if kol:
                cached_tweets = db.get_tweets(kol["id"], limit=request.max_tweets)
                print(f"Found {len(cached_tweets)} cached tweets for {username}")
        except Exception as e:
            print(f"Failed to get cached tweets: {e}")

    # If we have cached tweets, use them for analysis
    if cached_tweets and len(cached_tweets) >= 10:
        tweets_data = [
            {
                'id': t.get('tweet_id', t.get('id', '')),
                'text': t.get('text', ''),
                'timestamp': t.get('timestamp', ''),
                'likes': t.get('likes', 0),
                'retweets': t.get('retweets', 0),
                'replies': t.get('replies', 0),
                'has_media': t.get('has_media', False),
                'has_video': t.get('has_video', False),
                'is_quote_tweet': t.get('is_quote_tweet', False)
            }
            for t in cached_tweets
        ]

        # Get mentions if available
        mentions = []
        try:
            if db and kol:
                mentions_result = db.client.table("mentions").select("*").eq("kol_id", kol["id"]).limit(50).execute()
                mentions = mentions_result.data if mentions_result.data else []
        except:
            pass

        result = engine.analyze(
            tweets_data,
            kol.get('follower_count', 0) if kol else 0,
            username,
            mentions=mentions
        )

        # Save the new analysis
        if db and kol:
            try:
                db.save_analysis(kol["id"], result.to_dict(), len(tweets_data))
            except Exception as e:
                print(f"Failed to save analysis: {e}")

        return AnalysisResponse(
            username=username,
            display_name=kol.get('display_name') if kol else None,
            profile_image_url=kol.get('profile_image_url') if kol else None,
            follower_count=kol.get('follower_count') if kol else None,
            overall_score=result.overall_score,
            grade=result.grade,
            confidence=result.confidence,
            assessment=result.assessment,
            engagement_score=result.engagement_score,
            consistency_score=result.consistency_score,
            dissonance_score=result.dissonance_score,
            baiting_score=result.baiting_score,
            red_flags=result.red_flags,
            green_flags=result.green_flags,
            summary=result.summary,
            tweets_analyzed=len(tweets_data),
            analyzed_at=datetime.now().isoformat(),
            demo_mode=False,
            asshole_score=result.asshole_score,
            toxicity_level=result.toxicity_level,
            toxicity_emoji=result.toxicity_emoji,
            bs_score=result.bs_score,
            contradiction_count=result.contradiction_count,
            contradictions=result.contradictions
        )

    # No cached tweets - fetch from Twitter API
    if not TwitterCrawler:
        raise HTTPException(
            status_code=503,
            detail=f"Twitter crawler not available and no cached tweets found. Import errors: {import_errors}"
        )

    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    crawler = TwitterCrawler(rapidapi_key=rapidapi_key)

    try:
        await crawler.initialize()

        profile = await crawler.get_user_profile(username)
        if not profile:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        tweets = await crawler.get_user_tweets(
            username,
            max_tweets=min(request.max_tweets, 50)
        )

        # Fetch mentions (what others say about this KOL)
        mentions = await crawler.search_mentions(username, max_results=20)

        tweets_data = [
            {
                'id': t.id,
                'text': t.text,
                'timestamp': t.timestamp,
                'likes': t.likes,
                'retweets': t.retweets,
                'replies': t.replies,
                'has_media': t.has_media,
                'has_video': t.has_video,
                'is_quote_tweet': t.is_quote_tweet
            }
            for t in tweets
        ]

        result = engine.analyze(
            tweets_data,
            profile.follower_count,
            username,
            mentions=mentions  # Pass mentions for reputation analysis
        )

        # Try to cache
        if db:
            try:
                kol_id = db.upsert_kol(profile)
                db.save_tweets(kol_id, tweets)
                db.save_mentions(kol_id, mentions)  # Cache mentions
                db.save_analysis(kol_id, result.to_dict(), len(tweets))
            except Exception as e:
                print(f"Failed to cache: {e}")

        return AnalysisResponse(
            username=username,
            display_name=profile.display_name,
            profile_image_url=profile.profile_image_url,
            follower_count=profile.follower_count,
            overall_score=result.overall_score,
            grade=result.grade,
            confidence=result.confidence,
            assessment=result.assessment,
            engagement_score=result.engagement_score,
            consistency_score=result.consistency_score,
            dissonance_score=result.dissonance_score,
            baiting_score=result.baiting_score,
            red_flags=result.red_flags,
            green_flags=result.green_flags,
            summary=result.summary,
            tweets_analyzed=len(tweets),
            analyzed_at=datetime.now().isoformat(),
            demo_mode=crawler.demo_mode,
            asshole_score=result.asshole_score,
            toxicity_level=result.toxicity_level,
            toxicity_emoji=result.toxicity_emoji,
            bs_score=result.bs_score,
            contradiction_count=result.contradiction_count,
            contradictions=result.contradictions
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
        )
    finally:
        await crawler.close()


@app.get("/kol/{username}")
async def get_kol(username: str):
    """Get cached analysis for a KOL."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    username = username.lstrip('@').lower()
    kol = db.get_kol(username)

    if not kol:
        raise HTTPException(status_code=404, detail=f"No analysis found for @{username}")

    analysis = db.get_latest_analysis(username)

    # Check if we have significantly more tweets than were analyzed
    # If so, return 404 to trigger re-analysis via /analyze endpoint
    if analysis:
        try:
            count_result = db.client.table("tweets").select("id", count="exact").eq("kol_id", kol["id"]).execute()
            cached_tweet_count = count_result.count or 0
            analyzed_count = analysis.get('tweets_analyzed', 0)

            # If we have 20% more tweets (and at least 20 total), skip cached and trigger re-analysis
            if cached_tweet_count > analyzed_count * 1.2 and cached_tweet_count >= 20:
                print(f"[/kol] Skipping cache for {username}: have {cached_tweet_count} tweets but only analyzed {analyzed_count}")
                raise HTTPException(status_code=404, detail=f"Re-analysis needed for @{username}")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[/kol] Error checking tweet count: {e}")

    # Get actual tweet count from database
    tweet_count = 0
    try:
        count_result = db.client.table("tweets").select("id", count="exact").eq("kol_id", kol["id"]).execute()
        tweet_count = count_result.count or 0
    except Exception as e:
        print(f"[/kol] Error getting tweet count: {e}")

    return {"kol": kol, "analysis": analysis, "tweet_count": tweet_count}


@app.get("/kols")
async def list_kols(limit: int = 50):
    """List all analyzed KOLs."""
    if not db:
        return []
    try:
        return db.list_kols(limit=limit)
    except Exception:
        return []
