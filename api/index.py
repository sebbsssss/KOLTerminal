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
class AnalyzeRequest(BaseModel):
    username: str = Field(..., description="Twitter username to analyze")
    max_tweets: int = Field(1000, description="Maximum tweets to analyze", ge=10, le=2000)
    force_refresh: bool = Field(False, description="Force re-analysis even if cached")


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
    analyzed_at: str
    demo_mode: bool = False


class StatsResponse(BaseModel):
    kols_analyzed: int
    total_analyses: int
    total_tweets: int
    average_score: float


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
            "database": db is not None
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
                # Get KOL profile for display info
                kol = db.get_kol(username)
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
                    demo_mode=False
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
            demo_mode=False
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
            demo_mode=crawler.demo_mode
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
    return {"kol": kol, "analysis": analysis}


@app.get("/kols")
async def list_kols(limit: int = 50):
    """List all analyzed KOLs."""
    if not db:
        return []
    try:
        return db.list_kols(limit=limit)
    except Exception:
        return []
