"""Vercel Serverless Function Entry Point for KOL Analyzer API."""

import sys
import os
from pathlib import Path

# Add the kol-analyzer/src directory to Python path
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "kol-analyzer" / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(root_dir / "kol-analyzer"))

# Set environment for Vercel
os.environ.setdefault("VERCEL", "1")

from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Import analysis components
from analysis.credibility_engine import CredibilityEngine
from scraper.twitter_crawler import TwitterCrawler

# Try to import Supabase storage, fall back to in-memory
try:
    from storage.supabase_client import SupabaseDatabase
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Pydantic models
class AnalyzeRequest(BaseModel):
    username: str = Field(..., description="Twitter username to analyze")
    max_tweets: int = Field(200, description="Maximum tweets to analyze", ge=10, le=1000)
    force_refresh: bool = Field(False, description="Force re-analysis even if cached")


class CompareRequest(BaseModel):
    username1: str = Field(..., description="First Twitter username")
    username2: str = Field(..., description="Second Twitter username")


class KOLResponse(BaseModel):
    username: str
    display_name: Optional[str] = None
    follower_count: Optional[int] = None
    latest_score: Optional[float] = None
    latest_grade: Optional[str] = None
    updated_at: Optional[str] = None


class AnalysisResponse(BaseModel):
    username: str
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


# Create FastAPI app
app = FastAPI(
    title="KOL Credibility Analyzer API",
    description="Analyze crypto Twitter KOLs for credibility, detect LARPing and manipulation",
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

# Initialize components
engine = CredibilityEngine()

# Initialize database (Supabase if available)
db = None
if SUPABASE_AVAILABLE:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        try:
            db = SupabaseDatabase(url=supabase_url, key=supabase_key)
        except Exception as e:
            print(f"Failed to initialize Supabase: {e}")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the web UI."""
    static_file = root_dir / "kol-analyzer" / "static" / "index.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text(), status_code=200)
    return HTMLResponse(
        content="""
        <html>
        <head><title>KOL Analyzer</title></head>
        <body>
            <h1>KOL Credibility Analyzer API</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for documentation.</p>
        </body>
        </html>
        """,
        status_code=200
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "KOL Credibility Analyzer",
        "version": "1.0.0",
        "supabase_connected": db is not None
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics."""
    if db:
        stats = db.get_stats()
        return StatsResponse(**stats)
    return StatsResponse(kols_analyzed=0, total_analyses=0, total_tweets=0, average_score=0)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_kol(request: AnalyzeRequest):
    """Analyze a KOL's credibility."""
    username = request.username.lstrip('@').lower()

    # Check cache first (unless force refresh)
    if db and not request.force_refresh:
        cached = db.get_latest_analysis(username)
        if cached:
            return AnalysisResponse(
                username=username,
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

    # Fetch tweets
    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    crawler = TwitterCrawler(rapidapi_key=rapidapi_key)

    try:
        await crawler.initialize()

        profile = await crawler.get_user_profile(username)
        if not profile:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        tweets = await crawler.get_user_tweets(
            username,
            max_tweets=min(request.max_tweets, 50)  # Limit for serverless
        )

        # Convert tweets to dict format
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

        # Run analysis
        result = engine.analyze(
            tweets_data,
            profile.follower_count,
            username
        )

        # Save to database if available
        if db:
            try:
                kol_id = db.upsert_kol(profile)
                db.save_tweets(kol_id, tweets)
                db.save_analysis(kol_id, result.to_dict(), len(tweets))
            except Exception as e:
                print(f"Failed to save to database: {e}")

        return AnalysisResponse(
            username=username,
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

    finally:
        await crawler.close()


@app.get("/kol/{username}")
async def get_kol(username: str):
    """Get cached analysis for a KOL."""
    username = username.lstrip('@').lower()

    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    kol = db.get_kol(username)
    if not kol:
        raise HTTPException(status_code=404, detail=f"No analysis found for @{username}")

    analysis = db.get_latest_analysis(username)

    return {
        "kol": kol,
        "analysis": analysis
    }


@app.get("/kols", response_model=List[KOLResponse])
async def list_kols(limit: int = 50, order_by: str = "updated_at"):
    """List all analyzed KOLs."""
    if not db:
        return []

    kols = db.list_kols(limit=limit, order_by=order_by)
    return [KOLResponse(**kol) for kol in kols]


@app.post("/compare")
async def compare_kols(request: CompareRequest):
    """Compare two KOLs' credibility."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    username1 = request.username1.lstrip('@').lower()
    username2 = request.username2.lstrip('@').lower()

    analysis1 = db.get_latest_analysis(username1)
    analysis2 = db.get_latest_analysis(username2)

    if not analysis1:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for @{username1}. Run /analyze first."
        )
    if not analysis2:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for @{username2}. Run /analyze first."
        )

    comparison = {
        'overall': {
            username1: analysis1['overall_score'],
            username2: analysis2['overall_score'],
            'winner': username1 if analysis1['overall_score'] > analysis2['overall_score'] else username2,
            'difference': abs(analysis1['overall_score'] - analysis2['overall_score'])
        },
        'grades': {
            username1: analysis1['grade'],
            username2: analysis2['grade']
        }
    }

    return {
        "comparison": comparison,
        "user1_analysis": analysis1,
        "user2_analysis": analysis2
    }


@app.delete("/cache/{username}")
async def clear_cache(username: str):
    """Clear cached data for a KOL."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    username = username.lstrip('@').lower()
    success = db.delete_kol_cache(username)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data found for @{username}"
        )

    return {"message": f"Cache cleared for @{username}"}


@app.get("/history/{username}")
async def get_analysis_history(username: str, limit: int = 10):
    """Get analysis history for a KOL."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    username = username.lstrip('@').lower()
    analyses = db.get_all_analyses(username, limit=limit)

    if not analyses:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis history found for @{username}"
        )

    return {"username": username, "analyses": analyses}


# Export for Vercel
handler = app
