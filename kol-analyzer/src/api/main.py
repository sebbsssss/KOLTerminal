"""FastAPI server for KOL Credibility Analyzer."""

import asyncio
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scraper.twitter_crawler import TwitterCrawler
from src.storage.database import Database
from src.analysis.credibility_engine import CredibilityEngine


# Pydantic models for API
class AnalyzeRequest(BaseModel):
    username: str = Field(..., description="Twitter username to analyze")
    max_tweets: int = Field(200, description="Maximum tweets to analyze", ge=10, le=1000)
    force_refresh: bool = Field(False, description="Force re-analysis even if cached")


class CompareRequest(BaseModel):
    username1: str = Field(..., description="First Twitter username")
    username2: str = Field(..., description="Second Twitter username")


class KOLResponse(BaseModel):
    username: str
    display_name: Optional[str]
    follower_count: Optional[int]
    latest_score: Optional[float]
    latest_grade: Optional[str]
    updated_at: Optional[str]


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


class CompareResponse(BaseModel):
    comparison: dict
    user1_analysis: dict
    user2_analysis: dict


class StatsResponse(BaseModel):
    kols_analyzed: int
    total_analyses: int
    total_tweets: int
    average_score: float


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

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
    db = Database()
    engine = CredibilityEngine()

    @app.get("/", tags=["Health"])
    async def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "KOL Credibility Analyzer",
            "version": "1.0.0"
        }

    @app.get("/stats", response_model=StatsResponse, tags=["Stats"])
    async def get_stats():
        """Get database statistics."""
        stats = db.get_stats()
        return StatsResponse(**stats)

    @app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
    async def analyze_kol(request: AnalyzeRequest):
        """
        Analyze a KOL's credibility.

        This endpoint fetches tweets and runs comprehensive analysis.
        Results are cached in the database.
        """
        username = request.username.lstrip('@').lower()

        # Check cache first (unless force refresh)
        if not request.force_refresh:
            cached = db.get_latest_analysis(username)
            if cached:
                kol = db.get_kol(username)
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
        crawler = TwitterCrawler()
        try:
            await crawler.initialize()

            profile = await crawler.get_user_profile(username)
            if not profile:
                raise HTTPException(status_code=404, detail=f"User @{username} not found")

            tweets = await crawler.get_user_tweets(
                username,
                max_tweets=request.max_tweets
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

            # Save to database
            kol_id = db.upsert_kol(profile)
            db.save_tweets(kol_id, tweets)
            db.save_analysis(kol_id, result.to_dict(), len(tweets))

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

    @app.get("/kol/{username}", tags=["KOLs"])
    async def get_kol(username: str):
        """Get cached analysis for a KOL."""
        username = username.lstrip('@').lower()

        kol = db.get_kol(username)
        if not kol:
            raise HTTPException(status_code=404, detail=f"No analysis found for @{username}")

        analysis = db.get_latest_analysis(username)

        return {
            "kol": kol,
            "analysis": analysis
        }

    @app.get("/kols", response_model=List[KOLResponse], tags=["KOLs"])
    async def list_kols(
        limit: int = 50,
        order_by: str = "updated_at"
    ):
        """List all analyzed KOLs."""
        kols = db.list_kols(limit=limit, order_by=order_by)
        return [KOLResponse(**kol) for kol in kols]

    @app.post("/compare", response_model=CompareResponse, tags=["Analysis"])
    async def compare_kols(request: CompareRequest):
        """Compare two KOLs' credibility."""
        username1 = request.username1.lstrip('@').lower()
        username2 = request.username2.lstrip('@').lower()

        # Get analyses for both users
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

        # Build comparison
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
            },
            'engagement': {
                username1: analysis1['engagement_score'],
                username2: analysis2['engagement_score']
            },
            'consistency': {
                username1: analysis1['consistency_score'],
                username2: analysis2['consistency_score']
            },
            'dissonance': {
                username1: analysis1['dissonance_score'],
                username2: analysis2['dissonance_score']
            },
            'baiting': {
                username1: analysis1['baiting_score'],
                username2: analysis2['baiting_score']
            }
        }

        return CompareResponse(
            comparison=comparison,
            user1_analysis=analysis1,
            user2_analysis=analysis2
        )

    @app.delete("/cache/{username}", tags=["Cache"])
    async def clear_cache(username: str):
        """Clear cached data for a KOL."""
        username = username.lstrip('@').lower()

        success = db.delete_kol_cache(username)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"No cached data found for @{username}"
            )

        return {"message": f"Cache cleared for @{username}"}

    @app.get("/history/{username}", tags=["Analysis"])
    async def get_analysis_history(username: str, limit: int = 10):
        """Get analysis history for a KOL."""
        username = username.lstrip('@').lower()

        analyses = db.get_all_analyses(username, limit=limit)
        if not analyses:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis history found for @{username}"
            )

        return {"username": username, "analyses": analyses}

    return app


# Create default app instance
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
