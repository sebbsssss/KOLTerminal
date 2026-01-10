"""Supabase database client for KOL analysis data.

This module provides a scalable cloud-based storage solution that:
- Caches crawled KOL data to reduce API costs
- Supports incremental tweet fetching (only fetch new tweets)
- Enables sharing cached data across users
- Provides efficient queries with PostgreSQL and JSONB
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from supabase import create_client, Client

from ..scraper.twitter_crawler import Tweet, UserProfile


class SupabaseDatabase:
    """Supabase database client for KOL analysis data."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        cache_hours: int = 24
    ):
        """
        Initialize Supabase client.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase anon/service key. Defaults to SUPABASE_KEY env var.
            cache_hours: Hours before cache is considered stale. Default 24.
        """
        self.url = url or os.environ.get("SUPABASE_URL")
        self.key = key or os.environ.get("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them directly."
            )

        self.client: Client = create_client(self.url, self.key)
        self.cache_hours = cache_hours

    # =========================================================================
    # KOL OPERATIONS
    # =========================================================================

    def upsert_kol(self, profile: UserProfile, twitter_id: Optional[str] = None) -> str:
        """
        Insert or update a KOL profile.

        Args:
            profile: UserProfile dataclass
            twitter_id: Twitter's internal user ID (for API calls)

        Returns:
            The KOL's UUID.
        """
        data = {
            "username": profile.username.lower(),
            "display_name": profile.display_name,
            "bio": profile.bio,
            "follower_count": profile.follower_count,
            "following_count": profile.following_count,
            "tweet_count": profile.tweet_count,
            "verified": getattr(profile, 'verified', False),
            "profile_image_url": getattr(profile, 'profile_image_url', ''),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if twitter_id:
            data["twitter_id"] = twitter_id

        if hasattr(profile, 'joined_date') and profile.joined_date:
            data["joined_date"] = profile.joined_date

        # Upsert based on username
        result = self.client.table("kols").upsert(
            data,
            on_conflict="username"
        ).execute()

        return result.data[0]["id"]

    def get_kol(self, username: str) -> Optional[Dict[str, Any]]:
        """Get a KOL by username."""
        result = self.client.table("kols").select("*").eq(
            "username", username.lower()
        ).execute()

        return result.data[0] if result.data else None

    def get_kol_by_id(self, kol_id: str) -> Optional[Dict[str, Any]]:
        """Get a KOL by UUID."""
        result = self.client.table("kols").select("*").eq("id", kol_id).execute()
        return result.data[0] if result.data else None

    def update_kol_fetch_metadata(
        self,
        kol_id: str,
        last_tweet_id: Optional[str] = None
    ):
        """
        Update the KOL's fetch metadata after crawling.

        Args:
            kol_id: KOL's UUID
            last_tweet_id: ID of the most recent tweet fetched
        """
        data = {"last_fetched_at": datetime.now(timezone.utc).isoformat()}

        if last_tweet_id:
            data["last_tweet_id"] = last_tweet_id

        self.client.table("kols").update(data).eq("id", kol_id).execute()

    # =========================================================================
    # TWEET OPERATIONS
    # =========================================================================

    def save_tweets(self, kol_id: str, tweets: List[Tweet]) -> int:
        """
        Save tweets to the database (upsert to avoid duplicates).

        Args:
            kol_id: KOL's UUID
            tweets: List of Tweet dataclasses

        Returns:
            Number of tweets saved/updated.
        """
        if not tweets:
            return 0

        tweet_data = []
        for tweet in tweets:
            tweet_data.append({
                "kol_id": kol_id,
                "tweet_id": tweet.id,
                "text": tweet.text,
                "timestamp": tweet.timestamp,
                "likes": tweet.likes,
                "retweets": tweet.retweets,
                "replies": tweet.replies,
                "has_media": tweet.has_media,
                "has_video": tweet.has_video,
                "is_quote_tweet": tweet.is_quote_tweet,
                "is_reply": getattr(tweet, 'is_reply', False),
                "reply_to_user": getattr(tweet, 'reply_to', None),
            })

        # Batch upsert
        result = self.client.table("tweets").upsert(
            tweet_data,
            on_conflict="tweet_id"
        ).execute()

        # Update the KOL's last_tweet_id with the most recent tweet
        if tweets:
            # Sort by timestamp to find most recent
            sorted_tweets = sorted(tweets, key=lambda t: t.timestamp, reverse=True)
            self.update_kol_fetch_metadata(kol_id, sorted_tweets[0].id)

        return len(result.data)

    def get_tweets(
        self,
        kol_id: str,
        limit: int = 500,
        since_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tweets for a KOL.

        Args:
            kol_id: KOL's UUID
            limit: Maximum tweets to return
            since_id: Only return tweets newer than this ID

        Returns:
            List of tweet dictionaries.
        """
        query = self.client.table("tweets").select("*").eq(
            "kol_id", kol_id
        ).order("timestamp", desc=True).limit(limit)

        # Note: since_id filtering would need tweet timestamp comparison
        # as Twitter IDs aren't guaranteed to be sequential

        result = query.execute()
        return result.data

    def get_tweets_for_analysis(self, username: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get all cached tweets for a KOL by username.

        Args:
            username: KOL's Twitter username
            limit: Maximum tweets to return

        Returns:
            List of tweet dictionaries ready for analysis.
        """
        kol = self.get_kol(username)
        if not kol:
            return []

        return self.get_tweets(kol["id"], limit)

    def get_cached_tweet_count(self, username: str) -> int:
        """Get the number of cached tweets for a KOL."""
        kol = self.get_kol(username)
        if not kol:
            return 0

        result = self.client.table("tweets").select(
            "id", count="exact"
        ).eq("kol_id", kol["id"]).execute()

        return result.count or 0

    def get_latest_tweet_id(self, username: str) -> Optional[str]:
        """
        Get the ID of the most recent cached tweet for a KOL.

        Used for incremental fetching.
        """
        kol = self.get_kol(username)
        if not kol:
            return None

        return kol.get("last_tweet_id")

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def is_cache_fresh(self, username: str, max_age_hours: Optional[int] = None) -> bool:
        """
        Check if the cached data for a KOL is still fresh.

        Args:
            username: KOL's Twitter username
            max_age_hours: Override default cache_hours

        Returns:
            True if cache is fresh, False if stale or missing.
        """
        kol = self.get_kol(username)
        if not kol or not kol.get("last_fetched_at"):
            return False

        max_age = max_age_hours or self.cache_hours
        last_fetched = datetime.fromisoformat(
            kol["last_fetched_at"].replace("Z", "+00:00")
        )
        age = datetime.now(timezone.utc) - last_fetched

        return age < timedelta(hours=max_age)

    def get_cache_info(self, username: str) -> Dict[str, Any]:
        """
        Get cache status information for a KOL.

        Returns:
            Dict with cache status, tweet count, and staleness info.
        """
        kol = self.get_kol(username)
        if not kol:
            return {
                "cached": False,
                "tweet_count": 0,
                "last_fetched": None,
                "is_fresh": False,
                "hours_since_fetch": None
            }

        tweet_count = self.get_cached_tweet_count(username)
        last_fetched = kol.get("last_fetched_at")
        hours_since = None

        if last_fetched:
            last_fetched_dt = datetime.fromisoformat(
                last_fetched.replace("Z", "+00:00")
            )
            hours_since = (
                datetime.now(timezone.utc) - last_fetched_dt
            ).total_seconds() / 3600

        return {
            "cached": tweet_count > 0,
            "tweet_count": tweet_count,
            "last_fetched": last_fetched,
            "is_fresh": self.is_cache_fresh(username),
            "hours_since_fetch": round(hours_since, 1) if hours_since else None,
            "last_tweet_id": kol.get("last_tweet_id")
        }

    # =========================================================================
    # ANALYSIS OPERATIONS
    # =========================================================================

    def save_analysis(
        self,
        kol_id: str,
        analysis_data: Dict[str, Any],
        tweets_analyzed: int
    ) -> str:
        """
        Save an analysis result.

        Args:
            kol_id: KOL's UUID
            analysis_data: Analysis result dictionary
            tweets_analyzed: Number of tweets analyzed

        Returns:
            The analysis UUID.
        """
        data = {
            "kol_id": kol_id,
            "overall_score": analysis_data.get("overall_score"),
            "grade": analysis_data.get("grade"),
            "confidence": analysis_data.get("confidence"),
            "assessment": analysis_data.get("assessment"),
            "engagement_score": analysis_data.get("engagement_score"),
            "consistency_score": analysis_data.get("consistency_score"),
            "dissonance_score": analysis_data.get("dissonance_score"),
            "baiting_score": analysis_data.get("baiting_score"),
            "privilege_score": analysis_data.get("privilege_score"),
            "prediction_score": analysis_data.get("prediction_score"),
            "transparency_score": analysis_data.get("transparency_score"),
            "follower_quality_score": analysis_data.get("follower_quality_score"),
            "temporal_score": analysis_data.get("temporal_score"),
            "linguistic_score": analysis_data.get("linguistic_score"),
            "accountability_score": analysis_data.get("accountability_score"),
            "network_score": analysis_data.get("network_score"),
            "red_flags": analysis_data.get("red_flags", []),
            "green_flags": analysis_data.get("green_flags", []),
            "summary": analysis_data.get("summary"),
            "detailed_analysis": analysis_data.get("detailed_analysis", {}),
            "archetype": analysis_data.get("archetype"),
            "archetype_emoji": analysis_data.get("archetype_emoji"),
            "archetype_one_liner": analysis_data.get("archetype_one_liner"),
            "trust_level": analysis_data.get("trust_level"),
            "tweets_analyzed": tweets_analyzed,
        }

        result = self.client.table("analyses").insert(data).execute()
        analysis_id = result.data[0]["id"]

        # Update KOL's latest score
        self.client.table("kols").update({
            "latest_score": analysis_data.get("overall_score"),
            "latest_grade": analysis_data.get("grade"),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", kol_id).execute()

        return analysis_id

    def get_latest_analysis(self, username: str) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis for a KOL."""
        kol = self.get_kol(username)
        if not kol:
            return None

        result = self.client.table("analyses").select("*").eq(
            "kol_id", kol["id"]
        ).order("created_at", desc=True).limit(1).execute()

        if not result.data:
            return None

        analysis = result.data[0]
        # JSONB fields are already parsed by supabase-py
        return analysis

    def get_all_analyses(
        self,
        username: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get analysis history for a KOL."""
        kol = self.get_kol(username)
        if not kol:
            return []

        result = self.client.table("analyses").select("*").eq(
            "kol_id", kol["id"]
        ).order("created_at", desc=True).limit(limit).execute()

        return result.data

    # =========================================================================
    # LISTING & STATS
    # =========================================================================

    def list_kols(
        self,
        limit: int = 50,
        order_by: str = "updated_at",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """List all analyzed KOLs."""
        valid_columns = [
            "username", "follower_count", "latest_score",
            "updated_at", "created_at"
        ]

        if order_by not in valid_columns:
            order_by = "updated_at"

        result = self.client.table("kols").select("*").not_.is_(
            "latest_score", "null"
        ).order(order_by, desc=not ascending).limit(limit).execute()

        return result.data

    def get_leaderboard(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get KOL leaderboard ordered by score."""
        result = self.client.table("kol_leaderboard").select(
            "*"
        ).limit(limit).execute()

        return result.data

    def delete_kol_cache(self, username: str) -> bool:
        """Delete all cached data for a KOL."""
        kol = self.get_kol(username)
        if not kol:
            return False

        kol_id = kol["id"]

        # Delete tweets (cascade should handle this, but explicit is safer)
        self.client.table("tweets").delete().eq("kol_id", kol_id).execute()

        # Delete analyses
        self.client.table("analyses").delete().eq("kol_id", kol_id).execute()

        # Delete KOL
        self.client.table("kols").delete().eq("id", kol_id).execute()

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        kols_result = self.client.table("kols").select(
            "id", count="exact"
        ).execute()

        analyses_result = self.client.table("analyses").select(
            "id", count="exact"
        ).execute()

        tweets_result = self.client.table("tweets").select(
            "id", count="exact"
        ).execute()

        # Get average score
        scores_result = self.client.table("kols").select(
            "latest_score"
        ).not_.is_("latest_score", "null").execute()

        avg_score = 0
        if scores_result.data:
            scores = [k["latest_score"] for k in scores_result.data]
            avg_score = sum(scores) / len(scores)

        return {
            "kols_analyzed": kols_result.count or 0,
            "total_analyses": analyses_result.count or 0,
            "total_tweets": tweets_result.count or 0,
            "average_score": round(avg_score, 1)
        }

    def get_comparison_data(self, usernames: List[str]) -> List[Dict[str, Any]]:
        """Get comparison data for multiple KOLs."""
        results = []
        for username in usernames:
            kol = self.get_kol(username)
            if kol:
                analysis = self.get_latest_analysis(username)
                if analysis:
                    results.append({
                        "kol": kol,
                        "analysis": analysis
                    })
        return results

    # =========================================================================
    # API USAGE TRACKING
    # =========================================================================

    def track_api_usage(self, tweets_fetched: int = 0, api_calls: int = 1):
        """Track API usage for the current month."""
        current_month = datetime.now().strftime("%Y-%m")

        # Try to update existing record
        result = self.client.table("api_usage").select("*").eq(
            "month", current_month
        ).execute()

        if result.data:
            # Update existing
            existing = result.data[0]
            self.client.table("api_usage").update({
                "tweets_fetched": existing["tweets_fetched"] + tweets_fetched,
                "api_calls": existing["api_calls"] + api_calls,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("month", current_month).execute()
        else:
            # Insert new
            self.client.table("api_usage").insert({
                "month": current_month,
                "tweets_fetched": tweets_fetched,
                "api_calls": api_calls
            }).execute()

    def get_api_usage(self) -> Dict[str, Any]:
        """Get API usage for the current month."""
        current_month = datetime.now().strftime("%Y-%m")

        result = self.client.table("api_usage").select("*").eq(
            "month", current_month
        ).execute()

        if result.data:
            return result.data[0]

        return {
            "month": current_month,
            "tweets_fetched": 0,
            "api_calls": 0
        }


class IncrementalFetcher:
    """
    Helper class for incremental tweet fetching.

    Uses cached data to minimize API calls by only fetching new tweets.
    """

    def __init__(self, db: SupabaseDatabase):
        self.db = db

    def get_fetch_strategy(self, username: str) -> Dict[str, Any]:
        """
        Determine the best fetching strategy for a KOL.

        Returns:
            Dict with strategy info:
            - strategy: "full" | "incremental" | "cache_only"
            - since_id: Tweet ID to fetch from (for incremental)
            - cached_count: Number of cached tweets
            - reason: Explanation of strategy choice
        """
        cache_info = self.db.get_cache_info(username)

        if not cache_info["cached"]:
            return {
                "strategy": "full",
                "since_id": None,
                "cached_count": 0,
                "reason": "No cached data exists"
            }

        if cache_info["is_fresh"]:
            return {
                "strategy": "cache_only",
                "since_id": cache_info["last_tweet_id"],
                "cached_count": cache_info["tweet_count"],
                "reason": f"Cache is fresh ({cache_info['hours_since_fetch']:.1f}h old)"
            }

        return {
            "strategy": "incremental",
            "since_id": cache_info["last_tweet_id"],
            "cached_count": cache_info["tweet_count"],
            "reason": f"Cache is stale ({cache_info['hours_since_fetch']:.1f}h old), fetching new tweets only"
        }

    def merge_tweets(
        self,
        cached_tweets: List[Dict],
        new_tweets: List[Tweet]
    ) -> List[Dict]:
        """
        Merge cached tweets with newly fetched tweets.

        Deduplicates by tweet_id and sorts by timestamp.

        Returns:
            Combined list of tweets (as dicts), newest first.
        """
        # Convert new tweets to dicts
        new_tweet_dicts = []
        for tweet in new_tweets:
            new_tweet_dicts.append({
                "tweet_id": tweet.id,
                "text": tweet.text,
                "timestamp": tweet.timestamp,
                "likes": tweet.likes,
                "retweets": tweet.retweets,
                "replies": tweet.replies,
                "has_media": tweet.has_media,
                "has_video": tweet.has_video,
                "is_quote_tweet": tweet.is_quote_tweet,
            })

        # Create a dict keyed by tweet_id for deduplication
        all_tweets = {t["tweet_id"]: t for t in cached_tweets}

        # Add/update with new tweets
        for tweet in new_tweet_dicts:
            all_tweets[tweet["tweet_id"]] = tweet

        # Sort by timestamp descending
        sorted_tweets = sorted(
            all_tweets.values(),
            key=lambda t: t.get("timestamp", ""),
            reverse=True
        )

        return sorted_tweets
