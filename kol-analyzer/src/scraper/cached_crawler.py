"""
Cached Twitter Crawler with Supabase Integration.

This module wraps the TwitterCrawler to add intelligent caching:
- Checks Supabase for cached data before making API calls
- Only fetches new tweets when cache is stale (incremental fetching)
- Automatically saves fetched data to Supabase
- Dramatically reduces API costs by reusing cached data
"""

import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .twitter_crawler import Tweet, TwitterCrawler, UserProfile


class CachedTwitterCrawler:
    """
    Twitter crawler with Supabase caching layer.

    This wrapper provides:
    - Automatic caching of fetched tweets and profiles
    - Incremental fetching (only fetch new tweets since last crawl)
    - Cross-user cache sharing (everyone benefits from cached data)
    - Configurable cache freshness (default: 24 hours)
    """

    def __init__(
        self,
        rapidapi_key: str = None,
        supabase_url: str = None,
        supabase_key: str = None,
        max_tweets_per_user: int = 200,
        cache_hours: int = 24,
        usage_file: str = "data/api_usage.json"
    ):
        """
        Initialize the cached crawler.

        Args:
            rapidapi_key: RapidAPI key for Twitter241 API
            supabase_url: Supabase project URL
            supabase_key: Supabase service/anon key
            max_tweets_per_user: Maximum tweets to fetch per user
            cache_hours: Hours before cache is considered stale
            usage_file: Path to local API usage tracking file
        """
        # Initialize the base crawler
        self.crawler = TwitterCrawler(
            rapidapi_key=rapidapi_key,
            max_tweets_per_user=max_tweets_per_user,
            usage_file=usage_file
        )

        self.cache_hours = cache_hours
        self.max_tweets = max_tweets_per_user
        self._db = None

        # Store Supabase credentials for lazy initialization
        self._supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self._supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")

    @property
    def db(self):
        """Lazy initialization of Supabase client."""
        if self._db is None:
            if not self._supabase_url or not self._supabase_key:
                return None

            try:
                from ..storage.supabase_client import SupabaseDatabase
                self._db = SupabaseDatabase(
                    url=self._supabase_url,
                    key=self._supabase_key,
                    cache_hours=self.cache_hours
                )
            except ImportError:
                print("  [Warning] supabase package not installed. Caching disabled.")
                return None
            except Exception as e:
                print(f"  [Warning] Supabase init failed: {e}. Caching disabled.")
                return None

        return self._db

    @property
    def demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self.crawler.demo_mode

    @property
    def remaining_quota(self) -> int:
        """Get remaining monthly quota."""
        return self.crawler.remaining_quota

    @property
    def caching_enabled(self) -> bool:
        """Check if Supabase caching is available."""
        return self.db is not None

    async def initialize(self) -> bool:
        """
        Initialize the crawler and verify connections.

        Returns:
            True if API is authenticated, False if in demo mode.
        """
        result = await self.crawler.initialize()

        # Test Supabase connection
        if self.db:
            try:
                stats = self.db.get_stats()
                print(f"  [Supabase] Connected. Cache has {stats['total_tweets']} tweets from {stats['kols_analyzed']} KOLs")
            except Exception as e:
                print(f"  [Warning] Supabase connection test failed: {e}")

        return result

    async def get_user_profile(
        self,
        username: str,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[UserProfile]:
        """
        Get user profile with caching.

        Args:
            username: Twitter username
            force_refresh: Force fetch from API even if cached
            progress_callback: Optional callback for progress updates

        Returns:
            UserProfile or None if not found
        """
        # Check cache first
        if self.db and not force_refresh:
            cached_kol = self.db.get_kol(username)
            if cached_kol:
                if progress_callback:
                    progress_callback(f"Using cached profile for @{username}")

                return UserProfile(
                    username=cached_kol["username"],
                    display_name=cached_kol.get("display_name", username),
                    bio=cached_kol.get("bio", ""),
                    follower_count=cached_kol.get("follower_count", 0),
                    following_count=cached_kol.get("following_count", 0),
                    tweet_count=cached_kol.get("tweet_count", 0),
                    joined_date=cached_kol.get("joined_date", ""),
                    verified=cached_kol.get("verified", False),
                    profile_image_url=cached_kol.get("profile_image_url", "")
                )

        # Fetch from API
        profile = await self.crawler.get_user_profile(username, progress_callback)

        # Cache the result
        if profile and self.db:
            try:
                self.db.upsert_kol(profile)
            except Exception as e:
                print(f"  [Warning] Failed to cache profile: {e}")

        return profile

    async def get_user_tweets(
        self,
        username: str,
        max_tweets: int = 200,
        include_replies: bool = False,
        force_refresh: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[List[Tweet], Dict[str, Any]]:
        """
        Get user tweets with intelligent caching.

        This method:
        1. Checks if we have fresh cached data
        2. If cache is stale, fetches only NEW tweets since last crawl
        3. Merges new tweets with cached tweets
        4. Saves everything back to cache

        Args:
            username: Twitter username
            max_tweets: Maximum tweets to return
            include_replies: Include reply tweets
            force_refresh: Force full fetch even if cached
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (tweets list, cache_info dict)
        """
        cache_info = {
            "source": "api",
            "cached_count": 0,
            "new_count": 0,
            "total_count": 0,
            "from_cache": False
        }

        # No caching available - just fetch
        if not self.db:
            tweets = await self.crawler.get_user_tweets(
                username, max_tweets, include_replies, progress_callback
            )
            cache_info["new_count"] = len(tweets)
            cache_info["total_count"] = len(tweets)
            return tweets, cache_info

        # Get cache status
        kol = self.db.get_kol(username)
        kol_id = None

        if kol:
            kol_id = kol["id"]
            cache_status = self.db.get_cache_info(username)
            cache_info["cached_count"] = cache_status["tweet_count"]

            # If cache is fresh and not forcing refresh, use cached data
            if cache_status["is_fresh"] and not force_refresh:
                if progress_callback:
                    hours_ago = cache_status.get("hours_since_fetch", 0)
                    progress_callback(
                        f"Using cached data for @{username} "
                        f"({cache_status['tweet_count']} tweets, {hours_ago:.1f}h old)"
                    )

                cached_tweets = self.db.get_tweets(kol_id, limit=max_tweets)
                cache_info["source"] = "cache"
                cache_info["from_cache"] = True
                cache_info["total_count"] = len(cached_tweets)

                # Convert to Tweet objects
                tweets = self._dict_to_tweets(cached_tweets)
                return tweets, cache_info

        # Need to fetch - determine strategy
        if kol and cache_info["cached_count"] > 0 and not force_refresh:
            # Incremental fetch - only get new tweets
            if progress_callback:
                progress_callback(
                    f"Fetching new tweets for @{username} "
                    f"(have {cache_info['cached_count']} cached)"
                )

            # Fetch new tweets from API
            new_tweets = await self.crawler.get_user_tweets(
                username,
                max_tweets=min(max_tweets, 50),  # Limit API calls for incremental
                include_replies=include_replies,
                progress_callback=progress_callback
            )

            cache_info["new_count"] = len(new_tweets)
            cache_info["source"] = "incremental"

            # Save new tweets to cache
            if new_tweets:
                self.db.save_tweets(kol_id, new_tweets)
                self.db.track_api_usage(tweets_fetched=len(new_tweets))

            # Get combined tweets from cache
            all_cached = self.db.get_tweets(kol_id, limit=max_tweets)
            cache_info["total_count"] = len(all_cached)

            tweets = self._dict_to_tweets(all_cached)
            return tweets, cache_info

        else:
            # Full fetch - no cache or forcing refresh
            if progress_callback:
                progress_callback(f"Fetching all tweets for @{username}")

            new_tweets = await self.crawler.get_user_tweets(
                username, max_tweets, include_replies, progress_callback
            )

            cache_info["new_count"] = len(new_tweets)
            cache_info["source"] = "full_fetch"
            cache_info["total_count"] = len(new_tweets)

            # Ensure KOL exists in cache
            if not kol_id:
                profile = await self.get_user_profile(username)
                if profile:
                    kol_id = self.db.upsert_kol(profile)

            # Save to cache
            if kol_id and new_tweets:
                self.db.save_tweets(kol_id, new_tweets)
                self.db.track_api_usage(tweets_fetched=len(new_tweets))

            return new_tweets, cache_info

    def _dict_to_tweets(self, tweet_dicts: List[Dict]) -> List[Tweet]:
        """Convert cached tweet dictionaries to Tweet objects."""
        tweets = []
        for t in tweet_dicts:
            tweets.append(Tweet(
                id=t.get("tweet_id", t.get("id", "")),
                text=t.get("text", ""),
                timestamp=t.get("timestamp", ""),
                likes=t.get("likes", 0),
                retweets=t.get("retweets", 0),
                replies=t.get("replies", 0),
                has_media=t.get("has_media", False),
                has_video=t.get("has_video", False),
                is_quote_tweet=t.get("is_quote_tweet", False),
                is_reply=t.get("is_reply", False),
                reply_to=t.get("reply_to_user")
            ))
        return tweets

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get overall cache statistics."""
        if not self.db:
            return {"caching_enabled": False}

        stats = self.db.get_stats()
        api_usage = self.db.get_api_usage()

        return {
            "caching_enabled": True,
            "kols_cached": stats["kols_analyzed"],
            "tweets_cached": stats["total_tweets"],
            "analyses_cached": stats["total_analyses"],
            "average_score": stats["average_score"],
            "api_usage_this_month": api_usage["tweets_fetched"],
            "api_calls_this_month": api_usage["api_calls"]
        }

    async def close(self):
        """Clean up resources."""
        await self.crawler.close()


# Convenience function to create a cached crawler with env vars
def create_cached_crawler(
    cache_hours: int = 24,
    max_tweets: int = 200
) -> CachedTwitterCrawler:
    """
    Create a CachedTwitterCrawler with credentials from environment variables.

    Expected env vars:
    - RAPIDAPI_KEY: RapidAPI key for Twitter241
    - SUPABASE_URL: Supabase project URL
    - SUPABASE_KEY: Supabase service/anon key

    Args:
        cache_hours: Hours before cache is considered stale
        max_tweets: Maximum tweets to fetch per user

    Returns:
        Configured CachedTwitterCrawler instance
    """
    return CachedTwitterCrawler(
        rapidapi_key=os.environ.get("RAPIDAPI_KEY"),
        supabase_url=os.environ.get("SUPABASE_URL"),
        supabase_key=os.environ.get("SUPABASE_KEY"),
        max_tweets_per_user=max_tweets,
        cache_hours=cache_hours
    )
