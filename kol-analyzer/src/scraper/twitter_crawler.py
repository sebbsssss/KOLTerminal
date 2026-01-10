"""Twitter/X crawler with API support and demo mode fallback."""

import asyncio
import json
import random
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .rate_limiter import SimpleRateLimiter

# Try to import tweepy for API access
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False


@dataclass
class UserProfile:
    """Twitter user profile data."""
    username: str
    display_name: str
    bio: str
    follower_count: int
    following_count: int
    tweet_count: int
    joined_date: str
    verified: bool = False
    profile_image_url: str = ""


@dataclass
class Tweet:
    """Tweet data structure."""
    id: str
    text: str
    timestamp: str  # ISO format
    likes: int
    retweets: int
    replies: int
    has_media: bool = False
    has_video: bool = False
    is_quote_tweet: bool = False
    is_reply: bool = False
    reply_to: Optional[str] = None


class TwitterCrawler:
    """
    Twitter/X crawler with RapidAPI support.
    Falls back to demo mode when API is unavailable.
    """

    # Rate limiting: track monthly usage
    MONTHLY_LIMIT = 100
    DEFAULT_TWEETS_PER_USER = 10  # Conservative to save quota

    def __init__(
        self,
        rapidapi_key: str = None,
        max_tweets_per_user: int = 10,
        usage_file: str = "data/api_usage.json"
    ):
        # RapidAPI credentials - check env vars or use provided
        self.rapidapi_key = rapidapi_key or os.environ.get("RAPIDAPI_KEY", "")

        self.max_tweets = min(max_tweets_per_user, self.DEFAULT_TWEETS_PER_USER)
        self.usage_file = Path(usage_file)
        self.rate_limiter = SimpleRateLimiter()

        self._demo_mode = not self.rapidapi_key
        self._authenticated = False

        # Load usage tracking
        self._usage = self._load_usage()

    def _load_usage(self) -> dict:
        """Load API usage tracking."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file, 'r') as f:
                    usage = json.load(f)
                    # Reset if new month
                    if usage.get('month') != datetime.now().strftime('%Y-%m'):
                        return {'month': datetime.now().strftime('%Y-%m'), 'tweets_fetched': 0}
                    return usage
            except:
                pass
        return {'month': datetime.now().strftime('%Y-%m'), 'tweets_fetched': 0}

    def _save_usage(self):
        """Save API usage tracking."""
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_file, 'w') as f:
            json.dump(self._usage, f)

    def _can_fetch(self, count: int) -> bool:
        """Check if we have quota remaining."""
        return self._usage['tweets_fetched'] + count <= self.MONTHLY_LIMIT

    def _record_usage(self, count: int):
        """Record tweets fetched."""
        self._usage['tweets_fetched'] += count
        self._save_usage()
        remaining = self.MONTHLY_LIMIT - self._usage['tweets_fetched']
        print(f"  [API] Used {count} tweets. Remaining this month: {remaining}/{self.MONTHLY_LIMIT}")

    @property
    def demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self._demo_mode or not self._authenticated

    @property
    def remaining_quota(self) -> int:
        """Get remaining monthly quota."""
        return max(0, self.MONTHLY_LIMIT - self._usage['tweets_fetched'])

    async def initialize(self) -> bool:
        """
        Initialize the RapidAPI client (Twitter241 API).

        Returns:
            True if successfully authenticated, False if in demo mode.
        """
        if not self.rapidapi_key:
            print("  [Demo Mode] No RapidAPI key - using simulated data")
            print("  Get a free key at: https://rapidapi.com/davethebeast/api/twitter241")
            self._demo_mode = True
            return False

        try:
            import requests

            # Test the API with a simple request
            url = "https://twitter241.p.rapidapi.com/user"
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "twitter241.p.rapidapi.com"
            }
            params = {"username": "elonmusk"}  # Test with a known account

            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                self._authenticated = True
                self._demo_mode = False
                remaining = self.remaining_quota
                print(f"  [RapidAPI] Twitter241 API authenticated. Quota remaining: {remaining}/{self.MONTHLY_LIMIT}")
                return True
            elif response.status_code == 403:
                raise Exception("Invalid RapidAPI key or not subscribed to this API")
            else:
                raise Exception(f"API test failed: {response.status_code}")

        except Exception as e:
            print(f"  [Demo Mode] RapidAPI auth failed: {e}")
            self._demo_mode = True
            return False

    async def get_user_profile(
        self,
        username: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[UserProfile]:
        """
        Get a Twitter user's profile information using Twitter241 API.
        """
        if self.demo_mode:
            return self._get_demo_profile(username)

        try:
            import requests

            if progress_callback:
                progress_callback(f"Fetching profile for @{username}")

            url = "https://twitter241.p.rapidapi.com/user"
            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "twitter241.p.rapidapi.com"
            }
            params = {"username": username}

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                user_result = data.get("result", {}).get("data", {}).get("user", {}).get("result", {})
                legacy = user_result.get("legacy", {})
                core = user_result.get("core", {})

                return UserProfile(
                    username=core.get("screen_name", legacy.get("screen_name", username)),
                    display_name=core.get("name", legacy.get("name", username)),
                    bio=legacy.get("description", ""),
                    follower_count=legacy.get("followers_count", 0),
                    following_count=legacy.get("friends_count", 0),
                    tweet_count=legacy.get("statuses_count", 0),
                    joined_date=core.get("created_at", legacy.get("created_at", "")),
                    verified=legacy.get("verified", False) or user_result.get("is_blue_verified", False),
                    profile_image_url=legacy.get("profile_image_url_https", "")
                )
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching profile: {e}")
            print(f"  [Warning] RapidAPI error for @{username}, using demo data: {e}")
            return self._get_demo_profile(username)

    async def get_user_tweets(
        self,
        username: str,
        max_tweets: int = 10,
        include_replies: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Tweet]:
        """
        Get a user's tweets using Twitter241 API (limited to save quota).
        """
        # Enforce conservative limit
        max_tweets = min(max_tweets, self.max_tweets, 10)

        if self.demo_mode:
            return self._get_demo_tweets(username, max_tweets, progress_callback)

        # Check quota
        if not self._can_fetch(max_tweets):
            print(f"  [Warning] API quota exhausted ({self._usage['tweets_fetched']}/{self.MONTHLY_LIMIT}). Using demo data.")
            return self._get_demo_tweets(username, max_tweets, progress_callback)

        try:
            import requests

            if progress_callback:
                progress_callback(f"Fetching tweets for @{username} (max {max_tweets})")

            headers = {
                "x-rapidapi-key": self.rapidapi_key,
                "x-rapidapi-host": "twitter241.p.rapidapi.com"
            }

            # First get user ID
            user_url = "https://twitter241.p.rapidapi.com/user"
            user_response = requests.get(user_url, headers=headers, params={"username": username}, timeout=15)

            if user_response.status_code != 200:
                raise Exception(f"User lookup failed: {user_response.text}")

            user_data = user_response.json()
            user_id = user_data.get("result", {}).get("data", {}).get("user", {}).get("result", {}).get("rest_id")

            if not user_id:
                raise Exception("Could not get user ID")

            # Now get tweets
            tweets_url = "https://twitter241.p.rapidapi.com/user-tweets"
            params = {"user": user_id, "count": str(max_tweets)}

            response = requests.get(tweets_url, headers=headers, params=params, timeout=20)

            if response.status_code != 200:
                raise Exception(f"Tweets API error: {response.status_code} - {response.text}")

            tweets = []
            result = response.json()

            # Parse the timeline instructions
            instructions = result.get("result", {}).get("timeline", {}).get("instructions", [])

            for instruction in instructions:
                entries = instruction.get("entries", [])
                if instruction.get("type") == "TimelinePinEntry":
                    entries = [instruction.get("entry", {})]

                for entry in entries:
                    if len(tweets) >= max_tweets:
                        break

                    content = entry.get("content", {})
                    item_content = content.get("itemContent", {})
                    tweet_results = item_content.get("tweet_results", {}).get("result", {})

                    if tweet_results.get("__typename") != "Tweet":
                        continue

                    legacy = tweet_results.get("legacy", {})

                    # Get text
                    text = legacy.get("full_text", "")

                    # Get metrics
                    likes = legacy.get("favorite_count", 0)
                    retweets = legacy.get("retweet_count", 0)
                    replies = legacy.get("reply_count", 0)

                    # Check for media
                    has_media = "media" in legacy.get("entities", {})
                    has_video = any(
                        m.get("type") == "video"
                        for m in legacy.get("extended_entities", {}).get("media", [])
                    )

                    tweet = Tweet(
                        id=tweet_results.get("rest_id", ""),
                        text=text,
                        timestamp=legacy.get("created_at", ""),
                        likes=likes,
                        retweets=retweets,
                        replies=replies,
                        has_media=has_media,
                        has_video=has_video,
                        is_quote_tweet=legacy.get("is_quote_status", False),
                        is_reply=legacy.get("in_reply_to_status_id_str") is not None,
                        reply_to=legacy.get("in_reply_to_screen_name")
                    )
                    tweets.append(tweet)

            # Record usage
            self._record_usage(len(tweets))

            if progress_callback:
                progress_callback(f"Fetched {len(tweets)} tweets for @{username}")

            return tweets

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching tweets: {e}")
            print(f"  [Warning] RapidAPI error for @{username}, using demo data: {e}")
            return self._get_demo_tweets(username, max_tweets, progress_callback)

    async def close(self):
        """Clean up resources."""
        self._client = None

    # =========================================================================
    # Demo Mode Data Generation
    # =========================================================================

    def _get_demo_profile(self, username: str) -> UserProfile:
        """Get a demo profile for testing."""
        profiles = self._get_demo_profiles()

        if username.lower() in profiles:
            return profiles[username.lower()]

        # Generate consistent profile for unknown users
        random.seed(hash(username.lower()))
        profile = UserProfile(
            username=username,
            display_name=f"{username.title()} (Demo)",
            bio="Crypto enthusiast | Not financial advice | DYOR",
            follower_count=random.randint(1000, 50000),
            following_count=random.randint(100, 2000),
            tweet_count=random.randint(500, 10000),
            joined_date="2021-01-15",
            verified=random.random() > 0.8
        )
        random.seed()
        return profile

    def _get_demo_profiles(self) -> Dict[str, UserProfile]:
        """Get predefined demo profiles."""
        return {
            "minhxdynasty": UserProfile(
                username="MINHxDYNASTY",
                display_name="Minh | Dynasty",
                bio="DeFi degen | Memecoin hunter | CT regular | Sometimes right, mostly early | NFA",
                follower_count=45000,
                following_count=890,
                tweet_count=12500,
                joined_date="2021-03-22",
                verified=False
            ),
            "cobie": UserProfile(
                username="cobie",
                display_name="Cobie",
                bio="I talk about crypto sometimes",
                follower_count=750000,
                following_count=450,
                tweet_count=45000,
                joined_date="2017-06-15",
                verified=True
            ),
            "zachxbt": UserProfile(
                username="zachxbt",
                display_name="ZachXBT",
                bio="On-chain sleuth | Exposing scams and fraud in crypto | DMs open for tips",
                follower_count=620000,
                following_count=320,
                tweet_count=28000,
                joined_date="2018-09-10",
                verified=True
            ),
        }

    def _get_demo_tweets(
        self,
        username: str,
        max_tweets: int,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Tweet]:
        """Generate demo tweets for testing."""
        tweet_templates = self._get_demo_tweet_templates(username.lower())
        tweets = []

        random.seed(hash(username.lower()))
        base_time = datetime.now()

        for i in range(max_tweets):
            template = random.choice(tweet_templates)

            # Generate engagement
            base_likes = random.randint(50, 5000)
            likes = int(base_likes * random.uniform(0.5, 2.0))
            retweets = int(likes * random.uniform(0.05, 0.25))
            replies = int(likes * random.uniform(0.02, 0.15))

            hours_ago = random.uniform(1, 72)
            timestamp = base_time - timedelta(hours=hours_ago)

            tweet = Tweet(
                id=f"demo_{username}_{i}_{random.randint(1000000, 9999999)}",
                text=template['text'],
                timestamp=timestamp.isoformat(),
                likes=likes,
                retweets=retweets,
                replies=replies,
                has_media=template.get('has_media', False),
                has_video=template.get('has_video', False),
                is_quote_tweet=template.get('is_quote', False)
            )
            tweets.append(tweet)

        random.seed()

        if progress_callback:
            progress_callback(f"Generated {len(tweets)} demo tweets for @{username}")

        return tweets

    def _get_demo_tweet_templates(self, username: str) -> List[Dict]:
        """Get tweet templates."""
        return [
            {"text": "$BTC looking bullish here. Accumulation zone.", "has_media": True},
            {"text": "$SOL ecosystem is growing fast. 100x potential.", "has_media": False},
            {"text": "If you're not accumulating here, you'll regret it.", "has_media": False},
            {"text": "Market looking weak. Time to derisk.", "has_media": True},
            {"text": "What's your highest conviction play? Drop below 👇", "has_media": False},
            {"text": "INSIDER INFO: This token is about to pump.", "has_media": False},
            {"text": "Have fun staying poor. 🤡", "has_media": False},
            {"text": "Here's a thread on identifying rug pulls 🧵", "has_media": False},
            {"text": "GM everyone. Great day in the markets.", "has_media": False},
            {"text": "Taking profits here. Don't be greedy.", "has_media": False},
        ]
