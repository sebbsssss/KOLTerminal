#!/usr/bin/env python3
"""
Bulk tweet population script.

Fetches 200-300 tweets for all existing users in the database
using RapidAPI Twitter241 with pagination.

Usage:
    python scripts/populate_tweets.py                    # Populate all users
    python scripts/populate_tweets.py --user cobie       # Populate specific user
    python scripts/populate_tweets.py --dry-run          # Preview without fetching
    python scripts/populate_tweets.py --target 300       # Set target tweets per user
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")

from src.storage.supabase_client import SupabaseDatabase, Tweet, UserProfile


class BulkTweetFetcher:
    """
    Bulk tweet fetcher with pagination support.
    Bypasses normal quota limits for initial database population.
    """

    def __init__(self, rapidapi_key: str, target_tweets: int = 250):
        self.rapidapi_key = rapidapi_key
        self.target_tweets = target_tweets
        self.headers = {
            "x-rapidapi-key": rapidapi_key,
            "x-rapidapi-host": "twitter241.p.rapidapi.com"
        }
        self.base_url = "https://twitter241.p.rapidapi.com"
        self.request_delay = 1.5  # Seconds between requests to avoid rate limits

    def get_user_id(self, username: str) -> Optional[str]:
        """Get Twitter user ID from username."""
        url = f"{self.base_url}/user"
        params = {"username": username}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                user_id = data.get("result", {}).get("data", {}).get("user", {}).get("result", {}).get("rest_id")
                return user_id
            else:
                print(f"  Error getting user ID: {response.status_code} - {response.text[:200]}")
                return None
        except Exception as e:
            print(f"  Exception getting user ID: {e}")
            return None

    def get_user_profile(self, username: str) -> Optional[UserProfile]:
        """Get user profile data."""
        url = f"{self.base_url}/user"
        params = {"username": username}

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                user_result = data.get("result", {}).get("data", {}).get("user", {}).get("result", {})
                legacy = user_result.get("legacy", {})

                return UserProfile(
                    username=legacy.get("screen_name", username),
                    display_name=legacy.get("name", username),
                    bio=legacy.get("description", ""),
                    follower_count=legacy.get("followers_count", 0),
                    following_count=legacy.get("friends_count", 0),
                    tweet_count=legacy.get("statuses_count", 0),
                    joined_date=legacy.get("created_at", ""),
                    verified=legacy.get("verified", False) or user_result.get("is_blue_verified", False),
                    profile_image_url=legacy.get("profile_image_url_https", "")
                )
            return None
        except Exception as e:
            print(f"  Exception getting profile: {e}")
            return None

    def fetch_tweets_page(
        self,
        user_id: str,
        cursor: Optional[str] = None,
        count: int = 40
    ) -> tuple[List[Tweet], Optional[str]]:
        """
        Fetch a single page of tweets.

        Returns:
            Tuple of (tweets list, next cursor or None)
        """
        url = f"{self.base_url}/user-tweets"
        params = {"user": user_id, "count": str(count)}

        if cursor:
            params["cursor"] = cursor

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=20)

            if response.status_code != 200:
                print(f"  API error: {response.status_code}")
                return [], None

            result = response.json()
            tweets = []
            next_cursor = None

            # Parse the timeline instructions
            instructions = result.get("result", {}).get("timeline", {}).get("instructions", [])

            for instruction in instructions:
                # Handle different instruction types
                entries = instruction.get("entries", [])

                # Check for pinned entry
                if instruction.get("type") == "TimelinePinEntry":
                    entries = [instruction.get("entry", {})]

                for entry in entries:
                    entry_id = entry.get("entryId", "")

                    # Extract cursor for pagination
                    if "cursor-bottom" in entry_id:
                        cursor_content = entry.get("content", {})
                        next_cursor = cursor_content.get("value") or cursor_content.get("itemContent", {}).get("value")
                        continue

                    # Skip non-tweet entries
                    if not entry_id.startswith("tweet-"):
                        continue

                    content = entry.get("content", {})
                    item_content = content.get("itemContent", {})
                    tweet_results = item_content.get("tweet_results", {}).get("result", {})

                    # Handle different tweet types
                    typename = tweet_results.get("__typename", "")
                    if typename == "TweetWithVisibilityResults":
                        tweet_results = tweet_results.get("tweet", {})
                    elif typename != "Tweet":
                        continue

                    legacy = tweet_results.get("legacy", {})
                    if not legacy:
                        continue

                    # Extract tweet data
                    text = legacy.get("full_text", "")

                    # Skip retweets (they start with "RT @")
                    if text.startswith("RT @"):
                        continue

                    has_media = "media" in legacy.get("entities", {})
                    has_video = any(
                        m.get("type") == "video"
                        for m in legacy.get("extended_entities", {}).get("media", [])
                    )

                    tweet = Tweet(
                        id=tweet_results.get("rest_id", ""),
                        text=text,
                        timestamp=legacy.get("created_at", ""),
                        likes=legacy.get("favorite_count", 0),
                        retweets=legacy.get("retweet_count", 0),
                        replies=legacy.get("reply_count", 0),
                        has_media=has_media,
                        has_video=has_video,
                        is_quote_tweet=legacy.get("is_quote_status", False),
                        is_reply=legacy.get("in_reply_to_status_id_str") is not None,
                        reply_to=legacy.get("in_reply_to_screen_name")
                    )
                    tweets.append(tweet)

            return tweets, next_cursor

        except Exception as e:
            print(f"  Exception fetching tweets: {e}")
            return [], None

    def fetch_all_tweets(
        self,
        username: str,
        max_tweets: int = 250,
        progress_callback=None
    ) -> List[Tweet]:
        """
        Fetch up to max_tweets for a user using pagination.
        """
        print(f"\n  Fetching tweets for @{username} (target: {max_tweets})")

        # Get user ID first
        user_id = self.get_user_id(username)
        if not user_id:
            print(f"  Could not find user ID for @{username}")
            return []

        print(f"  Found user ID: {user_id}")

        all_tweets = []
        cursor = None
        page = 0
        max_pages = 15  # Safety limit
        seen_ids = set()

        while len(all_tweets) < max_tweets and page < max_pages:
            page += 1
            print(f"  Fetching page {page}... (have {len(all_tweets)} tweets)")

            tweets, next_cursor = self.fetch_tweets_page(user_id, cursor)

            if not tweets:
                print(f"  No more tweets returned")
                break

            # Deduplicate
            new_tweets = 0
            for tweet in tweets:
                if tweet.id not in seen_ids:
                    seen_ids.add(tweet.id)
                    all_tweets.append(tweet)
                    new_tweets += 1

            print(f"  Got {new_tweets} new tweets from page {page}")

            if not next_cursor or next_cursor == cursor:
                print(f"  No more pages (cursor exhausted)")
                break

            cursor = next_cursor

            # Rate limiting delay
            if page < max_pages and len(all_tweets) < max_tweets:
                time.sleep(self.request_delay)

        print(f"  Total: {len(all_tweets)} tweets for @{username}")
        return all_tweets[:max_tweets]


async def populate_user(
    db: SupabaseDatabase,
    fetcher: BulkTweetFetcher,
    username: str,
    target_tweets: int,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Populate tweets for a single user."""
    result = {
        "username": username,
        "success": False,
        "tweets_fetched": 0,
        "tweets_saved": 0,
        "error": None
    }

    try:
        # Check existing tweet count
        existing_count = db.get_cached_tweet_count(username)
        print(f"\n{'='*50}")
        print(f"Processing @{username}")
        print(f"  Existing tweets in DB: {existing_count}")

        if existing_count >= target_tweets:
            print(f"  Already have {existing_count} tweets (target: {target_tweets}). Skipping.")
            result["success"] = True
            result["tweets_fetched"] = 0
            result["tweets_saved"] = existing_count
            return result

        needed = target_tweets - existing_count
        print(f"  Need to fetch: {needed} more tweets")

        if dry_run:
            print(f"  [DRY RUN] Would fetch {needed} tweets")
            result["success"] = True
            return result

        # Fetch profile first to ensure KOL exists in DB
        profile = fetcher.get_user_profile(username)
        if profile:
            kol_id = db.upsert_kol(profile)
            print(f"  KOL ID: {kol_id}")
        else:
            # Try to get existing KOL
            kol = db.get_kol(username)
            if not kol:
                print(f"  Could not get profile for @{username}")
                result["error"] = "Could not fetch profile"
                return result
            kol_id = kol["id"]

        # Fetch tweets
        tweets = fetcher.fetch_all_tweets(username, max_tweets=target_tweets)
        result["tweets_fetched"] = len(tweets)

        if tweets:
            # Save to database
            saved = db.save_tweets(kol_id, tweets)
            result["tweets_saved"] = saved
            print(f"  Saved {saved} tweets to database")
            result["success"] = True
        else:
            print(f"  No tweets fetched for @{username}")
            result["error"] = "No tweets returned from API"

    except Exception as e:
        print(f"  Error processing @{username}: {e}")
        result["error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Bulk populate tweets for existing users")
    parser.add_argument("--user", "-u", type=str, help="Specific username to populate")
    parser.add_argument("--target", "-t", type=int, default=250, help="Target tweets per user (default: 250)")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Preview without fetching")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit number of users to process")
    args = parser.parse_args()

    # Check for API key
    rapidapi_key = os.environ.get("RAPIDAPI_KEY")
    if not rapidapi_key:
        print("ERROR: RAPIDAPI_KEY environment variable not set")
        print("Set it in your .env file or export RAPIDAPI_KEY=your_key")
        sys.exit(1)

    # Check for Supabase credentials
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print("ERROR: SUPABASE_URL and SUPABASE_KEY environment variables required")
        sys.exit(1)

    print("="*60)
    print("BULK TWEET POPULATION SCRIPT")
    print("="*60)
    print(f"Target tweets per user: {args.target}")
    print(f"Dry run: {args.dry_run}")

    # Initialize clients
    db = SupabaseDatabase()
    fetcher = BulkTweetFetcher(rapidapi_key, target_tweets=args.target)

    # Get users to process
    if args.user:
        usernames = [args.user.lower()]
        print(f"\nProcessing single user: @{args.user}")
    else:
        # Get all KOLs from database
        kols = db.list_kols(limit=100, order_by="updated_at")
        usernames = [kol["username"] for kol in kols]

        if args.limit > 0:
            usernames = usernames[:args.limit]

        print(f"\nFound {len(usernames)} users in database")

    if not usernames:
        print("No users found to process")
        sys.exit(0)

    print(f"Users to process: {', '.join(usernames)}")

    # Process each user
    results = []
    for i, username in enumerate(usernames, 1):
        print(f"\n[{i}/{len(usernames)}] Processing @{username}")
        result = await populate_user(db, fetcher, username, args.target, args.dry_run)
        results.append(result)

        # Add delay between users to avoid rate limits
        if i < len(usernames) and not args.dry_run:
            print("  Waiting 3 seconds before next user...")
            time.sleep(3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    total_fetched = sum(r["tweets_fetched"] for r in results)
    total_saved = sum(r["tweets_saved"] for r in results)

    print(f"Users processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total tweets fetched: {total_fetched}")
    print(f"Total tweets saved: {total_saved}")

    if failed:
        print("\nFailed users:")
        for r in failed:
            print(f"  - @{r['username']}: {r['error']}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
