"""Twitter/X crawler with Playwright automation and demo mode fallback."""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re

from .rate_limiter import HumanLikeRateLimiter, SimpleRateLimiter

# Try to import Playwright
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


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
    Twitter/X crawler with Playwright browser automation.
    Falls back to demo mode when Playwright is unavailable or not authenticated.
    """

    def __init__(
        self,
        cookies_path: str = "data/cookies.json",
        headless: bool = True,
        timeout: int = 30000
    ):
        self.cookies_path = Path(cookies_path)
        self.headless = headless
        self.timeout = timeout
        self.browser: Optional[Any] = None
        self.page: Optional[Any] = None
        self.rate_limiter = HumanLikeRateLimiter()
        self._demo_mode = not PLAYWRIGHT_AVAILABLE
        self._authenticated = False

    @property
    def demo_mode(self) -> bool:
        """Check if running in demo mode."""
        return self._demo_mode or not self._authenticated

    async def initialize(self) -> bool:
        """
        Initialize the browser and check authentication.

        Returns:
            True if successfully authenticated, False if in demo mode.
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("  [Demo Mode] Playwright not installed - using simulated data")
            self._demo_mode = True
            self.rate_limiter = SimpleRateLimiter()
            return False

        try:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=self.headless)

            # Try to load cookies
            if self.cookies_path.exists():
                context = await self.browser.new_context()
                with open(self.cookies_path, 'r') as f:
                    cookies = json.load(f)
                await context.add_cookies(cookies)
                self.page = await context.new_page()

                # Verify authentication
                await self.page.goto("https://twitter.com/home", timeout=self.timeout)
                await asyncio.sleep(2)

                if "login" not in self.page.url.lower():
                    self._authenticated = True
                    self._demo_mode = False
                    print("  [Authenticated] Using real Twitter data")
                    return True

            print("  [Demo Mode] No valid authentication - using simulated data")
            self._demo_mode = True
            self.rate_limiter = SimpleRateLimiter()
            return False

        except Exception as e:
            print(f"  [Demo Mode] Browser init failed: {e}")
            self._demo_mode = True
            self.rate_limiter = SimpleRateLimiter()
            return False

    async def login_manual(self) -> bool:
        """
        Open a browser window for manual Twitter login and save cookies.

        Returns:
            True if login successful and cookies saved.
        """
        if not PLAYWRIGHT_AVAILABLE:
            print("Error: Playwright is not installed.")
            print("Install it with: pip install playwright && playwright install chromium")
            return False

        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()

            print("\n" + "=" * 60)
            print("Manual Twitter Login")
            print("=" * 60)
            print("\n1. A browser window will open to Twitter")
            print("2. Log in to your Twitter account")
            print("3. Once logged in, press Enter here to save cookies")
            print("=" * 60 + "\n")

            await page.goto("https://twitter.com/login")

            input("Press Enter after you've logged in to Twitter...")

            # Save cookies
            cookies = await context.cookies()
            self.cookies_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cookies_path, 'w') as f:
                json.dump(cookies, f)

            print(f"\nCookies saved to {self.cookies_path}")

            await browser.close()
            await playwright.stop()

            return True

        except Exception as e:
            print(f"Login failed: {e}")
            return False

    async def get_user_profile(
        self,
        username: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[UserProfile]:
        """
        Get a Twitter user's profile information.

        Args:
            username: Twitter username (without @)
            progress_callback: Optional callback for progress updates

        Returns:
            UserProfile or None if user not found
        """
        if self.demo_mode:
            return self._get_demo_profile(username)

        try:
            await self.rate_limiter.wait()

            if progress_callback:
                progress_callback(f"Fetching profile for @{username}")

            await self.page.goto(
                f"https://twitter.com/{username}",
                timeout=self.timeout
            )
            await asyncio.sleep(2)

            # Extract profile data using JavaScript
            profile_data = await self.page.evaluate("""
                () => {
                    const displayName = document.querySelector('[data-testid="UserName"]')?.innerText?.split('\\n')[0] || '';
                    const bio = document.querySelector('[data-testid="UserDescription"]')?.innerText || '';
                    const stats = document.querySelectorAll('[data-testid="primaryColumn"] a[href*="/following"], [data-testid="primaryColumn"] a[href*="/verified_followers"]');

                    let followers = 0, following = 0;
                    stats.forEach(stat => {
                        const text = stat.innerText.toLowerCase();
                        const num = parseInt(text.replace(/[^0-9]/g, '')) || 0;
                        if (text.includes('following')) following = num;
                        if (text.includes('follower')) followers = num;
                    });

                    return {displayName, bio, followers, following};
                }
            """)

            return UserProfile(
                username=username,
                display_name=profile_data.get('displayName', username),
                bio=profile_data.get('bio', ''),
                follower_count=profile_data.get('followers', 0),
                following_count=profile_data.get('following', 0),
                tweet_count=0,  # Hard to get accurately
                joined_date="",
                verified=False
            )

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching profile: {e}")
            return self._get_demo_profile(username)

    async def get_user_tweets(
        self,
        username: str,
        max_tweets: int = 200,
        include_replies: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Tweet]:
        """
        Get a user's tweets.

        Args:
            username: Twitter username (without @)
            max_tweets: Maximum number of tweets to fetch
            include_replies: Whether to include replies
            progress_callback: Optional callback for progress updates

        Returns:
            List of Tweet objects
        """
        if self.demo_mode:
            return self._get_demo_tweets(username, max_tweets, progress_callback)

        tweets = []
        try:
            url = f"https://twitter.com/{username}"
            if not include_replies:
                url += "/with_replies"  # Counterintuitive but this excludes replies

            await self.page.goto(url, timeout=self.timeout)
            await asyncio.sleep(3)

            last_count = 0
            stale_count = 0

            while len(tweets) < max_tweets and stale_count < 5:
                # Extract tweets from current view
                new_tweets = await self._extract_tweets_from_page()

                for tweet in new_tweets:
                    if tweet.id not in [t.id for t in tweets]:
                        tweets.append(tweet)
                        if progress_callback:
                            progress_callback(f"Collected: {len(tweets)} tweets")

                # Check for progress
                if len(tweets) == last_count:
                    stale_count += 1
                else:
                    stale_count = 0
                last_count = len(tweets)

                # Scroll down
                await self.rate_limiter.wait()
                await self.page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(2)

            return tweets[:max_tweets]

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error fetching tweets: {e}")
            if not tweets:
                return self._get_demo_tweets(username, max_tweets, progress_callback)
            return tweets

    async def _extract_tweets_from_page(self) -> List[Tweet]:
        """Extract tweets from the current page state."""
        tweets_data = await self.page.evaluate("""
            () => {
                const tweets = [];
                document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => {
                    try {
                        const text = tweet.querySelector('[data-testid="tweetText"]')?.innerText || '';
                        const time = tweet.querySelector('time')?.getAttribute('datetime') || '';
                        const link = tweet.querySelector('a[href*="/status/"]')?.href || '';
                        const id = link.split('/status/')[1]?.split('?')[0] || '';

                        const metrics = tweet.querySelectorAll('[data-testid$="count"]');
                        let likes = 0, retweets = 0, replies = 0;

                        metrics.forEach(m => {
                            const val = parseInt(m.innerText.replace(/[^0-9]/g, '')) || 0;
                            const testId = m.getAttribute('data-testid') || '';
                            if (testId.includes('like')) likes = val;
                            if (testId.includes('retweet')) retweets = val;
                            if (testId.includes('reply')) replies = val;
                        });

                        const hasMedia = tweet.querySelector('[data-testid="tweetPhoto"]') !== null;
                        const hasVideo = tweet.querySelector('[data-testid="videoPlayer"]') !== null;
                        const isQuote = tweet.querySelector('[data-testid="quoteTweet"]') !== null;

                        if (id && text) {
                            tweets.push({id, text, timestamp: time, likes, retweets, replies, hasMedia, hasVideo, isQuote});
                        }
                    } catch (e) {}
                });
                return tweets;
            }
        """)

        return [
            Tweet(
                id=t['id'],
                text=t['text'],
                timestamp=t['timestamp'],
                likes=t['likes'],
                retweets=t['retweets'],
                replies=t['replies'],
                has_media=t['hasMedia'],
                has_video=t['hasVideo'],
                is_quote_tweet=t['isQuote']
            )
            for t in tweets_data
        ]

    async def close(self):
        """Close the browser and clean up."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None

    # =========================================================================
    # Demo Mode Data Generation
    # =========================================================================

    def _get_demo_profile(self, username: str) -> UserProfile:
        """Get a demo profile for testing."""
        profiles = self._get_demo_profiles()

        if username.lower() in profiles:
            return profiles[username.lower()]

        # Generate random profile for unknown users
        return UserProfile(
            username=username,
            display_name=f"{username.title()} (Demo)",
            bio="Crypto enthusiast | Not financial advice | DYOR",
            follower_count=random.randint(5000, 500000),
            following_count=random.randint(100, 2000),
            tweet_count=random.randint(1000, 50000),
            joined_date="2020-01-15",
            verified=random.random() > 0.7
        )

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
            "hsaka_": UserProfile(
                username="hsaka_",
                display_name="Hsaka",
                bio="Full-time trader | Sharing alpha | Join my telegram for calls",
                follower_count=180000,
                following_count=560,
                tweet_count=35000,
                joined_date="2020-11-05",
                verified=False
            ),
            "cryptokaleo": UserProfile(
                username="CryptoKaleo",
                display_name="K A L E O",
                bio="Charts and vibes | $BTC maxi adjacent | Not your financial advisor",
                follower_count=520000,
                following_count=380,
                tweet_count=52000,
                joined_date="2019-02-28",
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

        # Use seeded random for consistency per username
        random.seed(hash(username.lower()))

        base_time = datetime.now()

        for i in range(max_tweets):
            # Pick a template and customize it
            template = random.choice(tweet_templates)

            # Generate engagement based on profile size
            profiles = self._get_demo_profiles()
            profile = profiles.get(username.lower(), self._get_demo_profile(username))

            base_likes = profile.follower_count * random.uniform(0.01, 0.08)
            variance = random.uniform(0.3, 3.0)  # High variance for realism

            likes = int(base_likes * variance)
            retweets = int(likes * random.uniform(0.05, 0.25))
            replies = int(likes * random.uniform(0.02, 0.15))

            # Calculate timestamp (tweets spread over ~90 days)
            hours_ago = random.uniform(0.5, 90 * 24)
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

            if progress_callback and i % 20 == 0:
                progress_callback(f"Collected: {len(tweets)} tweets")

        # Reset random seed
        random.seed()

        if progress_callback:
            progress_callback(f"Collected: {len(tweets)} tweets (demo mode)")

        return tweets

    def _get_demo_tweet_templates(self, username: str) -> List[Dict]:
        """Get tweet templates based on username persona."""

        # Default templates with variety of patterns
        default_templates = [
            # Bullish tweets
            {"text": "$BTC looking incredibly bullish here. This is the accumulation zone everyone will wish they bought. Not financial advice but I'm loading up.", "has_media": True},
            {"text": "$SOL about to absolutely send it. The ecosystem growth is undeniable. 100x from here is programmed.", "has_media": False},
            {"text": "If you're not accumulating $ETH at these levels, you're going to regret it. This is generational wealth territory.", "has_media": False},
            {"text": "Just went 10x long on $BTC. This is the play. Watch this space.", "has_media": True},
            {"text": "$ARB is the most undervalued L2 play right now. Easy 5x from here. Mark this tweet.", "has_media": False},

            # Bearish tweets (for consistency tracking)
            {"text": "Getting concerned about $BTC here. The chart doesn't look great. Reducing exposure.", "has_media": True},
            {"text": "I've been wrong about $SOL. The network issues are a real problem. Exiting my position.", "has_media": False},
            {"text": "Market structure looking weak. Time to derisk. Cash is a position too.", "has_media": False},

            # Engagement bait patterns
            {"text": "What's your highest conviction play right now? Drop it below 👇", "has_media": False},
            {"text": "Like if you're bullish on crypto in 2024. RT if you think we hit new ATH this year.", "has_media": False},
            {"text": "Name your favorite memecoin. Wrong answers only 😂", "has_media": False},
            {"text": "Follow for more alpha. I'm giving away 1 ETH to someone who RTs this.", "has_media": True},

            # FOMO patterns
            {"text": "INSIDER INFO: This token is about to pump. Last chance to get in before 100x. Don't say I didn't warn you.", "has_media": False},
            {"text": "The alpha I'm about to drop will change your life. But first, follow me and RT for access.", "has_media": False},
            {"text": "This is your FINAL opportunity to get into $XYZ before it moons. Not financial advice but I'm all in.", "has_media": False},

            # Derisive/mocking tweets
            {"text": "NGMI if you sold the bottom. Literally couldn't be me. 🤡", "has_media": False},
            {"text": "Have fun staying poor. Some of y'all really don't deserve to make it.", "has_media": False},
            {"text": "Imagine selling $BTC below 50k. Skill issue tbh.", "has_media": False},
            {"text": "Paper hands getting shaken out as usual. Weak.", "has_media": False},

            # Instructional/helpful tweets
            {"text": "Here's a thread on how to identify rug pulls before they happen. Save this. 🧵", "has_media": False},
            {"text": "Pro tip for beginners: Never invest more than you can afford to lose. DYOR always.", "has_media": False},
            {"text": "Let me explain how to read order flow. This changed my trading completely.", "has_media": True},
            {"text": "Step by step guide to setting up a hardware wallet. Security first, always. NFA", "has_media": True},

            # Kaito/reward gaming
            {"text": "Bullish on $KAITO. Yapping to earn is the future. Make sure to engage for points!", "has_media": False},
            {"text": "Drop your @Kaito_ai yaps below. Let's farm some points together!", "has_media": False},
            {"text": "Complete the Galxe quest for this project. Free airdrop incoming.", "has_media": True},

            # Humble brag patterns
            {"text": "Down only 20% this month. Could be worse I guess. Still up 500% on the year though.", "has_media": False},
            {"text": "Accidentally made 50 ETH on a random memecoin. Just got lucky I suppose.", "has_media": False},
            {"text": "People asking how I caught the bottom. Just experience I guess. Been doing this since 2017.", "has_media": False},

            # Position flip acknowledgment
            {"text": "Flipping bearish on $ETH for now. I was wrong about the merge pump. Adjusting my thesis.", "has_media": False},
            {"text": "Did a complete 180 on $DOGE. Sometimes you have to admit when you're wrong.", "has_media": False},
            {"text": "Changing my stance on memecoins. I was too dismissive before. Learning and adapting.", "has_media": False},

            # Generic tweets
            {"text": "GM everyone. Let's have a great day in the markets.", "has_media": False},
            {"text": "Crypto Twitter is wild today. The timeline is chaotic.", "has_media": False},
            {"text": "Taking a break from charts. Touch grass, anon.", "has_media": True},
            {"text": "New week, new opportunities. Stay focused.", "has_media": False},
        ]

        # Persona-specific templates
        persona_templates = {
            "minhxdynasty": [
                {"text": "$PEPE still has legs. Memecoin season isn't over. Adding here.", "has_media": False},
                {"text": "Just aped into another Solana memecoin. This one feels different.", "has_media": False},
                {"text": "The CT meta is shifting. Pay attention to what's being shilled.", "has_media": False},
                {"text": "My portfolio is 90% memecoins and I'm not even sorry.", "has_media": False},
                {"text": "If you're not farming airdrops right now, what are you even doing?", "has_media": False},
                {"text": "New Kaito yap just dropped. Engage for points!", "has_media": False},
            ],
            "cobie": [
                {"text": "The best time to buy was yesterday. The second best time is now. Or something.", "has_media": False},
                {"text": "Market structure looks interesting here. Not saying it's the bottom but... it's interesting.", "has_media": True},
                {"text": "Remember when everyone was bearish? I remember.", "has_media": False},
                {"text": "The narratives shift but the game stays the same.", "has_media": False},
                {"text": "Sometimes the best trade is no trade.", "has_media": False},
            ],
            "zachxbt": [
                {"text": "Investigation thread incoming on a major influencer scam. Stay tuned.", "has_media": False},
                {"text": "This project's wallet movements are extremely suspicious. Thread 🧵", "has_media": True},
                {"text": "Another day, another rug. Please do your research before aping.", "has_media": False},
                {"text": "The on-chain data doesn't lie. Here's what I found.", "has_media": True},
                {"text": "Reminder: Most crypto influencers are paid promoters. Trust but verify.", "has_media": False},
            ],
        }

        templates = default_templates.copy()
        if username in persona_templates:
            templates.extend(persona_templates[username])

        return templates
