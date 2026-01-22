import tweepy
import os
from typing import List, Dict, Any, Optional
from loguru import logger
import time
from datetime import datetime, timezone, timedelta


class XClient:
    def __init__(self):
        self.api_key = os.getenv("X_API_KEY")
        self.api_secret = os.getenv("X_API_SECRET")
        self.access_token = os.getenv("X_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("X_ACCESS_TOKEN_SECRET")
        self.bearer_token = os.getenv("X_BEARER_TOKEN")
        
        if not all([self.api_key, self.api_secret, self.access_token, 
                   self.access_token_secret, self.bearer_token]):
            raise ValueError("Missing X API credentials in environment variables")
        
        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        logger.info("X API client initialized successfully")
    
    def test_api_permissions(self) -> bool:
        """Test if the API credentials have the necessary permissions"""
        try:
            # Test basic read access
            me = self.client.get_me()
            if me.data:
                logger.info(f"Authenticated as: @{me.data.username} ({me.data.name})")
                return True
            else:
                logger.error("Could not retrieve authenticated user info")
                return False
        except Exception as e:
            logger.error(f"Error testing API permissions: {str(e)}")
            return False
    
    def get_user_recent_tweets(self, username: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweets from a specific user"""
        try:
            # Get user by username
            user = self.client.get_user(username=username)
            if not user.data:
                logger.warning(f"User {username} not found")
                return []
            
            user_id = user.data.id
            
            # Ensure count is within API limits (5-100)
            api_count = max(5, min(count, 100))
            
            # Get recent tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=api_count,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'conversation_id']
            )
            
            if not tweets.data:
                logger.info(f"No recent tweets found for {username}")
                return []
            
            tweet_list = []
            for tweet in tweets.data:
                tweet_data = {
                    'id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_username': username,
                    'conversation_id': tweet.conversation_id,
                    'public_metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}
                }
                tweet_list.append(tweet_data)
            
            logger.info(f"Retrieved {len(tweet_list)} tweets from {username}")
            return tweet_list
            
        except Exception as e:
            logger.error(f"Error getting tweets for {username}: {str(e)}")
            return []
    
    def reply_to_tweet(self, tweet_id: str, reply_text: str, dry_run: bool = True) -> bool:
        """Reply to a specific tweet"""
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would reply to tweet {tweet_id}: {reply_text}")
                return True
            
            response = self.client.create_tweet(
                text=reply_text,
                in_reply_to_tweet_id=tweet_id
            )
            
            if response.data:
                logger.success(f"Successfully replied to tweet {tweet_id}")
                return True
            else:
                logger.error(f"Failed to reply to tweet {tweet_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error replying to tweet {tweet_id}: {str(e)}")
            return False
    
    def get_tweet_context(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get additional context for a tweet (replies, mentions, etc.)"""
        try:
            tweet = self.client.get_tweet(
                id=tweet_id,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'conversation_id'],
                expansions=['author_id']
            )
            
            if tweet.data:
                return {
                    'id': tweet.data.id,
                    'text': tweet.data.text,
                    'created_at': tweet.data.created_at,
                    'conversation_id': tweet.data.conversation_id,
                    'public_metrics': tweet.data.public_metrics if hasattr(tweet.data, 'public_metrics') else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting tweet context for {tweet_id}: {str(e)}")
            return None
    
    def check_if_already_replied(self, tweet_id: str, bot_username: str) -> bool:
        """Check if the bot has already replied to this tweet"""
        try:
            # Get replies to the tweet by searching for replies from our bot
            search_query = f"conversation_id:{tweet_id} from:{bot_username}"
            
            tweets = self.client.search_recent_tweets(
                query=search_query,
                max_results=10
            )
            
            return bool(tweets.data and len(tweets.data) > 0)
            
        except Exception as e:
            logger.error(f"Error checking if already replied to {tweet_id}: {str(e)}")
            return True  # Assume already replied to be safe
    
    def post_tweet(self, tweet_text: str, dry_run: bool = True) -> bool:
        """Post an original tweet"""
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would post tweet: {tweet_text}")
                return True

            response = self.client.create_tweet(text=tweet_text)

            if response.data:
                logger.success(f"Successfully posted tweet: {tweet_text}")
                return True
            else:
                logger.error(f"Failed to post tweet: {tweet_text}")
                return False

        except Exception as e:
            logger.error(f"Error posting tweet: {str(e)}")
            return False

    def search_mentions(self, username: str, max_results: int = 50, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Search for tweets mentioning a specific user.

        This finds what others are saying ABOUT this person - useful for
        reputation analysis and community sentiment.

        Args:
            username: The username to search mentions for (without @)
            max_results: Maximum tweets to return (10-100)
            days_back: How many days back to search (max 7 for recent search)

        Returns:
            List of tweets mentioning the user
        """
        try:
            username = username.lstrip('@').lower()

            # Build search query - mentions of the user, excluding their own tweets
            query = f"@{username} -from:{username} -is:retweet"

            # Calculate start time (X API allows max 7 days for recent search)
            start_time = datetime.now(timezone.utc) - timedelta(days=min(days_back, 7))

            # Ensure max_results is within API limits
            max_results = max(10, min(max_results, 100))

            logger.info(f"Searching mentions of @{username} (last {days_back} days)...")

            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'conversation_id'],
                expansions=['author_id'],
                user_fields=['username', 'name', 'verified', 'public_metrics']
            )

            if not tweets.data:
                logger.info(f"No mentions found for @{username}")
                return []

            # Build user lookup from includes
            users_lookup = {}
            if tweets.includes and 'users' in tweets.includes:
                for user in tweets.includes['users']:
                    users_lookup[user.id] = {
                        'username': user.username,
                        'name': user.name,
                        'verified': getattr(user, 'verified', False),
                        'followers': user.public_metrics.get('followers_count', 0) if hasattr(user, 'public_metrics') else 0
                    }

            mentions = []
            for tweet in tweets.data:
                author_info = users_lookup.get(tweet.author_id, {})
                metrics = tweet.public_metrics if hasattr(tweet, 'public_metrics') else {}

                mention_data = {
                    'id': str(tweet.id),
                    'text': tweet.text,
                    'created_at': str(tweet.created_at) if tweet.created_at else None,
                    'author_id': str(tweet.author_id),
                    'author_username': author_info.get('username', 'unknown'),
                    'author_name': author_info.get('name', 'Unknown'),
                    'author_verified': author_info.get('verified', False),
                    'author_followers': author_info.get('followers', 0),
                    'likes': metrics.get('like_count', 0),
                    'retweets': metrics.get('retweet_count', 0),
                    'replies': metrics.get('reply_count', 0),
                    'engagement': metrics.get('like_count', 0) + metrics.get('retweet_count', 0) + metrics.get('reply_count', 0)
                }
                mentions.append(mention_data)

            # Sort by engagement (most engagement first)
            mentions.sort(key=lambda x: x['engagement'], reverse=True)

            logger.info(f"Found {len(mentions)} mentions of @{username}")
            return mentions

        except Exception as e:
            logger.error(f"Error searching mentions for @{username}: {str(e)}")
            return []

    def search_reputation(self, username: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Comprehensive reputation search - find what the community says about a user.

        Searches for:
        1. Direct mentions (@username)
        2. Discussions about the user (username without @)
        3. Warnings/accusations
        4. Praise/endorsements

        Returns categorized findings for investigation.
        """
        try:
            username = username.lstrip('@').lower()

            result = {
                'username': username,
                'total_mentions': 0,
                'mentions': [],
                'warnings': [],      # Tweets with warning keywords
                'accusations': [],   # Tweets with scam/rug keywords
                'endorsements': [],  # Positive mentions
                'high_profile_mentions': [],  # From accounts with >10k followers
                'sentiment_summary': 'neutral',
                'red_flag_count': 0,
                'trust_signal_count': 0
            }

            # Warning/accusation keywords
            warning_keywords = ['scam', 'scammer', 'rug', 'rugged', 'fraud', 'fake', 'ponzi',
                              'exit liquidity', 'dump', 'dumped', 'avoid', 'warning', 'beware',
                              'don\'t trust', 'dont trust', 'be careful', 'stolen', 'hack']

            # Positive keywords
            positive_keywords = ['legit', 'trusted', 'reliable', 'accurate', 'respect',
                               'goat', 'legend', 'based', 'honest', 'transparent', 'real one']

            # Get mentions
            mentions = self.search_mentions(username, max_results=max_results, days_back=7)
            result['mentions'] = mentions
            result['total_mentions'] = len(mentions)

            for mention in mentions:
                text_lower = mention['text'].lower()

                # Check for warnings/accusations
                has_warning = any(kw in text_lower for kw in warning_keywords)
                has_positive = any(kw in text_lower for kw in positive_keywords)

                if has_warning:
                    result['warnings'].append(mention)
                    result['red_flag_count'] += 1
                    # Weight by author followers
                    if mention['author_followers'] > 10000:
                        result['red_flag_count'] += 2  # High-profile warning counts more

                if has_positive:
                    result['endorsements'].append(mention)
                    result['trust_signal_count'] += 1

                # Track high-profile mentions
                if mention['author_followers'] > 10000:
                    result['high_profile_mentions'].append(mention)

            # Determine overall sentiment
            if result['red_flag_count'] > 5:
                result['sentiment_summary'] = 'negative - multiple warnings detected'
            elif result['red_flag_count'] > 2:
                result['sentiment_summary'] = 'mixed - some warnings present'
            elif result['trust_signal_count'] > 3:
                result['sentiment_summary'] = 'positive - community endorsements'
            else:
                result['sentiment_summary'] = 'neutral - no strong signals'

            logger.info(f"Reputation scan for @{username}: {result['sentiment_summary']}")
            return result

        except Exception as e:
            logger.error(f"Error in reputation search for @{username}: {str(e)}")
            return {
                'username': username,
                'error': str(e),
                'total_mentions': 0,
                'mentions': [],
                'warnings': [],
                'accusations': [],
                'endorsements': [],
                'high_profile_mentions': [],
                'sentiment_summary': 'unknown',
                'red_flag_count': 0,
                'trust_signal_count': 0
            }

    def get_community_sentiment(self, username: str) -> Dict[str, Any]:
        """
        Quick community sentiment check - what's the vibe around this person?

        Returns a simplified sentiment analysis for the investigation.
        """
        try:
            rep = self.search_reputation(username, max_results=50)

            findings = []

            # Summarize findings
            if rep['red_flag_count'] > 0:
                findings.append(f"COMMUNITY WARNINGS: Found {rep['red_flag_count']} warning signal(s) in recent mentions")

                # Add specific warning examples
                for warning in rep['warnings'][:3]:
                    author = warning['author_username']
                    followers = warning['author_followers']
                    snippet = warning['text'][:100] + '...' if len(warning['text']) > 100 else warning['text']
                    findings.append(f"  - @{author} ({followers:,} followers): \"{snippet}\"")

            if rep['trust_signal_count'] > 0:
                findings.append(f"COMMUNITY TRUST: Found {rep['trust_signal_count']} positive mention(s)")

                for endorsement in rep['endorsements'][:2]:
                    author = endorsement['author_username']
                    followers = endorsement['author_followers']
                    snippet = endorsement['text'][:100] + '...' if len(endorsement['text']) > 100 else endorsement['text']
                    findings.append(f"  - @{author} ({followers:,} followers): \"{snippet}\"")

            if rep['high_profile_mentions']:
                findings.append(f"HIGH-PROFILE ATTENTION: {len(rep['high_profile_mentions'])} mention(s) from accounts with 10k+ followers")

            if not findings:
                findings.append("NO SIGNIFICANT COMMUNITY SIGNALS: Limited mentions in the past 7 days")

            return {
                'username': username,
                'sentiment': rep['sentiment_summary'],
                'findings': findings,
                'warning_count': rep['red_flag_count'],
                'trust_count': rep['trust_signal_count'],
                'total_mentions': rep['total_mentions'],
                'raw_data': rep
            }

        except Exception as e:
            logger.error(f"Error getting community sentiment for @{username}: {str(e)}")
            return {
                'username': username,
                'sentiment': 'unknown',
                'findings': [f"ERROR: Could not fetch community sentiment - {str(e)}"],
                'warning_count': 0,
                'trust_count': 0,
                'total_mentions': 0
            }
