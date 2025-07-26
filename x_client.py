import tweepy
import os
from typing import List, Optional, Dict, Any
from loguru import logger
from datetime import datetime, timezone


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
    
    def get_user_recent_tweets(self, username: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweets from a specific user"""
        try:
            # Get user by username
            user = self.client.get_user(username=username)
            if not user.data:
                logger.warning(f"User {username} not found")
                return []
            
            user_id = user.data.id
            
            # Get recent tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=count,
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
