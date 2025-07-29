import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Set
from dotenv import load_dotenv
from loguru import logger
import schedule
import random

from config import BotConfig, AccountConfig
from x_client import XClient
from gemini_client import GeminiClient
from content_generator import CryptoContentGenerator


class CryptoTwitterBot:
    def __init__(self, config_file: str = "crypto_config.json"):
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.load_config(config_file)
        
        # Initialize clients
        self.x_client = XClient()
        self.gemini_client = GeminiClient()
        self.content_generator = CryptoContentGenerator()
        
        # Track processed tweets to avoid duplicates
        self.processed_tweets: Set[str] = set()
        
        # Rate limiting
        self.replies_this_hour = 0
        self.posts_this_hour = 0
        self.last_hour_reset = datetime.now()
        
        # Content posting schedule
        self.last_content_post = datetime.now() - timedelta(hours=4)  # Allow immediate post
        self.content_post_interval_hours = 3  # Post original content every 3 hours
        
        logger.info(f"Crypto bot initialized with {len(self.config.monitored_accounts)} monitored accounts")
        logger.info(f"Dry run mode: {self.config.dry_run}")
    
    def load_config(self, config_file: str):
        """Load bot configuration from file"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.config = BotConfig(**config_data)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
                raise
        else:
            logger.error(f"Config file {config_file} not found")
            raise FileNotFoundError(f"Configuration file {config_file} is required")
    
    def save_config(self, config_file: str):
        """Save current configuration to file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.dict(), f, indent=2, default=str)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
    
    def reset_rate_limit_if_needed(self):
        """Reset hourly rate limit counters if an hour has passed"""
        now = datetime.now()
        if now - self.last_hour_reset >= timedelta(hours=1):
            self.replies_this_hour = 0
            self.posts_this_hour = 0
            self.last_hour_reset = now
            logger.info("Hourly rate limit counters reset")
    
    def can_reply(self) -> bool:
        """Check if we can send more replies this hour"""
        self.reset_rate_limit_if_needed()
        return self.replies_this_hour < self.config.max_replies_per_hour
    
    def can_post_content(self) -> bool:
        """Check if we can post original content"""
        self.reset_rate_limit_if_needed()
        now = datetime.now()
        
        # Check time-based interval
        time_since_last_post = now - self.last_content_post
        if time_since_last_post < timedelta(hours=self.content_post_interval_hours):
            return False
        
        # Check hourly rate limit (max 2 original posts per hour)
        if self.posts_this_hour >= 2:
            return False
        
        return True
    
    def process_account(self, username: str, account_config: AccountConfig):
        """Process tweets from a specific crypto-focused account"""
        try:
            logger.info(f"Processing crypto account: @{username}")
            
            # Get recent tweets
            tweets = self.x_client.get_user_recent_tweets(username, count=5)
            
            for tweet in tweets:
                tweet_id = tweet['id']
                
                # Skip if already processed
                if tweet_id in self.processed_tweets:
                    continue
                
                # Skip if already replied
                if self.x_client.check_if_already_replied(tweet_id, self.config.bot_username):
                    logger.info(f"Already replied to tweet {tweet_id}")
                    self.processed_tweets.add(tweet_id)
                    continue
                
                # Check if we should reply to this tweet
                if not self.gemini_client.should_reply_to_tweet(tweet, account_config):
                    self.processed_tweets.add(tweet_id)
                    continue
                
                # Check rate limits
                if not self.can_reply():
                    logger.warning(f"Reply rate limit reached ({self.config.max_replies_per_hour}/hour)")
                    break
                
                # Generate crypto-focused reply
                reply_text = self.gemini_client.generate_reply(tweet, account_config)
                
                if reply_text:
                    # Send reply
                    success = self.x_client.reply_to_tweet(
                        tweet_id, 
                        reply_text, 
                        dry_run=self.config.dry_run
                    )
                    
                    if success:
                        self.replies_this_hour += 1
                        logger.success(f"Crypto reply sent to @{username} (tweet {tweet_id}): {reply_text}")
                    else:
                        logger.error(f"Failed to send reply to @{username} (tweet {tweet_id})")
                else:
                    logger.warning(f"No reply generated for tweet {tweet_id}")
                
                # Mark as processed
                self.processed_tweets.add(tweet_id)
                
                # Small delay between replies
                time.sleep(3)
                
        except Exception as e:
            logger.error(f"Error processing crypto account {username}: {str(e)}")
    
    def post_technical_analysis(self):
        """Post original technical analysis content"""
        try:
            if not self.can_post_content():
                return
            
            # Generate technical analysis post
            post = self.content_generator.create_technical_analysis_post("auto")
            
            if post:
                success = self.content_generator.post_content(
                    post, 
                    dry_run=self.config.dry_run
                )
                
                if success:
                    self.posts_this_hour += 1
                    self.last_content_post = datetime.now()
                    logger.success(f"Posted {post.analysis_type} analysis: {post.content}")
                else:
                    logger.error(f"Failed to post {post.analysis_type} analysis")
            else:
                logger.warning("No technical analysis content generated")
                
        except Exception as e:
            logger.error(f"Error posting technical analysis: {str(e)}")
    
    def run_once(self):
        """Run one iteration of the crypto bot"""
        logger.info("Running crypto bot iteration...")
        
        # Post original content first (if due)
        self.post_technical_analysis()
        
        # Process monitored crypto accounts
        for username, account_config in self.config.monitored_accounts.items():
            self.process_account(username, account_config)
            
            # Delay between accounts to be respectful
            time.sleep(random.randint(5, 10))
        
        # Clean up old processed tweets (keep last 1000)
        if len(self.processed_tweets) > 1000:
            sorted_tweets = sorted(list(self.processed_tweets))
            self.processed_tweets = set(sorted_tweets[-1000:])
        
        logger.info("Crypto bot iteration completed")
        logger.info(f"Rate limits: {self.replies_this_hour} replies, {self.posts_this_hour} posts this hour")
    
    def get_status_report(self) -> Dict:
        """Get current bot status"""
        return {
            "monitored_accounts": len(self.config.monitored_accounts),
            "processed_tweets": len(self.processed_tweets),
            "replies_this_hour": self.replies_this_hour,
            "posts_this_hour": self.posts_this_hour,
            "dry_run": self.config.dry_run,
            "next_content_post_in_minutes": max(0, 
                (self.content_post_interval_hours * 60) - 
                int((datetime.now() - self.last_content_post).total_seconds() / 60)
            )
        }
    
    def start(self):
        """Start the crypto bot with scheduled execution"""
        logger.info(f"Starting crypto bot with {self.config.check_interval_minutes} minute intervals")
        logger.info(f"Content posting interval: {self.content_post_interval_hours} hours")
        
        # Schedule the bot to run
        schedule.every(self.config.check_interval_minutes).minutes.do(self.run_once)
        
        # Run once immediately
        self.run_once()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds


def main():
    """Main entry point for crypto bot"""
    try:
        # Setup logging
        logger.add("logs/crypto_bot.log", rotation="1 day", retention="7 days")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize and start crypto bot
        bot = CryptoTwitterBot()
        
        # Print status
        status = bot.get_status_report()
        logger.info(f"Bot status: {status}")
        
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Crypto bot stopped by user")
    except Exception as e:
        logger.error(f"Crypto bot crashed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
