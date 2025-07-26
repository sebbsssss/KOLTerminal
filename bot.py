import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Set
from dotenv import load_dotenv
from loguru import logger
import schedule

from config import BotConfig, AccountConfig
from x_client import XClient
from gemini_client import GeminiClient


class TwitterReplyBot:
    def __init__(self, config_file: str = "bot_config.json"):
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.load_config(config_file)
        
        # Initialize clients
        self.x_client = XClient()
        self.gemini_client = GeminiClient()
        
        # Track processed tweets to avoid duplicates
        self.processed_tweets: Set[str] = set()
        
        # Rate limiting
        self.replies_this_hour = 0
        self.last_hour_reset = datetime.now()
        
        logger.info(f"Bot initialized with {len(self.config.monitored_accounts)} monitored accounts")
        logger.info(f"Dry run mode: {self.config.dry_run}")
    
    def load_config(self, config_file: str):
        """Load bot configuration from file or use example"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.config = BotConfig(**config_data)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
                logger.info("Using example configuration")
                self.config = BotConfig.load_example()
        else:
            logger.warning(f"Config file {config_file} not found, using example configuration")
            self.config = BotConfig.load_example()
            self.save_config(config_file)
    
    def save_config(self, config_file: str):
        """Save current configuration to file"""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config.dict(), f, indent=2, default=str)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
    
    def reset_rate_limit_if_needed(self):
        """Reset hourly rate limit counter if an hour has passed"""
        now = datetime.now()
        if now - self.last_hour_reset >= timedelta(hours=1):
            self.replies_this_hour = 0
            self.last_hour_reset = now
            logger.info("Hourly rate limit counter reset")
    
    def can_reply(self) -> bool:
        """Check if we can send more replies this hour"""
        self.reset_rate_limit_if_needed()
        return self.replies_this_hour < self.config.max_replies_per_hour
    
    def process_account(self, username: str, account_config: AccountConfig):
        """Process tweets from a specific account"""
        try:
            logger.info(f"Processing account: @{username}")
            
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
                    logger.warning(f"Rate limit reached ({self.config.max_replies_per_hour}/hour)")
                    break
                
                # Generate reply
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
                        logger.success(f"Reply sent to @{username} (tweet {tweet_id})")
                    else:
                        logger.error(f"Failed to send reply to @{username} (tweet {tweet_id})")
                else:
                    logger.warning(f"No reply generated for tweet {tweet_id}")
                
                # Mark as processed
                self.processed_tweets.add(tweet_id)
                
                # Small delay between replies
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error processing account {username}: {str(e)}")
    
    def run_once(self):
        """Run one iteration of the bot"""
        logger.info("Running bot iteration...")
        
        for username, account_config in self.config.monitored_accounts.items():
            self.process_account(username, account_config)
            
            # Delay between accounts to be respectful
            time.sleep(5)
        
        # Clean up old processed tweets (keep last 1000)
        if len(self.processed_tweets) > 1000:
            # Convert to list, sort, and keep the last 1000
            sorted_tweets = sorted(list(self.processed_tweets))
            self.processed_tweets = set(sorted_tweets[-1000:])
        
        logger.info("Bot iteration completed")
    
    def start(self):
        """Start the bot with scheduled execution"""
        logger.info(f"Starting bot with {self.config.check_interval_minutes} minute intervals")
        
        # Schedule the bot to run
        schedule.every(self.config.check_interval_minutes).minutes.do(self.run_once)
        
        # Run once immediately
        self.run_once()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds


def main():
    """Main entry point"""
    try:
        # Setup logging
        logger.add("logs/bot.log", rotation="1 day", retention="7 days")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Initialize and start bot
        bot = TwitterReplyBot()
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
