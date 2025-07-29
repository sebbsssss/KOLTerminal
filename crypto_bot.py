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
    
    def enhance_reply_prompt_for_crypto(self, base_prompt: str, username: str) -> str:
        """Enhance the reply prompt with crypto-specific context"""
        crypto_context = f\"\"\"
        
CRYPTO CONTEXT FOR @{username}:
- This is a crypto/web3 focused conversation
- Use appropriate crypto terminology and culture references
- Focus on technical analysis, market trends, and blockchain technology
- Memecoin discussions should be analytical, not speculative
- Web3 and DeFi topics should showcase technical understanding
- BTC/ETH analysis should connect to broader market implications
        
CRYPTO LANGUAGE STYLE:
- Natural use of: \"gm\", \"wagmi\", \"based\", \"this is the way\", \"diamond hands\", \"lfg\"
- Technical terms: \"liquidity\", \"tokenomics\", \"yield farming\", \"TVL\", \"market cap\"
- Analysis terms: \"resistance\", \"support\", \"breakout\", \"consolidation\", \"volume profile\"
        \"\"\"
        
        return base_prompt + crypto_context
    
    def process_account(self, username: str, account_config: AccountConfig):
        """Process tweets from a specific crypto-focused account"""
        try:
            logger.info(f\"Processing crypto account: @{username}\")
            
            # Get recent tweets
            tweets = self.x_client.get_user_recent_tweets(username, count=5)
            
            for tweet in tweets:
                tweet_id = tweet['id']
                
                # Skip if already processed
                if tweet_id in self.processed_tweets:
                    continue
                
                # Skip if already replied
                if self.x_client.check_if_already_replied(tweet_id, self.config.bot_username):
                    logger.info(f\"Already replied to tweet {tweet_id}\")
                    self.processed_tweets.add(tweet_id)
                    continue
                
                # Check if we should reply to this tweet
                if not self.gemini_client.should_reply_to_tweet(tweet, account_config):
                    self.processed_tweets.add(tweet_id)
                    continue
                
                # Check rate limits
                if not self.can_reply():
                    logger.warning(f\"Reply rate limit reached ({self.config.max_replies_per_hour}/hour)\")
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
                        logger.success(f\"Crypto reply sent to @{username} (tweet {tweet_id}): {reply_text}\")\n                    else:\n                        logger.error(f\"Failed to send reply to @{username} (tweet {tweet_id})\")\n                else:\n                    logger.warning(f\"No reply generated for tweet {tweet_id}\")\n                \n                # Mark as processed\n                self.processed_tweets.add(tweet_id)\n                \n                # Small delay between replies\n                time.sleep(3)\n                \n        except Exception as e:\n            logger.error(f\"Error processing crypto account {username}: {str(e)}\")\n    \n    def post_technical_analysis(self):\n        \"\"\"Post original technical analysis content\"\"\"\n        try:\n            if not self.can_post_content():\n                return\n            \n            # Generate technical analysis post\n            post = self.content_generator.create_technical_analysis_post(\"auto\")\n            \n            if post:\n                success = self.content_generator.post_content(\n                    post, \n                    dry_run=self.config.dry_run\n                )\n                \n                if success:\n                    self.posts_this_hour += 1\n                    self.last_content_post = datetime.now()\n                    logger.success(f\"Posted {post.analysis_type} analysis: {post.content}\")\n                else:\n                    logger.error(f\"Failed to post {post.analysis_type} analysis\")\n            else:\n                logger.warning(\"No technical analysis content generated\")\n                \n        except Exception as e:\n            logger.error(f\"Error posting technical analysis: {str(e)}\")\n    \n    def run_once(self):\n        \"\"\"Run one iteration of the crypto bot\"\"\"\n        logger.info(\"Running crypto bot iteration...\")\n        \n        # Post original content first (if due)\n        self.post_technical_analysis()\n        \n        # Process monitored crypto accounts\n        for username, account_config in self.config.monitored_accounts.items():\n            self.process_account(username, account_config)\n            \n            # Delay between accounts to be respectful\n            time.sleep(random.randint(5, 10))\n        \n        # Clean up old processed tweets (keep last 1000)\n        if len(self.processed_tweets) > 1000:\n            sorted_tweets = sorted(list(self.processed_tweets))\n            self.processed_tweets = set(sorted_tweets[-1000:])\n        \n        logger.info(\"Crypto bot iteration completed\")\n        logger.info(f\"Rate limits: {self.replies_this_hour} replies, {self.posts_this_hour} posts this hour\")\n    \n    def get_status_report(self) -> Dict:\n        \"\"\"Get current bot status\"\"\"\n        return {\n            \"monitored_accounts\": len(self.config.monitored_accounts),\n            \"processed_tweets\": len(self.processed_tweets),\n            \"replies_this_hour\": self.replies_this_hour,\n            \"posts_this_hour\": self.posts_this_hour,\n            \"dry_run\": self.config.dry_run,\n            \"next_content_post_in_minutes\": max(0, \n                (self.content_post_interval_hours * 60) - \n                int((datetime.now() - self.last_content_post).total_seconds() / 60)\n            )\n        }\n    \n    def start(self):\n        \"\"\"Start the crypto bot with scheduled execution\"\"\"\n        logger.info(f\"Starting crypto bot with {self.config.check_interval_minutes} minute intervals\")\n        logger.info(f\"Content posting interval: {self.content_post_interval_hours} hours\")\n        \n        # Schedule the bot to run\n        schedule.every(self.config.check_interval_minutes).minutes.do(self.run_once)\n        \n        # Run once immediately\n        self.run_once()\n        \n        # Keep running\n        while True:\n            schedule.run_pending()\n            time.sleep(30)  # Check every 30 seconds\n\n\ndef main():\n    \"\"\"Main entry point for crypto bot\"\"\"\n    try:\n        # Setup logging\n        logger.add(\"logs/crypto_bot.log\", rotation=\"1 day\", retention=\"7 days\")\n        \n        # Create logs directory if it doesn't exist\n        os.makedirs(\"logs\", exist_ok=True)\n        \n        # Initialize and start crypto bot\n        bot = CryptoTwitterBot()\n        \n        # Print status\n        status = bot.get_status_report()\n        logger.info(f\"Bot status: {status}\")\n        \n        bot.start()\n        \n    except KeyboardInterrupt:\n        logger.info(\"Crypto bot stopped by user\")\n    except Exception as e:\n        logger.error(f\"Crypto bot crashed: {str(e)}\")\n        raise\n\n\nif __name__ == \"__main__\":\n    main()
