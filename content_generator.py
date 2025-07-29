import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import google.generativeai as genai
from x_client import XClient


@dataclass
class TrendingToken:
    symbol: str
    name: str
    price_change_24h: float
    volume_24h: float
    market_cap: Optional[float] = None
    social_mentions: int = 0
    momentum_score: float = 0.0


@dataclass
class TechnicalAnalysisPost:
    content: str
    tokens_mentioned: List[str]
    analysis_type: str  # "memecoin", "defi", "microfinance", "btc_eth"
    timestamp: datetime


class CryptoContentGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.x_client = XClient()
        
        # Track recent posts to avoid repetition
        self.recent_posts = []
        self.max_recent_posts = 50
        
        logger.info("Crypto content generator initialized")
    
    def get_trending_tokens(self) -> List[TrendingToken]:
        """Get trending tokens data - would integrate with CoinGecko/DexScreener in production"""
        # Mock trending data - replace with real API calls
        mock_trending = [
            TrendingToken("PEPE", "Pepe", 15.5, 50000000, 2000000000, 1250, 85.0),
            TrendingToken("BONK", "Bonk", 8.2, 25000000, 800000000, 980, 72.0),
            TrendingToken("WIF", "Dogwifhat", 22.1, 35000000, 1500000000, 1100, 78.5),
            TrendingToken("SHIB", "Shiba Inu", 3.2, 180000000, 15000000000, 2200, 65.0),
            TrendingToken("FLOKI", "Floki", 12.8, 45000000, 1200000000, 750, 68.5),
        ]
        
        # Sort by momentum score (combination of price change, volume, social mentions)
        return sorted(mock_trending, key=lambda x: x.momentum_score, reverse=True)
    
    def generate_memecoin_analysis(self, trending_tokens: List[TrendingToken]) -> str:
        """Generate technical analysis focused on memecoin trends"""
        top_tokens = trending_tokens[:3]
        
        prompt = f"""Create a technical analysis thread starter (single tweet, max 280 chars) about the current memecoin market trends. 

Current trending memecoins:
{chr(10).join([f"- {token.symbol} ({token.name}): +{token.price_change_24h}% (24h), Volume: ${token.volume_24h:,.0f}, Social: {token.social_mentions} mentions" for token in top_tokens])}

Requirements:
- Focus on technical patterns, not financial advice
- Mention specific tokens but analyze broader trends
- Use professional crypto terminology
- Include micro finance angle (how memecoins are becoming micro investment vehicles)
- Be analytical but accessible
- No hashtags, no quotes around response
- Sound like an experienced trader sharing insights
- Keep it concise and engaging

Write a single tweet that provides valuable technical analysis on the memecoin space:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                content = response.text.strip()
                if len(content) > 280:
                    content = content[:277] + "..."
                return content
        except Exception as e:
            logger.error(f"Error generating memecoin analysis: {str(e)}")
            return None
    
    def generate_defi_microfinance_analysis(self) -> str:
        """Generate analysis on DeFi micro finance trends"""
        prompt = """Create a technical analysis tweet (max 280 chars) about emerging micro finance trends in DeFi and web3.

Focus areas:
- Micro lending protocols
- Small-cap token yield farming opportunities  
- Fractional NFT finance
- Cross-chain micro transactions
- Decentralized peer-to-peer lending
- Micro investment DAOs

Requirements:
- Technical but accessible language
- Mention specific trends without financial advice
- Professional crypto trader perspective
- No hashtags, no quotes
- Focus on innovation and technical developments
- Highlight opportunities in the micro finance space

Write a single insightful tweet:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                content = response.text.strip()
                if len(content) > 280:
                    content = content[:277] + "..."
                return content
        except Exception as e:
            logger.error(f"Error generating DeFi analysis: {str(e)}")
            return None
    
    def generate_btc_eth_analysis(self) -> str:
        """Generate BTC/ETH technical analysis with memecoin context"""
        prompt = """Create a technical analysis tweet (max 280 chars) about BTC and ETH market structure, with context for how it affects the memecoin/altcoin market.

Focus on:
- BTC dominance trends
- ETH ecosystem developments  
- How major moves affect memecoin liquidity
- Technical levels and market structure
- Risk-on vs risk-off sentiment
- Correlation patterns

Requirements:
- Professional technical analysis language
- Connect macro to micro (how BTC/ETH moves affect smaller caps)
- No financial advice disclaimers
- No hashtags, no quotes
- Sound like an experienced trader
- Include memecoin/micro finance angle

Write a single analytical tweet:"""

        try:
            response = self.model.generate_content(prompt)
            if response.text:
                content = response.text.strip()
                if len(content) > 280:
                    content = content[:277] + "..."
                return content
        except Exception as e:
            logger.error(f"Error generating BTC/ETH analysis: {str(e)}")
            return None
    
    def create_technical_analysis_post(self, analysis_type: str = "auto") -> Optional[TechnicalAnalysisPost]:
        """Create a technical analysis post based on current market conditions"""
        try:
            content = None
            tokens_mentioned = []
            
            if analysis_type == "auto":
                # Rotate between different types of analysis
                analysis_types = ["memecoin", "defi", "btc_eth"]
                analysis_type = analysis_types[len(self.recent_posts) % len(analysis_types)]
            
            if analysis_type == "memecoin":
                trending_tokens = self.get_trending_tokens()
                content = self.generate_memecoin_analysis(trending_tokens)
                tokens_mentioned = [token.symbol for token in trending_tokens[:3]]
                
            elif analysis_type == "defi":
                content = self.generate_defi_microfinance_analysis()
                
            elif analysis_type == "btc_eth":
                content = self.generate_btc_eth_analysis()
                tokens_mentioned = ["BTC", "ETH"]
            
            if content:
                post = TechnicalAnalysisPost(
                    content=content,
                    tokens_mentioned=tokens_mentioned,
                    analysis_type=analysis_type,
                    timestamp=datetime.now()
                )
                
                # Add to recent posts
                self.recent_posts.append(post)
                if len(self.recent_posts) > self.max_recent_posts:
                    self.recent_posts.pop(0)
                
                return post
            
        except Exception as e:
            logger.error(f"Error creating technical analysis post: {str(e)}")
            return None
    
    def post_content(self, post: TechnicalAnalysisPost, dry_run: bool = True) -> bool:
        """Post the technical analysis content to Twitter"""
        try:
            logger.info(f"Posting {post.analysis_type} analysis: {post.content}")
            
            success = self.x_client.post_tweet(post.content, dry_run=dry_run)
            
            if success:
                logger.success(f"Successfully posted {post.analysis_type} analysis")
                return True
            else:
                logger.error(f"Failed to post {post.analysis_type} analysis")
                return False
                
        except Exception as e:
            logger.error(f"Error posting content: {str(e)}")
            return False
    
    def get_market_sentiment_context(self) -> Dict[str, Any]:
        """Get current market sentiment context for better content generation"""
        # In production, this would fetch real-time data from:
        # - Fear & Greed Index
        # - Social sentiment APIs
        # - On-chain metrics
        # - Volume analysis
        
        return {
            "sentiment": "neutral_bullish",
            "trending_narratives": ["AI tokens", "Gaming tokens", "Solana memes"],
            "volume_trend": "increasing",
            "institutional_activity": "moderate"
        }
    
    def save_post_history(self, filename: str = "post_history.json"):
        """Save post history to file"""
        try:
            history_data = []
            for post in self.recent_posts:
                history_data.append({
                    "content": post.content,
                    "tokens_mentioned": post.tokens_mentioned,
                    "analysis_type": post.analysis_type,
                    "timestamp": post.timestamp.isoformat()
                })
            
            with open(filename, 'w') as f:
                json.dump(history_data, f, indent=2)
                
            logger.info(f"Post history saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving post history: {str(e)}")


def main():
    """Test the content generator"""
    try:
        generator = CryptoContentGenerator()
        
        # Generate different types of content
        for analysis_type in ["memecoin", "defi", "btc_eth"]:
            post = generator.create_technical_analysis_post(analysis_type)
            if post:
                print(f"\n{analysis_type.upper()} Analysis:")
                print(f"Content: {post.content}")
                print(f"Tokens: {post.tokens_mentioned}")
                print(f"Type: {post.analysis_type}")
                print("-" * 50)
        
        # Save history
        generator.save_post_history()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")


if __name__ == "__main__":
    main()
