# Crypto Twitter Bot Setup Guide

## Overview
This enhanced KOLTerminal setup focuses on crypto, web3, and memecoin accounts with automatic replies and original technical analysis posts.

## Features
- **Auto-replies** to 10 key crypto influencers with crypto-specific context
- **Original content generation** every 3 hours with technical analysis on:
  - Memecoin trends and micro finance analysis
  - DeFi and web3 micro finance innovations  
  - BTC/ETH analysis with memecoin market context
- **Rate limiting** to maintain good standing (15 replies/hour, 2 posts/hour)
- **Crypto terminology** and culture integration

## Monitored Accounts
- **cobie** - DeFi and trading insights
- **punk6529** - NFT and web3 decentralization  
- **gmoneyNFT** - NFT culture and memecoin trends
- **DegenSpartan** - DeFi yield farming and altcoin analysis
- **loomdart** - On-chain analysis and DeFi protocols
- **rektcapital** - Technical analysis and chart patterns
- **DeFianceCapital** - VC perspective on DeFi
- **ZachXBT** - On-chain investigations and security
- **mileydyson** - Memecoin culture and web3 trends
- **CryptoCobain** - Memecoin alpha and degen culture

## Prerequisites
1. Twitter API v2 credentials with write permissions
2. Gemini API key
3. Python 3.8+ environment

## Environment Variables
Create a `.env` file with:
```
# Twitter API v2
X_API_KEY=your_api_key
X_API_SECRET=your_api_secret
X_ACCESS_TOKEN=your_access_token
X_ACCESS_TOKEN_SECRET=your_access_token_secret
X_BEARER_TOKEN=your_bearer_token

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key
```

## Installation
```bash
pip install tweepy google-generativeai python-dotenv loguru schedule pydantic
```

## Configuration
The `crypto_config.json` file contains all account settings. Key parameters:

- **reply_probability**: Chance of replying to each tweet (0.4-0.9 range)
- **tonality**: Response style (analytical, casual, professional, humorous, enthusiastic)
- **custom_instructions**: Crypto-specific guidance for each account
- **avoid_keywords**: Terms that trigger skipping tweets

## Usage

### Test Content Generation
```bash
python content_generator.py
```

### Test Bot Configuration
```bash
python crypto_bot_fixed.py
```

### Production Run
1. Update `crypto_config.json` with your bot username
2. Set `dry_run: false` for live posting
3. Run: `python crypto_bot_fixed.py`

## Content Types Generated

### Memecoin Analysis
- Technical patterns in trending memecoins
- Volume and social sentiment analysis
- Micro finance implications of memecoin adoption
- Cross-chain memecoin opportunities

### DeFi Micro Finance
- Micro lending protocol developments
- Small-cap yield farming opportunities
- Fractional NFT finance trends
- Decentralized P2P lending innovations

### BTC/ETH Context Analysis
- BTC dominance impact on altcoins
- ETH ecosystem effects on memecoin liquidity
- Risk-on vs risk-off sentiment analysis
- Correlation patterns affecting micro cap space

## Rate Limiting & Safety
- Maximum 15 replies per hour
- Maximum 2 original posts per hour  
- 3-hour intervals between content posts
- Built-in dry run mode for testing
- Comprehensive logging for monitoring

## Monitoring
Logs are stored in `logs/crypto_bot.log` with:
- All bot activities and decisions
- Rate limiting status
- Error handling and recovery
- Content generation details

## Customization
- Add/remove monitored accounts in `crypto_config.json`
- Adjust reply probabilities and tonalities
- Modify content generation prompts in `content_generator.py`
- Change posting intervals in `crypto_bot_fixed.py`

## Safety Notes
- Always test in dry run mode first
- Monitor Twitter API usage limits
- Respect account's posting frequency preferences
- Keep content analytical, avoid financial advice
- Use authentic crypto terminology naturally
