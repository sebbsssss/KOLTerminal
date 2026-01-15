# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

KOLTerminal is a dual-purpose crypto Twitter automation and analysis system consisting of:

1. **Twitter Reply Bot** (`bot.py`, `crypto_bot.py`) - Automated engagement and content generation for Twitter
2. **KOL Credibility Analyzer** (`kol-analyzer/`) - Comprehensive credibility analysis tool for crypto Twitter KOLs (Key Opinion Leaders)

## Development Setup

### Environment Setup
```bash
# Root project (Twitter bots)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt

# KOL Analyzer (separate environment)
cd kol-analyzer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium  # For Twitter scraping
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
- **Required for bots**: `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`, `X_BEARER_TOKEN`, `GEMINI_API_KEY`, `BOT_USERNAME`
- **Required for KOL Analyzer**: `SUPABASE_URL`, `SUPABASE_KEY` (for caching), `RAPIDAPI_KEY` (for Twitter241 API)
- **Bot settings**: `DRY_RUN=true` (for testing), `CHECK_INTERVAL_MINUTES=5`, `LOG_LEVEL=INFO`

## Common Commands

### Twitter Bots

#### Running Bots
```bash
# Generic reply bot (uses bot_config.json)
python bot.py

# Crypto-focused bot with content generation (uses crypto_config.json)
python crypto_bot.py

# Alternative crypto bot (fixed version)
python crypto_bot_fixed.py
```

#### Testing
```bash
# Test bot in dry-run for 25 seconds
python test_bot.py

# Test content generation
python test_content.py

# Test API setup and credentials
python test_setup.py
```

#### Configuration
Bot behavior is controlled by JSON config files:
- `bot_config.json` - Generic bot configuration
- `crypto_config.json` - Crypto-specific bot with content generation

### KOL Analyzer

#### CLI Usage
```bash
cd kol-analyzer

# Analyze a KOL (works in demo mode without auth)
python main.py scan <username>
python main.py scan cobie --tweets 500
python main.py scan zachxbt --output results.json

# Compare two KOLs
python main.py compare cobie zachxbt

# List analyzed KOLs
python main.py list

# Setup Twitter authentication (for real data)
python main.py login

# Start FastAPI server
python main.py server
python main.py server --port 8000 --reload
```

#### API Server
```bash
cd kol-analyzer
python main.py server

# API available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

Key endpoints:
- `POST /analyze` - Analyze a KOL
- `GET /kol/{username}` - Get cached analysis
- `POST /compare` - Compare two KOLs
- `GET /stats` - Database statistics

## Architecture

### Twitter Bot System

The bot system uses a modular architecture with clear separation of concerns:

**Core Components:**
- `config.py` - Configuration models using Pydantic (TonalityType, ContentType, AccountConfig, BotConfig)
- `x_client.py` - Twitter API wrapper (tweepy-based, handles all X/Twitter interactions)
- `gemini_client.py` - AI reply generation using Google Gemini
- `content_generator.py` - Original content creation for crypto analysis posts

**Bot Implementations:**
- `bot.py` - Generic reply bot (monitors accounts, generates replies based on config)
- `crypto_bot.py` / `crypto_bot_fixed.py` - Enhanced crypto bot with automatic content posting

**Key Design Patterns:**
1. **Rate Limiting**: Both bots track replies per hour and reset counters hourly
2. **Tweet Deduplication**: Processed tweet IDs stored in memory (last 1000) to avoid duplicate replies
3. **Dry Run Mode**: All posting can be simulated via `dry_run` config flag
4. **Scheduled Execution**: Uses `schedule` library for periodic checks
5. **Configuration-Driven**: Account-specific behavior (tonality, content types, reply probability) defined in JSON

**Content Generation Flow (Crypto Bot):**
1. Rotates between analysis types: memecoin, DeFi, BTC/ETH
2. Uses Gemini to generate technical analysis posts
3. Posts original content every 3 hours
4. Integrates crypto terminology naturally ("gm", "wagmi", "based", etc.)

### KOL Analyzer System

The KOL Analyzer is a comprehensive credibility scoring system with multi-module architecture:

**Project Structure:**
```
kol-analyzer/
├── main.py                 # CLI entry point
├── terminal_ui.py          # Rich TUI (currently not primary interface)
├── config/settings.py      # Centralized configuration
├── src/
│   ├── scraper/
│   │   ├── twitter_crawler.py  # Playwright-based scraping + demo mode
│   │   └── rate_limiter.py     # Human-like delays
│   ├── analysis/
│   │   ├── engagement_analyzer.py        # Bot detection
│   │   ├── consistency_tracker.py        # Position flip detection
│   │   ├── dissonance_analyzer.py        # Hypocrisy detection
│   │   ├── engagement_bait_analyzer.py   # Manipulation tactics
│   │   ├── credibility_engine.py         # Score aggregation
│   │   ├── archetype_classifier.py       # KOL personality types
│   │   ├── prediction_tracker.py         # Track predictions
│   │   ├── sponsored_detector.py         # Sponsored content detection
│   │   └── ... (other analyzers)
│   ├── storage/database.py   # SQLite storage
│   └── api/main.py           # FastAPI server
└── data/                     # SQLite DB, cookies, cached data
```

**Analysis Pipeline:**
1. **Scraping**: Playwright-based Twitter scraping with rate limiting (or demo mode with mock data)
2. **Multi-Module Analysis**: Each analyzer runs independently and generates scores (0-100)
3. **Scoring**: Credibility engine aggregates scores with weights (engagement 20%, consistency 25%, dissonance 25%, baiting 30%)
4. **Grading**: A (85-100), B (70-84), C (55-69), D (40-54), F (0-39)
5. **Storage**: Results cached in SQLite with historical tracking

**Demo Mode:**
- Works out of the box without Twitter authentication
- Provides realistic simulated data for testing
- Pre-configured profiles: MINHxDYNASTY, cobie, zachxbt, hsaka_, CryptoKaleo
- Run `python main.py login` to enable real Twitter data

**Key Features:**
- Detects engagement manipulation (bot followers, suspicious patterns)
- Tracks position flips on tokens/projects
- Identifies hypocrisy and two-faced behavior
- Flags engagement bait tactics (FOMO, rage bait, sympathy farming)
- Generates comprehensive credibility reports with red/green flags

## Testing Philosophy

- **Always test in dry-run mode first**: Set `DRY_RUN=true` or `dry_run: true` in config before any production run
- **Monitor rate limits**: Twitter has strict rate limits; bots track this internally
- **Test files are functional**: `test_bot.py`, `test_content.py`, `test_setup.py` are actual test scripts, not pytest tests
- **KOL Analyzer demo mode**: No credentials needed for initial testing

## Configuration Management

### Bot Configuration Files
- `bot_config.json` / `crypto_config.json` define per-account behavior
- Each account has: `tonality`, `content_types`, `reply_probability`, `custom_instructions`, `avoid_keywords`
- Global settings: `check_interval_minutes`, `dry_run`, `max_replies_per_hour`, `bot_username`

### KOL Analyzer Settings
- Centralized in `kol-analyzer/config/settings.py`
- Analysis thresholds, credibility weights, grade thresholds
- Can override via environment variables (e.g., `KOL_API_PORT`, `KOL_MAX_TWEETS`)

## Logging

### Twitter Bots
- Uses `loguru` for structured logging
- Logs written to `logs/bot.log` or `logs/crypto_bot.log`
- Rotation: daily, retention: 7 days
- All API calls, replies, errors, and rate limit status logged

### KOL Analyzer
- CLI uses colored terminal output with progress indicators
- Database stores all analysis results with timestamps
- Detailed analysis breakdowns available in API responses

## Security & Credentials

- **Never commit `.env` files** - `.env.example` provided as template
- **API keys in environment only** - No hardcoded credentials
- **Twitter bot username** must be configured in bot_config.json `bot_username` field
- **RapidAPI key** (Twitter241) optional for KOL Analyzer, demo mode works without it

## Common Pitfalls

1. **Invalid bot username**: Must be actual Twitter username (not "your_bot_username"), otherwise API returns 400 errors
2. **Missing Twitter permissions**: Twitter app must have Read+Write permissions for posting
3. **Rate limiting**: Both Twitter API and bot configs enforce rate limits - monitor closely
4. **Virtual environment confusion**: Root and kol-analyzer have separate venvs
5. **Config file format**: JSON must be valid - common error is trailing commas
6. **Dry run oversight**: Forgetting to disable dry_run will prevent actual posting

## Key Files Reference

### Root Level
- `bot.py` - Main generic reply bot
- `crypto_bot.py` / `crypto_bot_fixed.py` - Crypto bot variants
- `config.py` - Pydantic models for bot configuration
- `x_client.py` - Twitter API client (tweepy wrapper)
- `gemini_client.py` - AI reply generation
- `content_generator.py` - Crypto content generation
- `bot_config.json` / `crypto_config.json` - Bot behavior configs

### KOL Analyzer (`kol-analyzer/`)
- `main.py` - CLI entry point for all operations
- `terminal_ui.py` - TUI implementation (alternative interface)
- `config/settings.py` - Centralized configuration
- `src/analysis/credibility_engine.py` - Score aggregation and grading
- `src/scraper/twitter_crawler.py` - Twitter scraping with demo mode
- `src/api/main.py` - FastAPI server
- `src/storage/database.py` - SQLite persistence

## Crypto Bot Content Types

The crypto bot generates three types of technical analysis:

1. **Memecoin Analysis** - Trends, volume, social sentiment, micro-finance implications
2. **DeFi Micro Finance** - Lending protocols, yield farming, fractional NFT finance
3. **BTC/ETH Context** - Dominance impact on alts, risk sentiment, correlation patterns

Content rotates automatically and posts every 3 hours with crypto-native terminology.

## Development Workflow

1. Make code changes
2. Test in dry-run mode first
3. Check logs for expected behavior
4. Monitor rate limits
5. For KOL Analyzer: test with demo mode first, then authenticate if needed
6. Review generated content/replies before going live
7. Set `dry_run=false` only when confident
8. Monitor logs during live operation

## API Integration Notes

**Twitter API (v2):**
- All interactions via `x_client.py`
- Uses tweepy with `wait_on_rate_limit=True`
- Supports: fetch tweets, post tweets, reply to tweets, search recent tweets

**Gemini API:**
- Model: `gemini-2.0-flash-exp`
- Used for reply generation and content creation
- Prompts engineered for crypto context and natural language

**RapidAPI (Twitter241):**
- Optional for KOL Analyzer real-time data
- Falls back to demo mode if not configured
