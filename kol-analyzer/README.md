# KOL Credibility Analyzer

Analyze crypto Twitter/X Key Opinion Leaders (KOLs) for credibility. Detects LARPing, two-faced behavior, engagement baiting, and generates comprehensive credibility scores.

## Features

- **Engagement Analysis**: Detect fake/bot engagement patterns using Social Blade-style metrics
- **Consistency Tracking**: Track position changes on tokens/topics and detect flip-flopping
- **Dissonance Detection**: Identify hypocrisy and two-faced behavior
- **Engagement Bait Analysis**: Detect FOMO manufacturing, reward gaming, and manipulation tactics
- **Credibility Scoring**: Generate weighted overall credibility scores (A-F grades)
- **Comparison Mode**: Compare two KOLs side by side
- **REST API**: FastAPI server for programmatic access
- **Demo Mode**: Works out of the box without Twitter authentication

## Installation

```bash
# Clone/navigate to the project
cd kol-analyzer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) For real Twitter scraping, install Playwright browsers
playwright install chromium
```

## Quick Start

```bash
# Demo mode - works immediately without authentication
python main.py scan MINHxDYNASTY

# Scan with more tweets
python main.py scan cobie --tweets 500

# Export results to JSON
python main.py scan zachxbt --output results.json

# Compare two KOLs
python main.py compare cobie zachxbt

# List previously analyzed KOLs
python main.py list
```

## CLI Commands

### Scan a KOL

```bash
python main.py scan <username> [options]

Options:
  --tweets, -t    Maximum tweets to analyze (default: 200)
  --output, -o    Export results to JSON file
  --verbose, -v   Show detailed analysis output
```

### Compare Two KOLs

```bash
python main.py compare <user1> <user2>
```

### Setup Twitter Authentication

```bash
python main.py login
```

This opens a browser window for manual Twitter login. Once authenticated, the tool will use real data instead of demo mode.

### Start API Server

```bash
python main.py server [options]

Options:
  --host        Server host (default: 0.0.0.0)
  --port, -p    Server port (default: 8000)
  --reload, -r  Enable auto-reload for development
```

### List Analyzed KOLs

```bash
python main.py list
```

## API Endpoints

When running the server (`python main.py server`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/stats` | GET | Database statistics |
| `/analyze` | POST | Analyze a KOL |
| `/kol/{username}` | GET | Get cached analysis |
| `/kols` | GET | List all analyzed KOLs |
| `/compare` | POST | Compare two KOLs |
| `/cache/{username}` | DELETE | Clear cached data |
| `/history/{username}` | GET | Get analysis history |

API documentation available at `http://localhost:8000/docs`

## Analysis Components

### 1. Engagement Analyzer

Detects fake/bot engagement patterns:
- Calculates engagement rate coefficient of variation
- Flags suspiciously consistent engagement (CV < 0.1)
- Detects abnormal like-to-reply ratios (> 50:1)
- Identifies engagement spikes on text-only tweets
- Detects concentrated posting hours (automation)

### 2. Consistency Tracker

Tracks position changes over time:
- Extracts $TICKER mentions from tweets
- Classifies sentiment (bullish/bearish/neutral)
- Detects position flips with severity levels
- Credits self-acknowledged position changes
- Calculates consistency score

### 3. Dissonance Analyzer

Detects hypocrisy and two-faced behavior:
- Classifies tweet tones (instructional, derisive, gatekeeping, etc.)
- Detects power dynamic violations (mocking newcomers)
- Identifies hypocrisy (criticizing behaviors they've done)
- Calculates authenticity score

### 4. Engagement Bait Analyzer

Detects manipulation tactics:
- FOMO manufacturing
- Engagement farming (like/RT requests)
- Reward gaming (Kaito, Galxe mentions)
- Reply traps
- Rage bait
- Cliffhanger abuse
- Sympathy farming

## Credibility Score

The final score is calculated using weighted component scores:

| Component | Weight | Description |
|-----------|--------|-------------|
| Engagement | 20% | Authenticity of engagement patterns |
| Consistency | 25% | Position consistency over time |
| Dissonance | 25% | (Hypocrisy + Authenticity) / 2 |
| Baiting | 30% | Manipulation tactics score |

### Grade Thresholds

| Grade | Score Range | Assessment |
|-------|-------------|------------|
| A | 85-100 | HIGH CREDIBILITY |
| B | 70-84 | MODERATE CREDIBILITY |
| C | 55-69 | MIXED SIGNALS |
| D | 40-54 | LOW CREDIBILITY |
| F | 0-39 | POOR CREDIBILITY |

## Demo Mode

The tool works out of the box in demo mode, providing simulated but realistic data for testing. Demo profiles include:
- MINHxDYNASTY
- cobie
- zachxbt
- hsaka_
- CryptoKaleo

To analyze real Twitter data, run `python main.py login` to authenticate.

## Project Structure

```
kol-analyzer/
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── config/
│   └── settings.py        # Configuration
├── src/
│   ├── scraper/
│   │   ├── twitter_crawler.py  # Playwright automation
│   │   └── rate_limiter.py     # Human-like rate limiting
│   ├── analysis/
│   │   ├── engagement_analyzer.py      # Engagement patterns
│   │   ├── consistency_tracker.py      # Position flip detection
│   │   ├── dissonance_analyzer.py      # Hypocrisy detection
│   │   ├── engagement_bait_analyzer.py # Manipulation tactics
│   │   └── credibility_engine.py       # Final score calculator
│   ├── storage/
│   │   └── database.py     # SQLite storage
│   └── api/
│       └── main.py         # FastAPI server
└── data/                   # Database and cookies
```

## Output Example

```
╔══════════════════════════════════════════════════════════════╗
║  CREDIBILITY SCORE:  85.8/100    GRADE: A                   ║
║  Assessment: HIGH CREDIBILITY                                ║
║  Confidence: 72%                                             ║
╚══════════════════════════════════════════════════════════════╝

📊 COMPONENT SCORES:
   Engagement:    ████████████████░░░░ 75.0/100
   Consistency:   ███████████████████░ 95.0/100
   Dissonance:    ██████████████████░░ 90.0/100
   Baiting:       ████████████████░░░░ 80.0/100

🔴 RED FLAGS:
   • Active Kaito participant - reward incentives present
   • Frequent FOMO tactics (6 instances)

🟢 GREEN FLAGS:
   • Healthy engagement patterns
   • Transparently acknowledges position changes
   • Primarily instructional tone

📋 SUMMARY:
   This KOL shows strong credibility signals across all metrics.
   Content can generally be trusted with normal due diligence.
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License

## Disclaimer

This tool is for informational and educational purposes only. The credibility scores generated are based on pattern analysis and should not be used as the sole basis for financial decisions. Always do your own research (DYOR) and consult with financial advisors before making investment decisions.
