#!/usr/bin/env python3
"""
KOL Credibility Analyzer - CLI Entry Point

Analyze crypto Twitter/X Key Opinion Leaders (KOLs) for credibility.
Detects LARPing, two-faced behavior, engagement baiting, and generates credibility scores.

Usage:
    python main.py scan <username>              # Scan any Twitter username
    python main.py scan <username> --tweets 500 # Custom tweet count
    python main.py scan <username> --output results.json  # Export to JSON
    python main.py compare <user1> <user2>      # Compare two KOLs
    python main.py login                        # Setup Twitter authentication
    python main.py server                       # Start FastAPI server
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.scraper.twitter_crawler import TwitterCrawler, Tweet, UserProfile
from src.storage.database import Database
from src.analysis.credibility_engine import CredibilityEngine, CredibilityScore


class TerminalColors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_header():
    """Print the application header."""
    header = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ██╗  ██╗ ██████╗ ██╗          █████╗ ███╗   ██╗ █████╗ ██╗     ║
║   ██║ ██╔╝██╔═══██╗██║         ██╔══██╗████╗  ██║██╔══██╗██║     ║
║   █████╔╝ ██║   ██║██║         ███████║██╔██╗ ██║███████║██║     ║
║   ██╔═██╗ ██║   ██║██║         ██╔══██║██║╚██╗██║██╔══██║██║     ║
║   ██║  ██╗╚██████╔╝███████╗    ██║  ██║██║ ╚████║██║  ██║███████╗║
║   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝║
║                                                                   ║
║              KOL Credibility Analyzer v1.0.0                      ║
║         Detect LARPing, Manipulation & Two-Faced Behavior         ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(f"{TerminalColors.CYAN}{header}{TerminalColors.ENDC}")


def print_score_box(score: CredibilityScore, username: str):
    """Print the credibility score in a formatted box."""
    grade_colors = {
        'A': TerminalColors.GREEN,
        'B': TerminalColors.CYAN,
        'C': TerminalColors.YELLOW,
        'D': TerminalColors.RED,
        'F': TerminalColors.RED
    }

    color = grade_colors.get(score.grade, TerminalColors.ENDC)

    print(f"""
{TerminalColors.BOLD}╔══════════════════════════════════════════════════════════════╗
║  CREDIBILITY SCORE:  {color}{score.overall_score:5.1f}/100{TerminalColors.ENDC}{TerminalColors.BOLD}    GRADE: {color}{score.grade}{TerminalColors.ENDC}{TerminalColors.BOLD}                   ║
║  Assessment: {color}{score.assessment:<30}{TerminalColors.ENDC}{TerminalColors.BOLD}              ║
║  Confidence: {score.confidence:.0f}%                                            ║
╚══════════════════════════════════════════════════════════════╝{TerminalColors.ENDC}
""")


def print_component_scores(score: CredibilityScore):
    """Print component scores."""
    print(f"{TerminalColors.BOLD}📊 COMPONENT SCORES:{TerminalColors.ENDC}")

    def score_bar(value: float, width: int = 20) -> str:
        filled = int(value / 100 * width)
        bar = '█' * filled + '░' * (width - filled)
        if value >= 70:
            color = TerminalColors.GREEN
        elif value >= 50:
            color = TerminalColors.YELLOW
        else:
            color = TerminalColors.RED
        return f"{color}{bar}{TerminalColors.ENDC}"

    print(f"   Engagement:    {score_bar(score.engagement_score)} {score.engagement_score:5.1f}/100")
    print(f"   Consistency:   {score_bar(score.consistency_score)} {score.consistency_score:5.1f}/100")
    print(f"   Dissonance:    {score_bar(score.dissonance_score)} {score.dissonance_score:5.1f}/100")
    print(f"   Baiting:       {score_bar(score.baiting_score)} {score.baiting_score:5.1f}/100")
    print()


def print_flags(score: CredibilityScore):
    """Print red and green flags."""
    if score.red_flags:
        print(f"{TerminalColors.RED}{TerminalColors.BOLD}🔴 RED FLAGS:{TerminalColors.ENDC}")
        for flag in score.red_flags:
            print(f"   {TerminalColors.RED}• {flag}{TerminalColors.ENDC}")
        print()

    if score.green_flags:
        print(f"{TerminalColors.GREEN}{TerminalColors.BOLD}🟢 GREEN FLAGS:{TerminalColors.ENDC}")
        for flag in score.green_flags:
            print(f"   {TerminalColors.GREEN}• {flag}{TerminalColors.ENDC}")
        print()


def print_summary(score: CredibilityScore):
    """Print the analysis summary."""
    print(f"{TerminalColors.BOLD}📋 SUMMARY:{TerminalColors.ENDC}")
    print(f"   {score.summary}")
    print()


def print_progress(message: str):
    """Print progress message."""
    print(f"   {TerminalColors.DIM}{message}{TerminalColors.ENDC}", end='\r')


async def scan_command(
    username: str,
    max_tweets: int = 200,
    output: Optional[str] = None,
    verbose: bool = False
):
    """Execute the scan command."""
    username = username.lstrip('@')

    print(f"\n{TerminalColors.BOLD}Analyzing @{username}...{TerminalColors.ENDC}\n")

    # Initialize components
    crawler = TwitterCrawler(
        cookies_path=settings.scraper.cookies_path,
        headless=settings.scraper.headless
    )
    db = Database(settings.database.db_path)
    engine = CredibilityEngine()

    try:
        # Initialize crawler
        print(f"   Initializing scraper...")
        await crawler.initialize()

        # Get profile
        print(f"   Fetching profile for @{username}...")
        profile = await crawler.get_user_profile(username, print_progress)

        if not profile:
            print(f"\n{TerminalColors.RED}Error: Could not find user @{username}{TerminalColors.ENDC}")
            return

        print(f"   Found: {profile.display_name} ({profile.follower_count:,} followers)")

        # Get tweets
        print(f"   Fetching tweets...")

        def progress_callback(msg: str):
            print(f"   {TerminalColors.DIM}{msg}{TerminalColors.ENDC}          ", end='\r')

        tweets = await crawler.get_user_tweets(
            username,
            max_tweets=max_tweets,
            progress_callback=progress_callback
        )

        print(f"\n   Collected {len(tweets)} tweets" + (" (demo mode)" if crawler.demo_mode else ""))

        # Convert to dict format for analyzers
        tweets_data = [
            {
                'id': t.id,
                'text': t.text,
                'timestamp': t.timestamp,
                'likes': t.likes,
                'retweets': t.retweets,
                'replies': t.replies,
                'has_media': t.has_media,
                'has_video': t.has_video,
                'is_quote_tweet': t.is_quote_tweet
            }
            for t in tweets
        ]

        # Run analysis
        print(f"   Running credibility analysis...")
        result = engine.analyze(tweets_data, profile.follower_count, username)

        # Save to database
        kol_id = db.upsert_kol(profile)
        db.save_tweets(kol_id, tweets)
        db.save_analysis(kol_id, result.to_dict(), len(tweets))

        # Print results
        print_score_box(result, username)
        print_component_scores(result)
        print_flags(result)
        print_summary(result)

        # Export to JSON if requested
        if output:
            output_data = {
                'username': username,
                'profile': {
                    'display_name': profile.display_name,
                    'follower_count': profile.follower_count,
                    'following_count': profile.following_count,
                    'bio': profile.bio
                },
                'analysis': result.to_dict(),
                'tweets_analyzed': len(tweets),
                'demo_mode': crawler.demo_mode
            }

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

            print(f"{TerminalColors.GREEN}Results exported to: {output}{TerminalColors.ENDC}\n")

        # Print verbose details if requested
        if verbose:
            print(f"\n{TerminalColors.BOLD}📈 DETAILED ANALYSIS:{TerminalColors.ENDC}")
            print(json.dumps(result.to_dict(), indent=2, default=str))

    finally:
        await crawler.close()


async def compare_command(username1: str, username2: str):
    """Execute the compare command."""
    username1 = username1.lstrip('@')
    username2 = username2.lstrip('@')

    print(f"\n{TerminalColors.BOLD}Comparing @{username1} vs @{username2}...{TerminalColors.ENDC}\n")

    # Initialize components
    db = Database(settings.database.db_path)
    engine = CredibilityEngine()

    # Check for cached analyses
    analysis1 = db.get_latest_analysis(username1)
    analysis2 = db.get_latest_analysis(username2)

    if not analysis1:
        print(f"{TerminalColors.YELLOW}No cached analysis for @{username1}. Running scan first...{TerminalColors.ENDC}")
        await scan_command(username1)
        analysis1 = db.get_latest_analysis(username1)

    if not analysis2:
        print(f"{TerminalColors.YELLOW}No cached analysis for @{username2}. Running scan first...{TerminalColors.ENDC}")
        await scan_command(username2)
        analysis2 = db.get_latest_analysis(username2)

    if not analysis1 or not analysis2:
        print(f"{TerminalColors.RED}Error: Could not get analyses for comparison{TerminalColors.ENDC}")
        return

    # Build comparison
    print(f"""
{TerminalColors.BOLD}╔══════════════════════════════════════════════════════════════════╗
║                        COMPARISON RESULTS                         ║
╠══════════════════════════════════════════════════════════════════╣{TerminalColors.ENDC}
""")

    def compare_metric(name: str, val1: float, val2: float):
        winner = "=" if abs(val1 - val2) < 1 else ("←" if val1 > val2 else "→")
        color1 = TerminalColors.GREEN if val1 > val2 else (TerminalColors.RED if val1 < val2 else TerminalColors.ENDC)
        color2 = TerminalColors.GREEN if val2 > val1 else (TerminalColors.RED if val2 < val1 else TerminalColors.ENDC)
        print(f"   {name:<15} {color1}{val1:>6.1f}{TerminalColors.ENDC}  {winner}  {color2}{val2:<6.1f}{TerminalColors.ENDC}")

    print(f"   {'METRIC':<15} {'@'+username1:>8}     {'@'+username2:<8}")
    print(f"   {'-'*45}")

    compare_metric("Overall", analysis1['overall_score'], analysis2['overall_score'])
    compare_metric("Engagement", analysis1['engagement_score'], analysis2['engagement_score'])
    compare_metric("Consistency", analysis1['consistency_score'], analysis2['consistency_score'])
    compare_metric("Dissonance", analysis1['dissonance_score'], analysis2['dissonance_score'])
    compare_metric("Baiting", analysis1['baiting_score'], analysis2['baiting_score'])

    print(f"\n   {'Grade':<15} {analysis1['grade']:>8}     {analysis2['grade']:<8}")

    # Winner
    diff = analysis1['overall_score'] - analysis2['overall_score']
    if abs(diff) < 3:
        verdict = "TIE - Both accounts have similar credibility"
    elif diff > 0:
        verdict = f"@{username1} has higher credibility (+{diff:.1f} points)"
    else:
        verdict = f"@{username2} has higher credibility (+{-diff:.1f} points)"

    print(f"""
{TerminalColors.BOLD}╠══════════════════════════════════════════════════════════════════╣
║  VERDICT: {verdict:<55}║
╚══════════════════════════════════════════════════════════════════╝{TerminalColors.ENDC}
""")


async def login_command():
    """Execute the login command."""
    print(f"\n{TerminalColors.BOLD}Twitter Authentication Setup{TerminalColors.ENDC}\n")

    crawler = TwitterCrawler(
        cookies_path=settings.scraper.cookies_path,
        headless=False  # Need visible browser for login
    )

    success = await crawler.login_manual()

    if success:
        print(f"\n{TerminalColors.GREEN}Authentication successful! You can now scrape real Twitter data.{TerminalColors.ENDC}")
    else:
        print(f"\n{TerminalColors.YELLOW}Authentication skipped. Tool will run in demo mode.{TerminalColors.ENDC}")


def server_command(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Execute the server command."""
    print(f"\n{TerminalColors.BOLD}Starting KOL Analyzer API Server...{TerminalColors.ENDC}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print()

    try:
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        print(f"{TerminalColors.RED}Error: uvicorn not installed. Run: pip install uvicorn{TerminalColors.ENDC}")
        sys.exit(1)


def list_command():
    """List all analyzed KOLs."""
    db = Database(settings.database.db_path)
    kols = db.list_kols(limit=20)

    if not kols:
        print(f"\n{TerminalColors.YELLOW}No KOLs analyzed yet. Run 'python main.py scan <username>' first.{TerminalColors.ENDC}\n")
        return

    print(f"\n{TerminalColors.BOLD}Analyzed KOLs:{TerminalColors.ENDC}\n")
    print(f"   {'Username':<20} {'Score':>8} {'Grade':>6} {'Followers':>12}")
    print(f"   {'-'*50}")

    for kol in kols:
        score = kol.get('latest_score', 0) or 0
        grade = kol.get('latest_grade', '-') or '-'
        followers = kol.get('follower_count', 0) or 0

        grade_color = {
            'A': TerminalColors.GREEN,
            'B': TerminalColors.CYAN,
            'C': TerminalColors.YELLOW,
            'D': TerminalColors.RED,
            'F': TerminalColors.RED
        }.get(grade, TerminalColors.ENDC)

        print(f"   @{kol['username']:<19} {score:>7.1f} {grade_color}{grade:>6}{TerminalColors.ENDC} {followers:>12,}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='KOL Credibility Analyzer - Detect LARPing and manipulation in crypto Twitter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py scan MINHxDYNASTY           Scan a KOL
  python main.py scan cobie --tweets 500     Scan with more tweets
  python main.py scan zachxbt -o results.json Export to JSON
  python main.py compare user1 user2         Compare two KOLs
  python main.py login                       Setup Twitter auth
  python main.py server                      Start API server
  python main.py list                        List analyzed KOLs
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan a Twitter username')
    scan_parser.add_argument('username', help='Twitter username to analyze')
    scan_parser.add_argument('--tweets', '-t', type=int, default=200,
                            help='Maximum number of tweets to analyze (default: 200)')
    scan_parser.add_argument('--output', '-o', help='Export results to JSON file')
    scan_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed analysis output')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two KOLs')
    compare_parser.add_argument('user1', help='First Twitter username')
    compare_parser.add_argument('user2', help='Second Twitter username')

    # Login command
    login_parser = subparsers.add_parser('login', help='Setup Twitter authentication')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start the FastAPI server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    server_parser.add_argument('--port', '-p', type=int, default=8000,
                              help='Server port (default: 8000)')
    server_parser.add_argument('--reload', '-r', action='store_true',
                              help='Enable auto-reload for development')

    # List command
    list_parser = subparsers.add_parser('list', help='List analyzed KOLs')

    args = parser.parse_args()

    # Print header
    print_header()

    # Execute command
    if args.command == 'scan':
        asyncio.run(scan_command(
            args.username,
            max_tweets=args.tweets,
            output=args.output,
            verbose=args.verbose
        ))
    elif args.command == 'compare':
        asyncio.run(compare_command(args.user1, args.user2))
    elif args.command == 'login':
        asyncio.run(login_command())
    elif args.command == 'server':
        server_command(host=args.host, port=args.port, reload=args.reload)
    elif args.command == 'list':
        list_command()
    else:
        parser.print_help()
        print(f"\n{TerminalColors.YELLOW}Tip: Run 'python main.py scan MINHxDYNASTY' to try demo mode{TerminalColors.ENDC}\n")


if __name__ == '__main__':
    main()
