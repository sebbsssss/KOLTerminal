#!/usr/bin/env python3
"""
KOL Analyzer Terminal UI
A beautiful terminal interface for analyzing crypto Key Opinion Leaders.
Built with Textual using the Polar Night color palette.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import asdict

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    ProgressBar,
    Static,
)

from rich.text import Text

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import CredibilityEngine
from src.scraper import TwitterCrawler
from src.storage import Database

# ASCII Art Header
ASCII_LOGO = """
╔═══════════════════════════════════════╗
║  ██╗  ██╗ ██████╗ ██╗                 ║
║  ██║ ██╔╝██╔═══██╗██║                 ║
║  █████╔╝ ██║   ██║██║                 ║
║  ██╔═██╗ ██║   ██║██║                 ║
║  ██║  ██╗╚██████╔╝███████╗            ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝            ║
║      Credibility Analyzer             ║
╚═══════════════════════════════════════╝
"""


def get_grade_color(grade: str) -> str:
    """Get color for grade."""
    colors = {
        "A": "#a3be8c",
        "B": "#8fbcbb",
        "C": "#ebcb8b",
        "D": "#d08770",
        "F": "#bf616a",
    }
    return colors.get(grade, "#d8dee9")


def get_score_color(score: float) -> str:
    """Get color based on score."""
    if score >= 85:
        return "#a3be8c"
    elif score >= 70:
        return "#8fbcbb"
    elif score >= 55:
        return "#ebcb8b"
    elif score >= 40:
        return "#d08770"
    else:
        return "#bf616a"


class ScanModal(ModalScreen):
    """Modal for scanning a new KOL."""

    CSS = """
    ScanModal {
        align: center middle;
    }

    .modal-dialog {
        background: #3b4252;
        border: solid #434c5e;
        padding: 2;
        width: 60;
        height: auto;
    }

    .modal-title {
        text-align: center;
        text-style: bold;
        color: #88c0d0;
        margin-bottom: 1;
    }

    .modal-input {
        margin: 1 0;
    }

    .modal-buttons {
        margin-top: 1;
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog"):
            yield Static("Scan New KOL", classes="modal-title")
            yield Static("Enter Twitter username to analyze:", classes="dim")
            yield Input(placeholder="@username or username", id="scan-input", classes="modal-input")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Scan", variant="primary", id="scan-btn")

    @on(Button.Pressed, "#cancel-btn")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#scan-btn")
    def scan(self) -> None:
        username = self.query_one("#scan-input", Input).value.strip().lstrip("@")
        if username:
            self.dismiss(username)
        else:
            self.notify("Please enter a username", severity="warning")

    @on(Input.Submitted, "#scan-input")
    def submit_input(self) -> None:
        self.scan()


class CompareModal(ModalScreen):
    """Modal for comparing two KOLs."""

    CSS = """
    CompareModal {
        align: center middle;
    }

    .modal-dialog {
        background: #3b4252;
        border: solid #434c5e;
        padding: 2;
        width: 60;
        height: auto;
    }

    .modal-title {
        text-align: center;
        text-style: bold;
        color: #88c0d0;
        margin-bottom: 1;
    }

    .modal-input {
        margin: 1 0;
    }

    .modal-buttons {
        margin-top: 1;
        align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-dialog"):
            yield Static("Compare KOLs", classes="modal-title")
            yield Static("First KOL username:", classes="dim")
            yield Input(placeholder="@username", id="user1-input", classes="modal-input")
            yield Static("Second KOL username:", classes="dim")
            yield Input(placeholder="@username", id="user2-input", classes="modal-input")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Compare", variant="primary", id="compare-btn")

    @on(Button.Pressed, "#cancel-btn")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#compare-btn")
    def compare(self) -> None:
        user1 = self.query_one("#user1-input", Input).value.strip().lstrip("@")
        user2 = self.query_one("#user2-input", Input).value.strip().lstrip("@")
        if user1 and user2:
            self.dismiss((user1, user2))
        else:
            self.notify("Please enter both usernames", severity="warning")


class StatsPanel(Static):
    """Panel showing database statistics."""

    def __init__(self, db: Database) -> None:
        super().__init__()
        self.db = db

    def compose(self) -> ComposeResult:
        yield Static("DATABASE STATS", classes="stats-title")
        yield Static("Loading...", id="stats-content")

    def on_mount(self) -> None:
        self.refresh_stats()

    def refresh_stats(self) -> None:
        try:
            stats = self.db.get_stats()
            content = self.query_one("#stats-content", Static)
            text = Text()
            text.append("KOLs Analyzed: ", style="#d8dee9")
            text.append(f"{stats.get('kols_analyzed', 0)}\n", style="#8fbcbb bold")
            text.append("Total Analyses: ", style="#d8dee9")
            text.append(f"{stats.get('total_analyses', 0)}\n", style="#8fbcbb bold")
            text.append("Tweets Stored: ", style="#d8dee9")
            text.append(f"{stats.get('total_tweets', 0)}\n", style="#8fbcbb bold")
            text.append("Average Score: ", style="#d8dee9")
            avg = stats.get('average_score', 0)
            text.append(f"{avg:.1f}", style=f"{get_score_color(avg)} bold")
            content.update(text)
        except Exception as e:
            content = self.query_one("#stats-content", Static)
            content.update(f"Error: {e}")


class ScoreDisplay(Static):
    """Large score display widget."""

    def __init__(self, score: float, grade: str, confidence: float) -> None:
        super().__init__()
        self.score = score
        self.grade = grade
        self.confidence = confidence

    def compose(self) -> ComposeResult:
        text = Text()
        text.append("\n")
        text.append(f"{self.score:.0f}", style=f"{get_score_color(self.score)} bold")
        text.append(" / 100\n", style="#4c566a")
        text.append(f"Grade: ", style="#d8dee9")
        text.append(f"{self.grade}\n", style=f"{get_grade_color(self.grade)} bold")
        text.append(f"Confidence: {self.confidence:.0f}%", style="#81a1c1")
        yield Static(text, classes="score-large")


class ScoreBar(Static):
    """Score progress bar with label."""

    def __init__(self, label: str, score: float, bar_class: str = "") -> None:
        super().__init__()
        self.label = label
        self.score = score
        self.bar_class = bar_class

    def compose(self) -> ComposeResult:
        with Horizontal(classes="score-bar-container"):
            yield Static(f"{self.label}:", classes="score-label")
            bar = ProgressBar(total=100, show_eta=False)
            if self.bar_class:
                bar.add_class(self.bar_class)
            yield bar
            yield Static(f" {self.score:.0f}", classes="metric-value")

    def on_mount(self) -> None:
        bar = self.query_one(ProgressBar)
        bar.update(progress=self.score)


class FlagsPanel(Static):
    """Panel showing red/green flags."""

    def __init__(self, flags: list, is_red: bool = True) -> None:
        super().__init__()
        self.flags = flags or []
        self.is_red = is_red

    def compose(self) -> ComposeResult:
        title_class = "-red" if self.is_red else "-green"
        icon = "!" if self.is_red else "+"
        title = "RED FLAGS" if self.is_red else "GREEN FLAGS"

        yield Static(f" {icon} {title}", classes=f"flags-title {title_class}")

        if self.flags:
            for flag in self.flags[:8]:
                flag_class = "-red" if self.is_red else "-green"
                prefix = "  - " if self.is_red else "  + "
                yield Static(f"{prefix}{flag}", classes=f"flag-item {flag_class}")
        else:
            yield Static("  No flags detected", classes="dim")


class AnalysisView(Static):
    """Full analysis view for a KOL."""

    def __init__(self, username: str, analysis: dict, profile: dict = None) -> None:
        super().__init__()
        self.username = username
        self.analysis = analysis or {}
        self.profile = profile or {}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            # Header with username
            header_text = Text()
            header_text.append(f"@{self.username}", style="#88c0d0 bold")
            if self.profile.get("display_name"):
                header_text.append(f" ({self.profile['display_name']})", style="#81a1c1")
            yield Static(header_text, classes="card-header")

            # Profile info
            if self.profile:
                profile_text = Text()
                followers = self.profile.get("follower_count", 0) or 0
                following = self.profile.get("following_count", 0) or 0
                profile_text.append(f"Followers: ", style="#d8dee9")
                profile_text.append(f"{followers:,}", style="#8fbcbb bold")
                profile_text.append(f"  |  Following: ", style="#d8dee9")
                profile_text.append(f"{following:,}", style="#8fbcbb bold")
                yield Static(profile_text)
                if self.profile.get("bio"):
                    yield Static(f"\n{self.profile['bio'][:200]}", classes="dim")

            yield Static("")

            # Score display
            with Container(classes="score-display"):
                yield ScoreDisplay(
                    self.analysis.get("overall_score", 0),
                    self.analysis.get("grade", "?"),
                    self.analysis.get("confidence", 0)
                )

            # Assessment
            assessment = self.analysis.get("assessment", "Unknown")
            assess_text = Text()
            assess_text.append("Assessment: ", style="#d8dee9")
            assess_text.append(assessment, style=f"{get_score_color(self.analysis.get('overall_score', 50))} bold")
            yield Static(assess_text)

            yield Static("")

            # Component scores
            yield Static("COMPONENT SCORES", classes="title")
            yield ScoreBar("Engagement", self.analysis.get("engagement_score", 0), "-engagement")
            yield ScoreBar("Consistency", self.analysis.get("consistency_score", 0), "-consistency")
            yield ScoreBar("Dissonance", self.analysis.get("dissonance_score", 0), "-dissonance")
            yield ScoreBar("Baiting", self.analysis.get("baiting_score", 0), "-baiting")

            yield Static("")

            # Flags
            with Horizontal():
                with Vertical():
                    yield FlagsPanel(self.analysis.get("red_flags", []), is_red=True)
                with Vertical():
                    yield FlagsPanel(self.analysis.get("green_flags", []), is_red=False)

            yield Static("")

            # Summary
            if self.analysis.get("summary"):
                yield Static("SUMMARY", classes="title")
                yield Static(self.analysis["summary"], classes="summary-text")

            yield Static("")

            # Detailed metrics if available
            detailed = self.analysis.get("detailed_analysis", {})
            if detailed:
                yield Static("DETAILED METRICS", classes="title")

                # Engagement details
                engagement = detailed.get("engagement", {})
                if engagement:
                    yield Static("\nEngagement Analysis:", classes="subtitle")
                    yield Static(f"  Avg Engagement Rate: {engagement.get('avg_engagement_rate', 0):.2%}")
                    yield Static(f"  Bot Follower Estimate: {engagement.get('bot_follower_estimate', 0):.1f}%")
                    yield Static(f"  Like/Reply Ratio: {engagement.get('like_reply_ratio', 0):.1f}")

                # Consistency details
                consistency = detailed.get("consistency", {})
                if consistency:
                    yield Static("\nConsistency Analysis:", classes="subtitle")
                    yield Static(f"  Position Flips: {consistency.get('flip_count', 0)}")
                    yield Static(f"  Major Flips: {consistency.get('major_flip_count', 0)}")
                    yield Static(f"  Topics Tracked: {len(consistency.get('topics_tracked', []))}")

                # Baiting details
                baiting = detailed.get("baiting", {})
                if baiting:
                    yield Static("\nBait Analysis:", classes="subtitle")
                    yield Static(f"  Manipulation Index: {baiting.get('manipulation_index', 0):.1f}")
                    yield Static(f"  Bait Percentage: {baiting.get('bait_percentage', 0):.1f}%")
                    verdict = baiting.get('verdict', 'Unknown')
                    yield Static(f"  Verdict: {verdict}")


class CompareView(Static):
    """Side-by-side comparison view."""

    def __init__(self, user1: str, user2: str, analysis1: dict, analysis2: dict) -> None:
        super().__init__()
        self.user1 = user1
        self.user2 = user2
        self.analysis1 = analysis1 or {}
        self.analysis2 = analysis2 or {}

    def compose(self) -> ComposeResult:
        score1 = self.analysis1.get("overall_score", 0)
        score2 = self.analysis2.get("overall_score", 0)

        with Horizontal(classes="compare-container"):
            # Left panel
            with Vertical(classes="compare-panel"):
                header1 = Text()
                header1.append(f"@{self.user1}", style="#88c0d0 bold")
                if score1 > score2:
                    header1.append(" WINNER", style="#a3be8c bold")
                yield Static(header1, classes="compare-header")

                yield ScoreDisplay(
                    score1,
                    self.analysis1.get("grade", "?"),
                    self.analysis1.get("confidence", 0)
                )
                yield Static("")
                yield ScoreBar("Engagement", self.analysis1.get("engagement_score", 0))
                yield ScoreBar("Consistency", self.analysis1.get("consistency_score", 0))
                yield ScoreBar("Dissonance", self.analysis1.get("dissonance_score", 0))
                yield ScoreBar("Baiting", self.analysis1.get("baiting_score", 0))

            # Right panel
            with Vertical(classes="compare-panel"):
                header2 = Text()
                header2.append(f"@{self.user2}", style="#88c0d0 bold")
                if score2 > score1:
                    header2.append(" WINNER", style="#a3be8c bold")
                yield Static(header2, classes="compare-header")

                yield ScoreDisplay(
                    score2,
                    self.analysis2.get("grade", "?"),
                    self.analysis2.get("confidence", 0)
                )
                yield Static("")
                yield ScoreBar("Engagement", self.analysis2.get("engagement_score", 0))
                yield ScoreBar("Consistency", self.analysis2.get("consistency_score", 0))
                yield ScoreBar("Dissonance", self.analysis2.get("dissonance_score", 0))
                yield ScoreBar("Baiting", self.analysis2.get("baiting_score", 0))


class DashboardScreen(Screen):
    """Main dashboard screen."""

    BINDINGS = [
        Binding("s", "scan", "Scan KOL"),
        Binding("c", "compare", "Compare"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("enter", "view_selected", "View Details"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.db = Database()
        self.crawler = TwitterCrawler()
        self.engine = CredibilityEngine()

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(classes="main-container"):
            # Sidebar with stats
            with Vertical(classes="sidebar"):
                yield Static(ASCII_LOGO, classes="ascii-header")
                yield StatsPanel(self.db)
                yield Static("")
                yield Button("Scan New KOL", variant="primary", id="scan-btn")
                yield Button("Compare KOLs", id="compare-btn")
                yield Button("Refresh List", id="refresh-btn")

            # Main content - KOL list
            with Vertical(classes="content-panel"):
                yield Static("ANALYZED KOLS", classes="title")
                yield DataTable(id="kol-table", classes="kol-list")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#kol-table", DataTable)
        table.add_columns("Username", "Score", "Grade", "Followers", "Updated")
        table.cursor_type = "row"
        self.refresh_kol_list()

    def refresh_kol_list(self) -> None:
        table = self.query_one("#kol-table", DataTable)
        table.clear()

        try:
            kols = self.db.list_kols(limit=50)
            for kol in kols:
                score = kol.get("latest_score") or 0
                grade = kol.get("latest_grade") or "?"
                followers = kol.get("follower_count") or 0
                updated = kol.get("updated_at") or ""
                if updated:
                    try:
                        dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                        updated = dt.strftime("%Y-%m-%d")
                    except:
                        updated = str(updated)[:10]

                # Color the score based on value
                score_text = Text(f"{score:.0f}", style=get_score_color(score))
                grade_text = Text(grade, style=get_grade_color(grade))

                table.add_row(
                    f"@{kol['username']}",
                    score_text,
                    grade_text,
                    f"{followers:,}",
                    updated
                )
        except Exception as e:
            self.notify(f"Error loading KOLs: {e}", severity="error")

    @on(Button.Pressed, "#scan-btn")
    def show_scan_modal(self) -> None:
        self.action_scan()

    @on(Button.Pressed, "#compare-btn")
    def show_compare_modal(self) -> None:
        self.action_compare()

    @on(Button.Pressed, "#refresh-btn")
    def refresh_list(self) -> None:
        self.action_refresh()

    def action_scan(self) -> None:
        self.app.push_screen(ScanModal(), self.handle_scan_result)

    def action_compare(self) -> None:
        self.app.push_screen(CompareModal(), self.handle_compare_result)

    def action_refresh(self) -> None:
        self.notify("Refreshing...")
        self.refresh_kol_list()
        # Refresh stats
        stats_panel = self.query_one(StatsPanel)
        stats_panel.refresh_stats()
        self.notify("Refreshed!", severity="information")

    def action_view_selected(self) -> None:
        table = self.query_one("#kol-table", DataTable)
        if table.cursor_row is not None:
            row = table.get_row_at(table.cursor_row)
            if row:
                username = str(row[0]).lstrip("@")
                self.view_kol(username)

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        row = event.data_table.get_row_at(event.cursor_row)
        if row:
            username = str(row[0]).lstrip("@")
            self.view_kol(username)

    def view_kol(self, username: str) -> None:
        """View detailed analysis for a KOL."""
        self.app.push_screen(AnalysisScreen(username, self.db, self.crawler, self.engine))

    def handle_scan_result(self, username: Optional[str]) -> None:
        if username:
            self.run_scan(username)

    def handle_compare_result(self, result: Optional[tuple]) -> None:
        if result:
            user1, user2 = result
            self.run_compare(user1, user2)

    @work(thread=True)
    async def run_scan(self, username: str) -> None:
        """Run a scan in the background."""
        self.notify(f"Scanning @{username}...")

        try:
            # Initialize crawler
            await self.crawler.initialize()

            # Fetch profile
            profile = await self.crawler.get_user_profile(username)
            if not profile:
                self.app.call_from_thread(
                    self.notify, f"Could not find user @{username}", severity="error"
                )
                return

            # Fetch tweets
            tweets = await self.crawler.get_user_tweets(username, max_tweets=200)

            # Convert tweets to dict format for analyzer
            tweets_data = []
            for tweet in tweets:
                tweets_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'timestamp': tweet.timestamp,
                    'likes': tweet.likes,
                    'retweets': tweet.retweets,
                    'replies': tweet.replies,
                    'has_media': tweet.has_media,
                    'has_video': tweet.has_video,
                    'is_quote_tweet': tweet.is_quote_tweet
                })

            # Run analysis
            result = self.engine.analyze(tweets_data, profile.follower_count, username)

            # Save to database
            kol_id = self.db.upsert_kol(profile)
            self.db.save_tweets(kol_id, tweets)
            self.db.save_analysis(kol_id, result.to_dict(), len(tweets))

            self.app.call_from_thread(
                self.notify, f"Scan complete for @{username}: {result.grade}", severity="information"
            )

            # Refresh list
            self.app.call_from_thread(self.refresh_kol_list)

        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Scan failed: {e}", severity="error"
            )

    @work(thread=True)
    async def run_compare(self, user1: str, user2: str) -> None:
        """Run comparison in the background."""
        self.notify(f"Comparing @{user1} vs @{user2}...")

        try:
            # Get or run analysis for both users
            analysis1 = await self.get_or_run_analysis(user1)
            analysis2 = await self.get_or_run_analysis(user2)

            if analysis1 and analysis2:
                self.app.call_from_thread(
                    self.app.push_screen,
                    CompareScreen(user1, user2, analysis1, analysis2)
                )
            else:
                self.app.call_from_thread(
                    self.notify, "Could not get analysis for one or both users", severity="error"
                )

        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Comparison failed: {e}", severity="error"
            )

    async def get_or_run_analysis(self, username: str) -> Optional[dict]:
        """Get cached analysis or run a new one."""
        cached = self.db.get_latest_analysis(username)
        if cached:
            return cached

        # Initialize crawler if needed
        await self.crawler.initialize()

        # Run fresh analysis
        profile = await self.crawler.get_user_profile(username)
        if not profile:
            return None

        tweets = await self.crawler.get_user_tweets(username, max_tweets=200)

        # Convert tweets to dict format
        tweets_data = []
        for tweet in tweets:
            tweets_data.append({
                'id': tweet.id,
                'text': tweet.text,
                'timestamp': tweet.timestamp,
                'likes': tweet.likes,
                'retweets': tweet.retweets,
                'replies': tweet.replies,
                'has_media': tweet.has_media,
                'has_video': tweet.has_video,
                'is_quote_tweet': tweet.is_quote_tweet
            })

        result = self.engine.analyze(tweets_data, profile.follower_count, username)

        # Save to database
        kol_id = self.db.upsert_kol(profile)
        self.db.save_tweets(kol_id, tweets)
        self.db.save_analysis(kol_id, result.to_dict(), len(tweets))

        return result.to_dict()


class AnalysisScreen(Screen):
    """Screen showing detailed analysis for a KOL."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("r", "refresh_analysis", "Re-scan"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, username: str, db: Database, crawler: TwitterCrawler, engine: CredibilityEngine) -> None:
        super().__init__()
        self.username = username
        self.db = db
        self.crawler = crawler
        self.engine = engine
        self.analysis = None
        self.profile = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="content-panel"):
            yield LoadingIndicator()
        yield Footer()

    def on_mount(self) -> None:
        self.load_analysis()

    def load_analysis(self) -> None:
        try:
            # Get cached profile
            self.profile = self.db.get_kol(self.username)

            # Get cached analysis
            self.analysis = self.db.get_latest_analysis(self.username)

            if not self.analysis:
                self.notify(f"No analysis found for @{self.username}, running scan...")
                self.run_fresh_analysis()
                return

            # Update display
            container = self.query_one(VerticalScroll)
            container.remove_children()
            container.mount(AnalysisView(self.username, self.analysis, self.profile))

        except Exception as e:
            self.notify(f"Error loading analysis: {e}", severity="error")

    @work(thread=True)
    async def run_fresh_analysis(self) -> None:
        try:
            await self.crawler.initialize()

            profile = await self.crawler.get_user_profile(self.username)
            if not profile:
                self.app.call_from_thread(
                    self.notify, f"Could not find user @{self.username}", severity="error"
                )
                return

            tweets = await self.crawler.get_user_tweets(self.username, max_tweets=200)

            # Convert tweets to dict format
            tweets_data = []
            for tweet in tweets:
                tweets_data.append({
                    'id': tweet.id,
                    'text': tweet.text,
                    'timestamp': tweet.timestamp,
                    'likes': tweet.likes,
                    'retweets': tweet.retweets,
                    'replies': tweet.replies,
                    'has_media': tweet.has_media,
                    'has_video': tweet.has_video,
                    'is_quote_tweet': tweet.is_quote_tweet
                })

            result = self.engine.analyze(tweets_data, profile.follower_count, self.username)

            # Save to database
            kol_id = self.db.upsert_kol(profile)
            self.db.save_tweets(kol_id, tweets)
            self.db.save_analysis(kol_id, result.to_dict(), len(tweets))

            self.profile = {
                'username': profile.username,
                'display_name': profile.display_name,
                'bio': profile.bio,
                'follower_count': profile.follower_count,
                'following_count': profile.following_count,
                'tweet_count': profile.tweet_count
            }
            self.analysis = result.to_dict()

            def update_view():
                container = self.query_one(VerticalScroll)
                container.remove_children()
                container.mount(AnalysisView(self.username, self.analysis, self.profile))

            self.app.call_from_thread(update_view)

        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Analysis failed: {e}", severity="error"
            )

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh_analysis(self) -> None:
        self.notify(f"Re-scanning @{self.username}...")
        container = self.query_one(VerticalScroll)
        container.remove_children()
        container.mount(LoadingIndicator())
        self.run_fresh_analysis()


class CompareScreen(Screen):
    """Screen showing comparison between two KOLs."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, user1: str, user2: str, analysis1: dict, analysis2: dict) -> None:
        super().__init__()
        self.user1 = user1
        self.user2 = user2
        self.analysis1 = analysis1
        self.analysis2 = analysis2

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(classes="content-panel"):
            yield Static("KOL COMPARISON", classes="title")
            yield CompareView(self.user1, self.user2, self.analysis1, self.analysis2)
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()


class KOLAnalyzerApp(App):
    """Main KOL Analyzer Terminal UI Application."""

    TITLE = "KOL Analyzer"
    SUB_TITLE = "Crypto Influencer Credibility Analysis"
    CSS_PATH = "terminal_ui.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen())


def main():
    """Run the KOL Analyzer Terminal UI."""
    app = KOLAnalyzerApp()
    app.run()


if __name__ == "__main__":
    main()
