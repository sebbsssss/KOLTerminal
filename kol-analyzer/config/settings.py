"""Configuration settings for KOL Credibility Analyzer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple
import os


@dataclass
class ScraperSettings:
    """Settings for the Twitter scraper."""
    base_delay: Tuple[float, float] = (3.0, 8.0)
    burst_probability: float = 0.15
    fatigue_factor: float = 0.1
    max_tweets_default: int = 200
    cookies_path: str = "data/cookies.json"
    headless: bool = True
    timeout: int = 30000


@dataclass
class AnalysisSettings:
    """Settings for analysis modules."""
    # Engagement thresholds
    engagement_cv_threshold: float = 0.1  # Below this = suspicious
    like_reply_ratio_threshold: float = 50.0  # Above this = suspicious

    # Consistency scoring
    flip_penalty_major: float = 15.0
    flip_penalty_moderate: float = 8.0
    flip_penalty_minor: float = 3.0
    self_acknowledged_reduction: float = 0.5  # 50% reduction for transparency

    # Credibility weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'engagement': 0.20,
        'consistency': 0.25,
        'dissonance': 0.25,
        'baiting': 0.30
    })

    # Grade thresholds
    grade_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'A': 85.0,
        'B': 70.0,
        'C': 55.0,
        'D': 40.0,
        'F': 0.0
    })


@dataclass
class DatabaseSettings:
    """Settings for database storage."""
    db_path: str = "data/kol_analyzer.db"


@dataclass
class APISettings:
    """Settings for FastAPI server."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


@dataclass
class Settings:
    """Main settings container."""
    scraper: ScraperSettings = field(default_factory=ScraperSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    api: APISettings = field(default_factory=APISettings)

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")

    # Demo mode
    demo_mode: bool = True  # Set to False when authenticated

    def __post_init__(self):
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Update paths to be absolute
        self.scraper.cookies_path = str(self.data_dir / "cookies.json")
        self.database.db_path = str(self.data_dir / "kol_analyzer.db")

    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        settings = cls()

        # Override from environment
        if os.getenv('KOL_DEMO_MODE', '').lower() == 'false':
            settings.demo_mode = False

        if os.getenv('KOL_API_PORT'):
            settings.api.port = int(os.getenv('KOL_API_PORT'))

        if os.getenv('KOL_MAX_TWEETS'):
            settings.scraper.max_tweets_default = int(os.getenv('KOL_MAX_TWEETS'))

        return settings


# Global settings instance
settings = Settings.from_env()
