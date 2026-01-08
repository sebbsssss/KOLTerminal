"""SQLite database storage for KOL analysis data."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..scraper.twitter_crawler import Tweet, UserProfile


class Database:
    """SQLite database for storing KOL analysis data."""

    def __init__(self, db_path: str = "data/kol_analyzer.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper handling."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # KOLs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    display_name TEXT,
                    bio TEXT,
                    follower_count INTEGER,
                    following_count INTEGER,
                    tweet_count INTEGER,
                    latest_score REAL,
                    latest_grade TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Analyses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kol_id INTEGER NOT NULL,
                    overall_score REAL,
                    grade TEXT,
                    confidence REAL,
                    assessment TEXT,
                    engagement_score REAL,
                    consistency_score REAL,
                    dissonance_score REAL,
                    baiting_score REAL,
                    red_flags TEXT,
                    green_flags TEXT,
                    summary TEXT,
                    detailed_analysis TEXT,
                    tweets_analyzed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kol_id) REFERENCES kols(id)
                )
            """)

            # Tweets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kol_id INTEGER NOT NULL,
                    tweet_id TEXT UNIQUE,
                    text TEXT,
                    timestamp TIMESTAMP,
                    likes INTEGER,
                    retweets INTEGER,
                    replies INTEGER,
                    has_media BOOLEAN,
                    has_video BOOLEAN,
                    is_quote_tweet BOOLEAN,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (kol_id) REFERENCES kols(id)
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tweets_kol_id ON tweets(kol_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyses_kol_id ON analyses(kol_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kols_username ON kols(username)
            """)

    def upsert_kol(self, profile: UserProfile) -> int:
        """
        Insert or update a KOL profile.

        Returns:
            The KOL's database ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT id FROM kols WHERE username = ?",
                (profile.username.lower(),)
            )
            row = cursor.fetchone()

            if row:
                # Update existing
                cursor.execute("""
                    UPDATE kols SET
                        display_name = ?,
                        bio = ?,
                        follower_count = ?,
                        following_count = ?,
                        tweet_count = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    profile.display_name,
                    profile.bio,
                    profile.follower_count,
                    profile.following_count,
                    profile.tweet_count,
                    row['id']
                ))
                return row['id']
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO kols (
                        username, display_name, bio,
                        follower_count, following_count, tweet_count
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    profile.username.lower(),
                    profile.display_name,
                    profile.bio,
                    profile.follower_count,
                    profile.following_count,
                    profile.tweet_count
                ))
                return cursor.lastrowid

    def save_tweets(self, kol_id: int, tweets: List[Tweet]):
        """Save tweets to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            for tweet in tweets:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO tweets (
                            kol_id, tweet_id, text, timestamp,
                            likes, retweets, replies,
                            has_media, has_video, is_quote_tweet
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        kol_id,
                        tweet.id,
                        tweet.text,
                        tweet.timestamp,
                        tweet.likes,
                        tweet.retweets,
                        tweet.replies,
                        tweet.has_media,
                        tweet.has_video,
                        tweet.is_quote_tweet
                    ))
                except sqlite3.IntegrityError:
                    # Tweet already exists, skip
                    pass

    def save_analysis(
        self,
        kol_id: int,
        analysis_data: Dict[str, Any],
        tweets_analyzed: int
    ) -> int:
        """
        Save an analysis result.

        Returns:
            The analysis ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert analysis
            cursor.execute("""
                INSERT INTO analyses (
                    kol_id, overall_score, grade, confidence, assessment,
                    engagement_score, consistency_score, dissonance_score, baiting_score,
                    red_flags, green_flags, summary, detailed_analysis, tweets_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kol_id,
                analysis_data.get('overall_score'),
                analysis_data.get('grade'),
                analysis_data.get('confidence'),
                analysis_data.get('assessment'),
                analysis_data.get('engagement_score'),
                analysis_data.get('consistency_score'),
                analysis_data.get('dissonance_score'),
                analysis_data.get('baiting_score'),
                json.dumps(analysis_data.get('red_flags', [])),
                json.dumps(analysis_data.get('green_flags', [])),
                analysis_data.get('summary'),
                json.dumps(analysis_data.get('detailed_analysis', {})),
                tweets_analyzed
            ))

            analysis_id = cursor.lastrowid

            # Update KOL's latest score
            cursor.execute("""
                UPDATE kols SET
                    latest_score = ?,
                    latest_grade = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                analysis_data.get('overall_score'),
                analysis_data.get('grade'),
                kol_id
            ))

            return analysis_id

    def get_kol(self, username: str) -> Optional[Dict[str, Any]]:
        """Get a KOL by username."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM kols WHERE username = ?",
                (username.lower(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_kol_by_id(self, kol_id: int) -> Optional[Dict[str, Any]]:
        """Get a KOL by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM kols WHERE id = ?", (kol_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_latest_analysis(self, username: str) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis for a KOL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.* FROM analyses a
                JOIN kols k ON a.kol_id = k.id
                WHERE k.username = ?
                ORDER BY a.created_at DESC
                LIMIT 1
            """, (username.lower(),))
            row = cursor.fetchone()

            if row:
                result = dict(row)
                result['red_flags'] = json.loads(result['red_flags'] or '[]')
                result['green_flags'] = json.loads(result['green_flags'] or '[]')
                result['detailed_analysis'] = json.loads(result['detailed_analysis'] or '{}')
                return result
            return None

    def get_all_analyses(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history for a KOL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.* FROM analyses a
                JOIN kols k ON a.kol_id = k.id
                WHERE k.username = ?
                ORDER BY a.created_at DESC
                LIMIT ?
            """, (username.lower(), limit))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['red_flags'] = json.loads(result['red_flags'] or '[]')
                result['green_flags'] = json.loads(result['green_flags'] or '[]')
                result['detailed_analysis'] = json.loads(result['detailed_analysis'] or '{}')
                results.append(result)
            return results

    def get_tweets(self, kol_id: int, limit: int = 500) -> List[Dict[str, Any]]:
        """Get tweets for a KOL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tweets
                WHERE kol_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (kol_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def list_kols(
        self,
        limit: int = 50,
        order_by: str = "updated_at",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """List all analyzed KOLs."""
        order_dir = "ASC" if ascending else "DESC"
        valid_columns = ["username", "follower_count", "latest_score", "updated_at", "created_at"]

        if order_by not in valid_columns:
            order_by = "updated_at"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM kols
                WHERE latest_score IS NOT NULL
                ORDER BY {order_by} {order_dir}
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_kol_cache(self, username: str) -> bool:
        """Delete all cached data for a KOL."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get KOL ID
            cursor.execute(
                "SELECT id FROM kols WHERE username = ?",
                (username.lower(),)
            )
            row = cursor.fetchone()

            if not row:
                return False

            kol_id = row['id']

            # Delete tweets
            cursor.execute("DELETE FROM tweets WHERE kol_id = ?", (kol_id,))

            # Delete analyses
            cursor.execute("DELETE FROM analyses WHERE kol_id = ?", (kol_id,))

            # Delete KOL
            cursor.execute("DELETE FROM kols WHERE id = ?", (kol_id,))

            return True

    def get_comparison_data(self, usernames: List[str]) -> List[Dict[str, Any]]:
        """Get comparison data for multiple KOLs."""
        results = []
        for username in usernames:
            kol = self.get_kol(username)
            if kol:
                analysis = self.get_latest_analysis(username)
                if analysis:
                    results.append({
                        'kol': kol,
                        'analysis': analysis
                    })
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM kols")
            kol_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM analyses")
            analysis_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM tweets")
            tweet_count = cursor.fetchone()['count']

            cursor.execute("""
                SELECT AVG(latest_score) as avg_score
                FROM kols WHERE latest_score IS NOT NULL
            """)
            avg_score = cursor.fetchone()['avg_score'] or 0

            return {
                'kols_analyzed': kol_count,
                'total_analyses': analysis_count,
                'total_tweets': tweet_count,
                'average_score': round(avg_score, 1)
            }
