"""Admin API endpoints for managing KOLs and uploading scraped tweets."""

import csv
import io
import json
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, Response, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel


router = APIRouter(prefix="/admin", tags=["admin"])

# Will be set by main app
db = None
ADMIN_PASSWORD = None  # Set from environment variable

# Simple session store (in production, use Redis or database)
active_sessions = {}


def set_db(database):
    """Set the database instance."""
    global db
    db = database


def set_admin_password(password: str):
    """Set the admin password."""
    global ADMIN_PASSWORD
    ADMIN_PASSWORD = password


def hash_password(password: str) -> str:
    """Hash a password for comparison."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_session(admin_session: Optional[str] = Cookie(None)):
    """Verify the admin session cookie."""
    if not ADMIN_PASSWORD:
        # No password set, allow access (for development)
        return True

    if not admin_session:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login at /admin-panel"
        )

    if admin_session not in active_sessions:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please login again."
        )

    # Check if session is expired (24 hours)
    session_time = active_sessions[admin_session]
    if datetime.now(timezone.utc) - session_time > timedelta(hours=24):
        del active_sessions[admin_session]
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please login again."
        )

    return True


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
async def admin_login(request: LoginRequest, response: Response):
    """Login to admin panel."""
    if not ADMIN_PASSWORD:
        # No password configured, create session anyway
        session_token = secrets.token_urlsafe(32)
        active_sessions[session_token] = datetime.now(timezone.utc)
        response.set_cookie(
            key="admin_session",
            value=session_token,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return {"success": True, "message": "Logged in (no password configured)"}

    if request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Create session
    session_token = secrets.token_urlsafe(32)
    active_sessions[session_token] = datetime.now(timezone.utc)

    response.set_cookie(
        key="admin_session",
        value=session_token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax"
    )

    return {"success": True, "message": "Logged in successfully"}


@router.post("/logout")
async def admin_logout(response: Response, admin_session: Optional[str] = Cookie(None)):
    """Logout from admin panel."""
    if admin_session and admin_session in active_sessions:
        del active_sessions[admin_session]

    response.delete_cookie("admin_session")
    return {"success": True, "message": "Logged out"}


@router.get("/check-auth")
async def check_auth(admin_session: Optional[str] = Cookie(None)):
    """Check if user is authenticated."""
    if not ADMIN_PASSWORD:
        return {"authenticated": True, "password_required": False}

    if admin_session and admin_session in active_sessions:
        session_time = active_sessions[admin_session]
        if datetime.now(timezone.utc) - session_time <= timedelta(hours=24):
            return {"authenticated": True, "password_required": True}

    return {"authenticated": False, "password_required": True}


class UserCreate(BaseModel):
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    follower_count: Optional[int] = 0
    following_count: Optional[int] = 0
    tweet_count: Optional[int] = 0
    verified: Optional[bool] = False
    profile_image_url: Optional[str] = None


class TweetCreate(BaseModel):
    tweet_id: str
    text: str
    timestamp: Optional[str] = None
    likes: Optional[int] = 0
    retweets: Optional[int] = 0
    replies: Optional[int] = 0
    quotes: Optional[int] = 0
    views: Optional[int] = 0
    bookmarks: Optional[int] = 0
    has_media: Optional[bool] = False
    has_video: Optional[bool] = False
    media_type: Optional[str] = None
    media_urls: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = None
    tweet_type: Optional[str] = None  # Tweet, Quoted, Origin, etc.


class BulkTweetUpload(BaseModel):
    username: str
    tweets: List[TweetCreate]


@router.get("/users")
async def list_users(limit: int = 100, offset: int = 0, _: bool = Depends(verify_session)):
    """List all KOLs in the database with tweet counts."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        # Get all KOLs
        result = db.client.table("kols").select(
            "id, username, display_name, follower_count, tweet_count, latest_score, latest_grade, created_at, updated_at"
        ).order("created_at", desc=True).range(offset, offset + limit - 1).execute()

        if not result.data:
            return {"users": [], "total": 0, "message": "No users in database. Add users using the form above."}

        users = []
        for kol in result.data:
            # Get actual tweet count from tweets table
            try:
                tweet_result = db.client.table("tweets").select(
                    "id", count="exact"
                ).eq("kol_id", kol["id"]).execute()
                stored_tweets = tweet_result.count if tweet_result.count else 0
            except:
                stored_tweets = 0

            users.append({
                **kol,
                "stored_tweets": stored_tweets
            })

        return {"users": users, "total": len(users)}

    except Exception as e:
        import traceback
        print(f"Failed to list users: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to list users: {e}")


@router.get("/users/{username}")
async def get_user(username: str, _: bool = Depends(verify_session)):
    """Get a specific user's details and tweet count."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        # Get tweet count
        tweet_result = db.client.table("tweets").select(
            "id", count="exact"
        ).eq("kol_id", kol["id"]).execute()

        return {
            **kol,
            "stored_tweets": tweet_result.count if tweet_result.count else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {e}")


@router.post("/users")
async def create_user(user: UserCreate, _: bool = Depends(verify_session)):
    """Create a new KOL entry."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        # Check if user already exists
        existing = db.get_kol(user.username.lower())
        if existing:
            return {"message": "User already exists", "user": existing}

        # Create new user
        data = {
            "username": user.username.lower(),
            "display_name": user.display_name or user.username,
            "bio": user.bio or "",
            "follower_count": user.follower_count,
            "following_count": user.following_count,
            "tweet_count": user.tweet_count,
            "verified": user.verified,
            "profile_image_url": user.profile_image_url or f"https://ui-avatars.com/api/?name={user.username}&background=random&size=200",
        }

        result = db.client.table("kols").insert(data).execute()

        return {"message": "User created", "user": result.data[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {e}")


@router.get("/users/{username}/tweets")
async def get_user_tweets(username: str, limit: int = 50, offset: int = 0, _: bool = Depends(verify_session)):
    """Get stored tweets for a user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        result = db.client.table("tweets").select("*").eq(
            "kol_id", kol["id"]
        ).order("timestamp", desc=True).range(offset, offset + limit - 1).execute()

        # Get total count
        count_result = db.client.table("tweets").select(
            "id", count="exact"
        ).eq("kol_id", kol["id"]).execute()

        return {
            "tweets": result.data,
            "total": count_result.count if count_result.count else 0,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tweets: {e}")


@router.post("/users/{username}/tweets")
async def upload_tweets(username: str, tweets: List[TweetCreate], _: bool = Depends(verify_session)):
    """Upload tweets for a user (JSON format)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found. Create user first.")

        kol_id = kol["id"]
        saved_count = 0
        errors = []

        for tweet in tweets:
            try:
                data = {
                    "kol_id": kol_id,
                    "tweet_id": tweet.tweet_id,
                    "text": tweet.text[:2000] if tweet.text else "",
                    "timestamp": tweet.timestamp,
                    "likes": tweet.likes or 0,
                    "retweets": tweet.retweets or 0,
                    "replies": tweet.replies or 0,
                    "quotes": tweet.quotes or 0,
                    "views": tweet.views or 0,
                    "has_media": tweet.has_media or (tweet.media_type is not None),
                    "has_video": tweet.has_video or (tweet.media_type == "video"),
                    "raw_data": {
                        "bookmarks": tweet.bookmarks,
                        "media_type": tweet.media_type,
                        "media_urls": tweet.media_urls,
                        "source": tweet.source,
                        "language": tweet.language,
                        "tweet_type": tweet.tweet_type
                    }
                }

                db.client.table("tweets").upsert(
                    data, on_conflict="tweet_id"
                ).execute()
                saved_count += 1

            except Exception as e:
                errors.append(f"Tweet {tweet.tweet_id}: {e}")

        return {
            "message": f"Uploaded {saved_count} tweets for @{username}",
            "saved": saved_count,
            "errors": errors[:10] if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload tweets: {e}")


@router.post("/users/{username}/tweets/csv")
async def upload_tweets_csv(username: str, file: UploadFile = File(...), _: bool = Depends(verify_session)):
    """
    Upload tweets from CSV file.

    Expected columns (from your scraper):
    - ID (tweet ID)
    - Text
    - Language
    - Type (Tweet, Quoted, Origin)
    - Author Name
    - Author Username
    - View Count
    - Reply Count
    - Retweet Count
    - Quote Count
    - Favorite Count
    - Bookmark Count
    - Created At
    - Source
    - hashtags
    - urls
    - media_type
    - media_urls
    """
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found. Create user first.")

        kol_id = kol["id"]

        # Read CSV content
        content = await file.read()
        decoded = content.decode('utf-8')

        # Parse CSV
        reader = csv.DictReader(io.StringIO(decoded))

        saved_count = 0
        skipped_count = 0
        errors = []

        for row in reader:
            try:
                # Map CSV columns to our schema
                tweet_id = row.get('ID') or row.get('id') or row.get('tweet_id')
                if not tweet_id:
                    skipped_count += 1
                    continue

                # Parse timestamp
                created_at = row.get('Created At') or row.get('created_at') or row.get('timestamp')
                timestamp = None
                if created_at:
                    try:
                        # Try different date formats
                        for fmt in [
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%d %H:%M:%S.%f',
                            '%Y-%m-%dT%H:%M:%SZ',
                            '%a %b %d %H:%M:%S %z %Y'  # Twitter format
                        ]:
                            try:
                                timestamp = datetime.strptime(created_at, fmt).isoformat()
                                break
                            except:
                                continue
                        if not timestamp:
                            timestamp = created_at  # Store as-is if parsing fails
                    except:
                        timestamp = created_at

                # Get media info
                media_type = row.get('media_type') or row.get('Media Type')
                has_media = bool(media_type)
                has_video = media_type == 'video' if media_type else False

                data = {
                    "kol_id": kol_id,
                    "tweet_id": str(tweet_id),
                    "text": (row.get('Text') or row.get('text') or "")[:2000],
                    "timestamp": timestamp,
                    "likes": int(row.get('Favorite Count') or row.get('favorite_count') or row.get('likes') or 0),
                    "retweets": int(row.get('Retweet Count') or row.get('retweet_count') or row.get('retweets') or 0),
                    "replies": int(row.get('Reply Count') or row.get('reply_count') or row.get('replies') or 0),
                    "quotes": int(row.get('Quote Count') or row.get('quote_count') or row.get('quotes') or 0),
                    "views": int(row.get('View Count') or row.get('view_count') or row.get('views') or 0),
                    "has_media": has_media,
                    "has_video": has_video,
                    "raw_data": {
                        "bookmarks": int(row.get('Bookmark Count') or row.get('bookmark_count') or 0),
                        "media_type": media_type,
                        "media_urls": row.get('media_urls') or row.get('Media URLs'),
                        "source": row.get('Source') or row.get('source'),
                        "language": row.get('Language') or row.get('language'),
                        "tweet_type": row.get('Type') or row.get('type'),
                        "hashtags": row.get('hashtags') or row.get('Hashtags'),
                        "urls": row.get('urls') or row.get('URLs'),
                        "author_name": row.get('Author Name') or row.get('author_name'),
                    }
                }

                db.client.table("tweets").upsert(
                    data, on_conflict="tweet_id"
                ).execute()
                saved_count += 1

            except Exception as e:
                errors.append(f"Row error: {e}")
                if len(errors) > 50:
                    break

        # Update KOL's tweet count
        try:
            count_result = db.client.table("tweets").select(
                "id", count="exact"
            ).eq("kol_id", kol_id).execute()

            db.client.table("kols").update({
                "tweet_count": count_result.count,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", kol_id).execute()
        except:
            pass

        return {
            "message": f"Uploaded {saved_count} tweets for @{username}",
            "saved": saved_count,
            "skipped": skipped_count,
            "errors": errors[:10] if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload CSV: {e}")


@router.delete("/users/{username}/tweets")
async def delete_user_tweets(username: str, _: bool = Depends(verify_session)):
    """Delete all tweets for a user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        db.client.table("tweets").delete().eq("kol_id", kol["id"]).execute()

        return {"message": f"Deleted all tweets for @{username}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete tweets: {e}")


@router.delete("/users/{username}")
async def delete_user(username: str, _: bool = Depends(verify_session)):
    """Delete a user and all their data."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        kol = db.get_kol(username.lower())
        if not kol:
            raise HTTPException(status_code=404, detail=f"User @{username} not found")

        # Cascade delete will handle tweets, mentions, analyses
        db.client.table("kols").delete().eq("id", kol["id"]).execute()

        return {"message": f"Deleted @{username} and all associated data"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {e}")


@router.get("/stats")
async def get_admin_stats(_: bool = Depends(verify_session)):
    """Get database statistics."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    stats = {
        "total_kols": 0,
        "total_tweets": 0,
        "total_analyses": 0,
        "total_mentions": 0
    }

    try:
        # Count KOLs
        kols_result = db.client.table("kols").select("id", count="exact").execute()
        stats["total_kols"] = kols_result.count or 0
    except Exception as e:
        print(f"Error counting kols: {e}")

    try:
        # Count tweets
        tweets_result = db.client.table("tweets").select("id", count="exact").execute()
        stats["total_tweets"] = tweets_result.count or 0
    except Exception as e:
        print(f"Error counting tweets: {e}")

    try:
        # Count analyses
        analyses_result = db.client.table("analyses").select("id", count="exact").execute()
        stats["total_analyses"] = analyses_result.count or 0
    except Exception as e:
        print(f"Error counting analyses: {e}")

    try:
        # Count mentions (may not exist if schema not updated)
        mentions_result = db.client.table("mentions").select("id", count="exact").execute()
        stats["total_mentions"] = mentions_result.count or 0
    except Exception as e:
        print(f"Error counting mentions (table may not exist): {e}")

    return stats
