-- Supabase Schema for KOL Terminal
-- Run this in your Supabase SQL Editor to set up the database

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLES
-- ============================================================================

-- KOLs (Key Opinion Leaders) - User profiles
CREATE TABLE IF NOT EXISTS kols (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    twitter_id TEXT UNIQUE,  -- Twitter's internal user ID for API calls
    display_name TEXT,
    bio TEXT,
    follower_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    tweet_count INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE,
    profile_image_url TEXT,
    joined_date TIMESTAMPTZ,

    -- Caching metadata
    latest_score REAL,
    latest_grade TEXT,
    last_tweet_id TEXT,  -- Most recent tweet ID we have (for incremental fetching)
    last_fetched_at TIMESTAMPTZ,  -- When we last fetched tweets for this user

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tweets - Cached tweet data
CREATE TABLE IF NOT EXISTS tweets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kol_id UUID NOT NULL REFERENCES kols(id) ON DELETE CASCADE,
    tweet_id TEXT UNIQUE NOT NULL,  -- Twitter's tweet ID
    text TEXT,
    timestamp TIMESTAMPTZ,

    -- Engagement metrics
    likes INTEGER DEFAULT 0,
    retweets INTEGER DEFAULT 0,
    replies INTEGER DEFAULT 0,
    quotes INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,

    -- Tweet metadata
    has_media BOOLEAN DEFAULT FALSE,
    has_video BOOLEAN DEFAULT FALSE,
    is_quote_tweet BOOLEAN DEFAULT FALSE,
    is_reply BOOLEAN DEFAULT FALSE,
    is_retweet BOOLEAN DEFAULT FALSE,
    reply_to_user TEXT,
    reply_to_tweet_id TEXT,

    -- Raw data for future analysis
    raw_data JSONB,

    -- Timestamps
    scraped_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analyses - Credibility analysis results
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kol_id UUID NOT NULL REFERENCES kols(id) ON DELETE CASCADE,

    -- Core scores
    overall_score REAL,
    grade TEXT,
    confidence REAL,
    assessment TEXT,

    -- Module scores (12 modules)
    engagement_score REAL,
    consistency_score REAL,
    dissonance_score REAL,
    baiting_score REAL,
    privilege_score REAL,
    prediction_score REAL,
    transparency_score REAL,
    follower_quality_score REAL,
    temporal_score REAL,
    linguistic_score REAL,
    accountability_score REAL,
    network_score REAL,

    -- Flags (stored as JSONB for efficient querying)
    red_flags JSONB DEFAULT '[]'::jsonb,
    green_flags JSONB DEFAULT '[]'::jsonb,

    -- Summary and detailed analysis
    summary TEXT,
    detailed_analysis JSONB DEFAULT '{}'::jsonb,

    -- Archetype classification
    archetype TEXT,
    archetype_emoji TEXT,
    archetype_one_liner TEXT,
    trust_level TEXT,

    -- Metadata
    tweets_analyzed INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- API Usage tracking (for rate limiting awareness)
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    month TEXT NOT NULL,  -- Format: YYYY-MM
    tweets_fetched INTEGER DEFAULT 0,
    api_calls INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(month)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- KOLs indexes
CREATE INDEX IF NOT EXISTS idx_kols_username ON kols(username);
CREATE INDEX IF NOT EXISTS idx_kols_twitter_id ON kols(twitter_id);
CREATE INDEX IF NOT EXISTS idx_kols_last_fetched ON kols(last_fetched_at);
CREATE INDEX IF NOT EXISTS idx_kols_latest_score ON kols(latest_score);

-- Tweets indexes
CREATE INDEX IF NOT EXISTS idx_tweets_kol_id ON tweets(kol_id);
CREATE INDEX IF NOT EXISTS idx_tweets_tweet_id ON tweets(tweet_id);
CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tweets_kol_timestamp ON tweets(kol_id, timestamp DESC);

-- Analyses indexes
CREATE INDEX IF NOT EXISTS idx_analyses_kol_id ON analyses(kol_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_grade ON analyses(grade);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
DROP TRIGGER IF EXISTS kols_updated_at ON kols;
CREATE TRIGGER kols_updated_at
    BEFORE UPDATE ON kols
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to get cached tweets count for a KOL
CREATE OR REPLACE FUNCTION get_cached_tweet_count(p_username TEXT)
RETURNS INTEGER AS $$
DECLARE
    v_count INTEGER;
BEGIN
    SELECT COUNT(*)::INTEGER INTO v_count
    FROM tweets t
    JOIN kols k ON t.kol_id = k.id
    WHERE k.username = LOWER(p_username);

    RETURN COALESCE(v_count, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to check if cache is stale (older than X hours)
CREATE OR REPLACE FUNCTION is_cache_stale(p_username TEXT, p_hours INTEGER DEFAULT 24)
RETURNS BOOLEAN AS $$
DECLARE
    v_last_fetched TIMESTAMPTZ;
BEGIN
    SELECT last_fetched_at INTO v_last_fetched
    FROM kols
    WHERE username = LOWER(p_username);

    IF v_last_fetched IS NULL THEN
        RETURN TRUE;
    END IF;

    RETURN v_last_fetched < (NOW() - (p_hours || ' hours')::INTERVAL);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for KOL leaderboard
CREATE OR REPLACE VIEW kol_leaderboard AS
SELECT
    k.username,
    k.display_name,
    k.follower_count,
    k.latest_score,
    k.latest_grade,
    a.archetype,
    a.archetype_emoji,
    a.trust_level,
    k.last_fetched_at,
    (SELECT COUNT(*) FROM tweets WHERE kol_id = k.id) as cached_tweets
FROM kols k
LEFT JOIN LATERAL (
    SELECT archetype, archetype_emoji, trust_level
    FROM analyses
    WHERE kol_id = k.id
    ORDER BY created_at DESC
    LIMIT 1
) a ON true
WHERE k.latest_score IS NOT NULL
ORDER BY k.latest_score DESC;

-- View for recent analyses
CREATE OR REPLACE VIEW recent_analyses AS
SELECT
    k.username,
    k.display_name,
    a.overall_score,
    a.grade,
    a.archetype,
    a.red_flags,
    a.green_flags,
    a.tweets_analyzed,
    a.created_at
FROM analyses a
JOIN kols k ON a.kol_id = k.id
ORDER BY a.created_at DESC;

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if needed)
-- ============================================================================

-- For public read access (if you want to make data publicly accessible)
-- ALTER TABLE kols ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE tweets ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Public read access" ON kols FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON tweets FOR SELECT USING (true);
-- CREATE POLICY "Public read access" ON analyses FOR SELECT USING (true);

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Initialize current month's API usage
INSERT INTO api_usage (month, tweets_fetched, api_calls)
VALUES (TO_CHAR(NOW(), 'YYYY-MM'), 0, 0)
ON CONFLICT (month) DO NOTHING;
