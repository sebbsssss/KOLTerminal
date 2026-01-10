from .database import Database

# Supabase client - optional, requires supabase package
try:
    from .supabase_client import SupabaseDatabase, IncrementalFetcher
    SUPABASE_AVAILABLE = True
except ImportError:
    SupabaseDatabase = None
    IncrementalFetcher = None
    SUPABASE_AVAILABLE = False

__all__ = ['Database', 'SupabaseDatabase', 'IncrementalFetcher', 'SUPABASE_AVAILABLE']
