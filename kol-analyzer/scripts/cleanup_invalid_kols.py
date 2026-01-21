#!/usr/bin/env python3
"""Delete invalid KOL entries from the database."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

INVALID_HANDLES = [
    "hello",
    "kapo",
    "spider",
    "btcfdugbq6mcygimsqta4bvtstuknd2khbqzge3eory",
    "ھل تستطيع تحليل خالتى النفسية  والعصبية من خلال الكلام",
    "مرحيا",
    "4mcero",
    "letterbomb",
    "degensealsa",
    "elon",
    "crypto_ser",
    "mr punk",
    "lady millionaire",
    "hku",
    "alguém",
    "j0shcrypto",
    "ga-ke",
    "ga_ke",
    "loopier",
    "darky1l",
    "darkyll",
    "onchainvaction",
    "ignaanft",
    "0xuberm",
    "midcurveportal",
    "midcurvrportal",
    "enzoknol",
]

def main():
    print("=" * 60)
    print("CLEANUP INVALID KOLS")
    print("=" * 60)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    deleted_count = 0
    failed = []

    for handle in INVALID_HANDLES:
        try:
            # First check if the KOL exists
            result = supabase.table("kols").select("id, username").eq("username", handle.lower()).execute()

            if result.data:
                kol_id = result.data[0]["id"]
                print(f"Deleting @{handle} (ID: {kol_id})...")

                # Delete related records first (foreign key constraints)
                supabase.table("tweets").delete().eq("kol_id", kol_id).execute()
                supabase.table("mentions").delete().eq("kol_id", kol_id).execute()
                supabase.table("analyses").delete().eq("kol_id", kol_id).execute()

                # Delete the KOL
                supabase.table("kols").delete().eq("id", kol_id).execute()

                deleted_count += 1
                print(f"  ✓ Deleted")
            else:
                print(f"Skipping @{handle} - not found in database")

        except Exception as e:
            print(f"  ✗ Failed to delete @{handle}: {e}")
            failed.append(handle)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Deleted: {deleted_count}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed handles: {', '.join(failed)}")
    print("Done!")

if __name__ == "__main__":
    main()
