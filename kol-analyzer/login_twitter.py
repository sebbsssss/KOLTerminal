#!/usr/bin/env python3
"""Script to manually login to Twitter and save cookies for the scraper."""

import asyncio
import json
import os
from pathlib import Path

async def main():
    from playwright.async_api import async_playwright

    cookies_path = Path("data/cookies.json")
    cookies_path.parent.mkdir(parents=True, exist_ok=True)

    # Find Chrome user data directory
    home = os.path.expanduser("~")
    chrome_user_data = f"{home}/Library/Application Support/Google/Chrome"

    print("\n" + "=" * 60)
    print("Twitter Login - KOL Analyzer")
    print("=" * 60)
    print("\nUsing your existing Chrome profile...")
    print("If you're logged into Twitter in Chrome, this should work.")
    print("=" * 60 + "\n")

    async with async_playwright() as p:
        # Launch with persistent context (uses existing Chrome data)
        context = await p.chromium.launch_persistent_context(
            user_data_dir=f"{home}/.kol-chrome-profile",
            headless=False,
            channel="chrome",  # Use installed Chrome
        )
        page = context.pages[0] if context.pages else await context.new_page()

        await page.goto("https://twitter.com/login")

        print("Waiting for you to log in...")
        print("(Browser will auto-detect when you're logged in)\n")

        # Wait for login by detecting redirect to home or navigation away from login
        try:
            # Wait up to 5 minutes for login
            for _ in range(300):  # 300 seconds = 5 minutes
                await asyncio.sleep(1)
                url = page.url
                # Check if we've left the login page and are on home/main feed
                if "/home" in url or (("twitter.com" in url or "x.com" in url) and "/login" not in url and "/i/flow" not in url):
                    # Give it a moment to fully load
                    await asyncio.sleep(2)
                    if "/login" not in page.url and "/i/flow" not in page.url:
                        print("Login detected!")
                        break
            else:
                print("Timeout waiting for login. Saving cookies anyway...")
        except Exception as e:
            print(f"Note: {e}")

        # Save cookies
        cookies = await context.cookies()
        with open(cookies_path, 'w') as f:
            json.dump(cookies, f, indent=2)

        print(f"\nCookies saved to {cookies_path}")
        print("You can now use the analyzer with real Twitter data!")

        await asyncio.sleep(1)
        await context.close()

if __name__ == "__main__":
    asyncio.run(main())
