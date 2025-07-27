#!/usr/bin/env python3
"""
Test script to verify API connections and basic functionality
"""

import os
from dotenv import load_dotenv
from loguru import logger

def test_environment_variables():
    """Test that all required environment variables are set"""
    logger.info("Testing environment variables...")
    
    required_vars = [
        'X_API_KEY', 'X_API_SECRET', 'X_ACCESS_TOKEN', 
        'X_ACCESS_TOKEN_SECRET', 'X_BEARER_TOKEN', 'GEMINI_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing or not configured: {', '.join(missing_vars)}")
        return False
    else:
        logger.success("✅ All environment variables are set")
        return True

def test_x_api():
    """Test X API connection"""
    logger.info("Testing X API connection...")
    
    try:
        from x_client import XClient
        x_client = XClient()
        logger.success("✅ X API client initialized successfully")
        
        # Test API permissions first
        if not x_client.test_api_permissions():
            logger.error("❌ API permissions test failed")
            return False
        
        # Test getting user info for a public account
        test_accounts = ["elonmusk", "naval", "jack", "verge"]
        
        for account in test_accounts:
            tweets = x_client.get_user_recent_tweets(account, count=5)
            if tweets:
                logger.success(f"✅ X API connection working - able to fetch tweets from @{account}")
                return True
        
        logger.warning("⚠️ X API connected but couldn't fetch tweets from any test account")
        return False
            
    except Exception as e:
        logger.error(f"❌ X API connection failed: {str(e)}")
        return False

def test_gemini_api():
    """Test Gemini API connection"""
    logger.info("Testing Gemini API connection...")
    
    try:
        from gemini_client import GeminiClient
        gemini_client = GeminiClient()
        logger.success("✅ Gemini API client initialized successfully")
        
        # Test generating a simple response
        test_tweet = {
            'text': 'Hello world! This is a test tweet.',
            'author_username': 'test_user',
            'id': '123456789'
        }
        
        from config import AccountConfig, TonalityType, ContentType
        test_config = AccountConfig(
            username="test_user",
            tonality=TonalityType.FRIENDLY,
            content_types=[ContentType.INSIGHT]
        )
        
        reply = gemini_client.generate_reply(test_tweet, test_config)
        if reply:
            logger.success(f"✅ Gemini API working - generated reply: '{reply}'")
            return True
        else:
            logger.warning("⚠️ Gemini API connected but couldn't generate reply")
            return False
            
    except Exception as e:
        logger.error(f"❌ Gemini API connection failed: {str(e)}")
        return False

def test_bot_config():
    """Test bot configuration loading"""
    logger.info("Testing bot configuration...")
    
    try:
        from config import BotConfig
        config = BotConfig.load_example()
        logger.success(f"✅ Bot configuration loaded with {len(config.monitored_accounts)} accounts")
        return True
    except Exception as e:
        logger.error(f"❌ Bot configuration failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🤖 KOL Terminal Bot - Setup Test")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("X API Connection", test_x_api),
        ("Gemini API Connection", test_gemini_api),
        ("Bot Configuration", test_bot_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your bot is ready to run.")
        print("   Run 'python3 bot.py' to start the bot in dry-run mode.")
    else:
        print("\n⚠️  Some tests failed. Please check the setup guide (SETUP.md)")
        print("   and ensure all API credentials are properly configured.")

if __name__ == "__main__":
    main()
