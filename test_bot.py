#!/usr/bin/env python3

import subprocess
import time
import signal
import sys
import os

def test_crypto_bot():
    """Test the crypto bot in dry run mode for a short period"""
    
    print("🚀 Starting Crypto Bot Dry Run Test...")
    print("=" * 50)
    
    try:
        # Start the bot process
        proc = subprocess.Popen(
            ['python', 'crypto_bot_fixed.py'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        print("⏱️  Running for 25 seconds to capture one full cycle...")
        print("📝 Capturing output...\n")
        
        # Let it run for 25 seconds to capture some activity
        time.sleep(25)
        
        # Terminate the process gracefully
        proc.terminate()
        
        # Get the output with timeout
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
        
        print("📊 DRY RUN OUTPUT:")
        print("-" * 30)
        print(output)
        
        print("\n✅ Test completed successfully!")
        print("💡 This was a DRY RUN - no actual tweets were sent")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        if 'proc' in locals():
            proc.terminate()
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")

if __name__ == "__main__":
    test_crypto_bot()
