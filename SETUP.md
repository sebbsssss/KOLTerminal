# X Auto-Reply Bot Setup Guide

## 1. Get X (Twitter) API Credentials

### Step 1: Apply for Twitter Developer Account
1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Sign in with your Twitter account
3. Apply for a developer account (may take 1-2 days for approval)
4. Once approved, create a new app

### Step 2: Create Twitter App
1. In the Twitter Developer Dashboard, click "Create App"
2. Fill in app details:
   - App name: "KOL Terminal Bot" (or any name you prefer)
   - Description: "Automated reply bot for engaging with key accounts"
   - Website: Your website or GitHub repo URL
3. Create the app

### Step 3: Generate API Keys
1. Go to your app's "Keys and Tokens" tab
2. Generate/copy the following:
   - **API Key** (Consumer Key)
   - **API Secret** (Consumer Secret)
   - **Bearer Token**
3. Under "Access Token and Secret", click "Generate"
4. Copy the:
   - **Access Token**
   - **Access Token Secret**

⚠️ **Important**: Make sure your app has **Read and Write** permissions in the app settings!

## 2. Get Google Gemini API Key

### Step 1: Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## 3. Configure Your Bot

### Step 1: Update Environment Variables
Edit the `.env` file and replace the placeholder values:

```bash
# X (Twitter) API Credentials
X_API_KEY=your_actual_api_key_here
X_API_SECRET=your_actual_api_secret_here
X_ACCESS_TOKEN=your_actual_access_token_here
X_ACCESS_TOKEN_SECRET=your_actual_access_token_secret_here
X_BEARER_TOKEN=your_actual_bearer_token_here

# Google Gemini API
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Bot Configuration
BOT_USERNAME=your_twitter_username_here
DRY_RUN=true
LOG_LEVEL=INFO
CHECK_INTERVAL_MINUTES=5
```

### Step 2: Configure Target Accounts
The bot will create a `bot_config.json` file with example accounts. You can customize this to add your target accounts.

## 4. Test Your Setup

### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate
```

### Step 2: Test API Connections
```bash
python3 -c "
from x_client import XClient
from gemini_client import GeminiClient
from dotenv import load_dotenv
load_dotenv()

print('Testing X API connection...')
try:
    x_client = XClient()
    print('✅ X API connection successful!')
except Exception as e:
    print(f'❌ X API connection failed: {e}')

print('Testing Gemini API connection...')
try:
    gemini_client = GeminiClient()
    print('✅ Gemini API connection successful!')
except Exception as e:
    print(f'❌ Gemini API connection failed: {e}')
"
```

### Step 3: Run Bot in Dry-Run Mode
```bash
python3 bot.py
```

The bot will run in dry-run mode by default, which means it will generate replies but not actually post them. Check the logs to see what replies it would generate.

## 5. Customize Your Bot

### Account Configuration
Edit `bot_config.json` to configure which accounts to monitor and how to respond to them:

- **tonality**: friendly, professional, casual, humorous, supportive, analytical, enthusiastic
- **content_types**: question, insight, agreement, constructive_criticism, appreciation, resource_sharing
- **reply_probability**: 0.0 to 1.0 (chance of replying to each tweet)
- **custom_instructions**: Specific instructions for this account
- **avoid_keywords**: Keywords to avoid in tweets
- **preferred_hashtags**: Hashtags to occasionally add

### Going Live
When you're ready to go live:
1. Set `DRY_RUN=false` in your `.env` file
2. Restart the bot
3. Monitor the logs carefully

## 6. Monitoring and Maintenance

- Check logs in the `logs/` directory
- Monitor rate limits (default: 10 replies per hour)
- Regularly review and update your account configurations
- Keep your API keys secure and never commit them to version control

## Troubleshooting

### Common Issues
1. **"Missing X API credentials"**: Check your .env file and ensure all X API variables are set
2. **"GEMINI_API_KEY environment variable is required"**: Add your Gemini API key to .env
3. **Rate limit errors**: Reduce CHECK_INTERVAL_MINUTES or max_replies_per_hour
4. **Authentication errors**: Verify your API keys and ensure your Twitter app has proper permissions

### Getting Help
- Check the logs in `logs/bot.log`
- Review the bot configuration in `bot_config.json`
- Test individual components using the test script above
