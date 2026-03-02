#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=$(which python3)
VENV_PATH="$SCRIPT_DIR/.venv"

echo "📋 Stock Analysis Cron Setup"
echo "=============================="

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

CRON_WRAPPER="$SCRIPT_DIR/run_analysis.sh"
cat > "$CRON_WRAPPER" << 'EOF'
#!/bin/bash
# Wrapper script for cron to run daily stock analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Load environment variables from .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Activate virtual environment and run analysis
source "$VENV_PATH/bin/activate"
cd "$SCRIPT_DIR"

# Run the main analysis and log output
python3 main.py >> "$SCRIPT_DIR/cron_logs/daily_analysis_$(date +%Y-%m-%d).log" 2>&1

echo "✅ Analysis completed at $(date)" >> "$SCRIPT_DIR/cron_logs/daily_analysis_$(date +%Y-%m-%d).log"
EOF

chmod +x "$CRON_WRAPPER"
echo "✅ Created wrapper script: $CRON_WRAPPER"

mkdir -p "$SCRIPT_DIR/cron_logs"
echo "✅ Created logs directory: $SCRIPT_DIR/cron_logs"

echo ""
echo "📅 Cron Job Setup Instructions"
echo "==============================="
echo ""
echo "Copy one of the following commands to your crontab:"
echo ""
echo "1️⃣  Daily at 9:00 AM:"
echo "   0 9 * * * $CRON_WRAPPER"
echo ""
echo "2️⃣  Daily at 4:00 PM (after market close):"
echo "   0 16 * * * $CRON_WRAPPER"
echo ""
echo "3️⃣  Every 4 hours (business hours):"
echo "   0 9,13,17,21 * * * $CRON_WRAPPER"
echo ""
echo "─────────────────────────────────────────"
echo ""
echo "To edit crontab, run:"
echo "   crontab -e"
echo ""
echo "To view existing crontab:"
echo "   crontab -l"
echo ""
echo "To remove all cron jobs:"
echo "   crontab -r"
echo ""
echo "Logs will be saved to:"
echo "   $SCRIPT_DIR/cron_logs/daily_analysis_YYYY-MM-DD.log"
echo ""
echo "✅ Setup complete! Review .env file to ensure all API keys are set:"
cat << 'ENVFILE'
   - OPENROUTER_API_KEY (required)
   - ALPHA_VANTAGE_API_KEY (optional, for technical indicators)
   - DISCORD_BOT_TOKEN (optional, for Discord bot)
   - DISCORD_CHANNEL_ID (optional, for Discord bot)
   - SLACK_BOT_TOKEN (optional, for Slack bot)
   - SLACK_CHANNEL_ID (optional, for Slack bot)
ENVFILE
