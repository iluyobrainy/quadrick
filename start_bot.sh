#!/bin/bash
# ========================================
# Quadrick AI Trading Bot - Unix/Linux Starter
# ========================================

echo "========================================"
echo "   QUADRICK AI TRADING SYSTEM"
echo "   Mission: \$15 → \$100,000"
echo "========================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.11+ required. You have Python $python_version${NC}"
    exit 1
fi

# Check/create virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    if [ -f "config/env.example" ]; then
        echo "Creating .env file from template..."
        cp config/env.example .env
        echo
        echo -e "${YELLOW}======================================="
        echo " IMPORTANT: Edit .env file with your"
        echo " API keys before continuing!"
        echo -e "=======================================${NC}"
        echo "Press Enter to continue..."
        read
    fi
fi

# Function to show menu
show_menu() {
    echo
    echo "Select an option:"
    echo "  1. Run Setup Wizard"
    echo "  2. Test Connections"
    echo "  3. Start Trading Bot"
    echo "  4. Start Bot (Background with screen)"
    echo "  5. Start Bot (Background with nohup)"
    echo "  6. View Logs"
    echo "  7. Stop Background Bot"
    echo "  8. Exit"
    echo
    read -p "Enter choice (1-8): " choice
}

# Main menu loop
while true; do
    show_menu
    
    case $choice in
        1)
            echo "Running setup wizard..."
            python3 setup.py
            echo "Press Enter to continue..."
            read
            ;;
            
        2)
            echo "Testing connections..."
            python3 test_connection.py
            echo "Press Enter to continue..."
            read
            ;;
            
        3)
            echo -e "${GREEN}Starting trading bot...${NC}"
            echo "Press Ctrl+C to stop"
            echo
            python3 main.py
            echo "Press Enter to continue..."
            read
            ;;
            
        4)
            # Check if screen is installed
            if ! command -v screen &> /dev/null; then
                echo -e "${RED}screen not installed. Install with: sudo apt install screen${NC}"
            else
                echo "Starting bot in screen session..."
                screen -dmS quadrick_trading python3 main.py
                echo -e "${GREEN}Bot started in screen session 'quadrick_trading'${NC}"
                echo "Commands:"
                echo "  View: screen -r quadrick_trading"
                echo "  Detach: Ctrl+A, D"
                echo "  Stop: screen -X -S quadrick_trading quit"
            fi
            echo "Press Enter to continue..."
            read
            ;;
            
        5)
            echo "Starting bot with nohup..."
            nohup python3 main.py > output.log 2>&1 &
            echo $! > bot.pid
            echo -e "${GREEN}Bot started in background (PID: $(cat bot.pid))${NC}"
            echo "Logs: tail -f output.log"
            echo "Press Enter to continue..."
            read
            ;;
            
        6)
            if [ -f "logs/trading.log" ]; then
                echo "Showing last 50 lines of log..."
                echo "========================================"
                tail -50 logs/trading.log
                echo "========================================"
                echo
                echo "For live updates: tail -f logs/trading.log"
            else
                echo "No log file found yet."
            fi
            echo "Press Enter to continue..."
            read
            ;;
            
        7)
            echo "Stopping background bot..."
            
            # Try to stop screen session
            if screen -list | grep -q "quadrick_trading"; then
                screen -X -S quadrick_trading quit
                echo "Screen session stopped"
            fi
            
            # Try to stop nohup process
            if [ -f "bot.pid" ]; then
                pid=$(cat bot.pid)
                if ps -p $pid > /dev/null; then
                    kill $pid
                    echo "Process $pid stopped"
                fi
                rm bot.pid
            fi
            
            echo -e "${GREEN}Background bot stopped${NC}"
            echo "Press Enter to continue..."
            read
            ;;
            
        8)
            echo "Goodbye!"
            exit 0
            ;;
            
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac
done
