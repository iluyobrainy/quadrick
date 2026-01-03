#!/usr/bin/env python
"""
Quick launcher for Quadrick Trading Bot with mode selection
"""
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from select_mode import print_banner, get_mode_selection, save_mode_selection


def main():
    """Main launcher"""
    print_banner()
    
    # Get mode selection
    mode = get_mode_selection()
    
    # Save mode to .env
    save_mode_selection(mode)
    
    # Import and run main bot
    print("\n" + "="*70)
    print(f"🚀 Starting {mode.upper()} mode trading bot...")
    print("="*70)
    print()
    
    import asyncio
    from main import QuadrickTradingBot, main as run_bot
    
    # Run the bot
    asyncio.run(run_bot())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Bot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
