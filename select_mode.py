#!/usr/bin/env python
"""
Trading Mode Selector - Choose Normal or Scalper Quad mode
"""
import sys
import os
from pathlib import Path


def print_banner():
    """Print mode selection banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          QUADRICK AI TRADING SYSTEM - MODE SELECTOR              ║
║                                                                  ║
║                    Mission: $15 → $100,000                      ║
║                 Powered by DeepSeek & Bybit                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

SELECT YOUR TRADING MODE:

┌──────────────────────────────────────────────────────────────────┐
│  1. NORMAL MODE (Conservative Swing Trading)                     │
├──────────────────────────────────────────────────────────────────┤
│  • Multi-timeframe analysis (1m through 1w)                      │
│  • Decision every 2 minutes                                      │
│  • Hold trades: 30 minutes - 24 hours                            │
│  • Target profit: 2-5% per trade                                 │
│  • Trades per day: 1-5                                           │
│  • Risk: 15-18% per trade                                        │
│  • Leverage: 10-15x                                              │
│  • Best for: Patient traders, trending markets                  │
│  • $15 → $50: ~7-14 days                                         │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  2. SCALPER QUAD ⚡💰 (ULTRA-AGGRESSIVE FAST MONEY!)            │
├──────────────────────────────────────────────────────────────────┤
│  • Lightning-fast 1m/5m timeframes ONLY                          │
│  • Decision every 15-30 seconds                                  │
│  • Hold trades: 1-5 minutes (FAST IN/OUT!)                       │
│  • Target profit: 0.5-1.5% per trade                             │
│  • Trades per day: 50-150+ (MAXIMUM FREQUENCY!)                  │
│  • Risk: 20% per trade (BIG POSITIONS!)                          │
│  • Leverage: 30-50x (AMPLIFY TINY MOVES!)                        │
│  • Multiple positions: Up to 3 simultaneous                      │
│  • Best for: Aggressive scalpers, volatile markets               │
│  • $15 → $50: ~6-48 hours (RAPID!)                               │
│  • Volume spike hunting, momentum surfing                        │
│  • 🔥 HIGH RISK, HIGH REWARD 🔥                                  │
└──────────────────────────────────────────────────────────────────┘

⚠️  WARNING: Scalper Quad is EXTREMELY AGGRESSIVE!
   - Uses 20% risk per trade (can lose fast)
   - 30-50x leverage (high liquidation risk)
   - Requires constant market volatility
   - API costs higher (~$3-8/day)
   - Best for experienced traders who can monitor

💡 RECOMMENDATION:
   - New users: Start with NORMAL MODE
   - Experienced: Try SCALPER QUAD
   - Your current balance: $16.70
"""
    print(banner)


def get_mode_selection():
    """Get user's mode selection"""
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                print("\n✅ NORMAL MODE selected")
                print("   Starting conservative swing trading system...")
                return "normal"
            
            elif choice == "2":
                print("\n⚡ SCALPER QUAD MODE selected")
                print("   WARNING: Ultra-aggressive high-frequency trading!")
                
                confirm = input("\n   Are you sure? This uses 20% risk + 30-50x leverage (y/n): ").strip().lower()
                
                if confirm == 'y':
                    print("\n🔥 SCALPER QUAD MODE ACTIVATED!")
                    print("   Optimizing for maximum trade frequency...")
                    print("   Target: 50-150 trades per day")
                    print("   Hold time: 1-5 minutes")
                    print("   LET'S MAKE FAST MONEY! 💰⚡")
                    return "scalper"
                else:
                    print("\n   Returning to menu...")
                    continue
            
            else:
                print("\n❌ Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\n\n❌ Mode selection cancelled")
            sys.exit(1)


def save_mode_selection(mode: str):
    """Save selected mode to config file"""
    env_path = Path(".env")
    
    if env_path.exists():
        # Read current .env
        with open(env_path, 'r') as f:
            lines = f.readlines()
        
        # Update or add TRADING_MODE
        mode_updated = False
        for i, line in enumerate(lines):
            if line.startswith("TRADING_MODE="):
                lines[i] = f"TRADING_MODE={mode}\n"
                mode_updated = True
                break
        
        if not mode_updated:
            lines.append(f"\nTRADING_MODE={mode}\n")
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        print(f"\n✅ Mode saved to .env: TRADING_MODE={mode}")
    else:
        print("\n⚠️  .env file not found, mode not saved")


if __name__ == "__main__":
    print_banner()
    mode = get_mode_selection()
    save_mode_selection(mode)
    
    print("\n" + "="*70)
    print("🚀 Starting trading bot in", mode.upper(), "mode...")
    print("="*70)
    print()
    
    # Run main.py with selected mode
    import subprocess
    subprocess.run([sys.executable, "main.py", "--mode", mode])
