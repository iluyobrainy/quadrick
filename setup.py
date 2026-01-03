#!/usr/bin/env python
"""
Setup script for Quadrick AI Trading System
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          QUADRICK AI TRADING SYSTEM - SETUP WIZARD              ║
║                                                                  ║
║                    Mission: $15 → $100,000                      ║
║                 Powered by DeepSeek & Bybit                     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_python_version():
    """Check Python version"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required. You have {sys.version}")
        print("Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "logs",
        "data",
        "backtest",
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✓ {dir_name}/")


def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    print("This may take a few minutes...")
    
    # Check if in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("\n⚠️  WARNING: Not in a virtual environment!")
        print("It's recommended to use a virtual environment.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    try:
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install TA-Lib (special handling for Windows)
        if sys.platform == "win32":
            print("\n📊 Installing TA-Lib for Windows...")
            print("Note: If this fails, you may need to install TA-Lib manually:")
            print("Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            
            # Try to install with pip
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib"])
            except:
                print("⚠️  TA-Lib installation failed. Please install manually.")
                print("After downloading the .whl file, run:")
                print("pip install path/to/TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl")
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_environment():
    """Setup environment configuration"""
    print("\n⚙️  Setting up environment configuration...")
    
    # Copy example env if .env doesn't exist
    env_path = Path(".env")
    example_path = Path("config/env.example")
    
    if not env_path.exists() and example_path.exists():
        shutil.copy(example_path, env_path)
        print("✅ Created .env file from template")
    elif env_path.exists():
        print("ℹ️  .env file already exists")
    else:
        print("⚠️  No env.example found")
    
    return True


def configure_api_keys():
    """Interactive API key configuration"""
    print("\n🔑 API Key Configuration")
    print("-" * 50)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    print("\nYou need to configure the following API keys:")
    print("\n1. BYBIT API:")
    print("   - Go to: https://testnet.bybit.com/ (for testing)")
    print("   - Or: https://www.bybit.com/ (for live trading)")
    print("   - Create API key with trading permissions (NO withdrawal)")
    
    print("\n2. DEEPSEEK API:")
    print("   - Go to: https://platform.deepseek.com/")
    print("   - Sign up and get API key")
    print("   - Alternative: Use OpenAI GPT-4 API")
    
    configure_now = input("\nDo you want to configure API keys now? (y/n): ").lower()
    
    if configure_now == 'y':
        # Bybit configuration
        print("\n--- Bybit Configuration ---")
        use_testnet = input("Use Bybit Testnet? (recommended for testing) (y/n): ").lower() == 'y'
        bybit_key = input("Enter Bybit API Key: ").strip()
        bybit_secret = input("Enter Bybit API Secret: ").strip()
        
        # DeepSeek configuration
        print("\n--- LLM Configuration ---")
        llm_choice = input("Use DeepSeek (d) or OpenAI (o)? [d/o]: ").lower()
        
        if llm_choice == 'd':
            deepseek_key = input("Enter DeepSeek API Key: ").strip()
            llm_provider = "deepseek"
            
            # Update env content
            env_content = env_content.replace("your_api_key_here", bybit_key)
            env_content = env_content.replace("your_api_secret_here", bybit_secret)
            env_content = env_content.replace("BYBIT_TESTNET=true", f"BYBIT_TESTNET={'true' if use_testnet else 'false'}")
            env_content = env_content.replace("your_deepseek_api_key", deepseek_key)
            env_content = env_content.replace("LLM_PROVIDER=deepseek", f"LLM_PROVIDER=deepseek")
            
        else:
            openai_key = input("Enter OpenAI API Key: ").strip()
            llm_provider = "openai"
            
            # Update env content
            env_content = env_content.replace("your_api_key_here", bybit_key)
            env_content = env_content.replace("your_api_secret_here", bybit_secret)
            env_content = env_content.replace("BYBIT_TESTNET=true", f"BYBIT_TESTNET={'true' if use_testnet else 'false'}")
            env_content = env_content.replace("your_openai_api_key", openai_key)
            env_content = env_content.replace("LLM_PROVIDER=deepseek", f"LLM_PROVIDER=openai")
        
        # Save updated .env
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print("✅ API keys configured successfully")
        
    else:
        print("\n📝 Please edit .env file manually with your API keys before running the bot.")
    
    return True


def test_setup():
    """Test the setup"""
    print("\n🧪 Testing setup...")
    
    try:
        # Test imports
        print("Testing imports...")
        from config.settings import Settings
        from src.exchange.bybit_client import BybitClient
        from src.llm.deepseek_client import DeepSeekClient
        
        print("✅ All imports successful")
        
        # Try loading settings
        print("Testing configuration...")
        try:
            settings = Settings.load()
            print("✅ Configuration loaded")
        except Exception as e:
            print(f"⚠️  Configuration error: {e}")
            print("Make sure to configure your .env file")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Configure API Keys (if not done):")
    print("   Edit .env file with your Bybit and DeepSeek API keys")
    
    print("\n2. Fund Your Account:")
    print("   - For testing: Use Bybit Testnet (free test funds)")
    print("   - For live: Deposit $15+ to your Bybit account")
    
    print("\n3. Run the Bot:")
    print("   python main.py")
    
    print("\n4. Monitor Performance:")
    print("   - Check logs/ folder for detailed logs")
    print("   - Watch console for real-time updates")
    
    print("\n⚠️  IMPORTANT WARNINGS:")
    print("   - Start with TESTNET to verify everything works")
    print("   - This bot can lose money - only risk what you can afford to lose")
    print("   - Monitor closely during first few trades")
    print("   - The $15 → $100k goal is EXTREMELY ambitious")
    
    print("\n📚 Documentation:")
    print("   - Bybit API: https://bybit-exchange.github.io/docs/")
    print("   - DeepSeek: https://platform.deepseek.com/api-docs")
    
    print("\n💡 Tips:")
    print("   - Run 'python main.py' in a screen/tmux session for 24/7 operation")
    print("   - Set up Telegram alerts for important events")
    print("   - Review logs daily to understand bot decisions")
    
    print("\nGood luck with your trading! 🚀")


def main():
    """Main setup flow"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Setup incomplete due to dependency issues")
        print("Please fix the issues and run setup again.")
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Configure API keys
    configure_api_keys()
    
    # Test setup
    test_setup()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
