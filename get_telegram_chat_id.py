#!/usr/bin/env python
"""
Get your Telegram Chat ID - Required for bot notifications
"""
import asyncio
import aiohttp
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def get_telegram_chat_id():
    """Get Telegram chat ID from bot updates"""
    
    print("=" * 60)
    print("     TELEGRAM CHAT ID FINDER")
    print("=" * 60)
    
    # Read bot token from .env
    bot_token = None
    env_file = Path(".env")
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    bot_token = line.split("=", 1)[1].strip()
                    break
    
    if not bot_token:
        print("\n❌ Bot token not found in .env file")
        bot_token = input("Enter your Telegram bot token: ").strip()
    
    print(f"\n✅ Using bot token: {bot_token[:20]}...")
    
    base_url = f"https://api.telegram.org/bot{bot_token}"
    
    print("\n📱 INSTRUCTIONS:")
    print("1. Open Telegram")
    print("2. Search for your bot")
    print("3. Send /start to your bot")
    print("4. Come back here and press Enter")
    
    input("\nPress Enter after sending /start to your bot...")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Get updates
            url = f"{base_url}/getUpdates"
            async with session.get(url) as response:
                data = await response.json()
                
                if not data.get("ok"):
                    print(f"\n❌ Error: {data.get('description', 'Unknown error')}")
                    return
                
                if not data.get("result"):
                    print("\n❌ No messages found!")
                    print("Make sure you sent /start to your bot")
                    return
                
                # Get all unique chat IDs
                chat_ids = set()
                usernames = {}
                
                for update in data["result"]:
                    if "message" in update:
                        chat_id = update["message"]["chat"]["id"]
                        chat_ids.add(chat_id)
                        
                        # Get username if available
                        if "from" in update["message"]:
                            user = update["message"]["from"]
                            username = user.get("username", user.get("first_name", "Unknown"))
                            usernames[chat_id] = username
                
                if not chat_ids:
                    print("\n❌ No chat IDs found")
                    return
                
                print("\n✅ FOUND CHAT IDs:")
                print("-" * 40)
                
                for chat_id in chat_ids:
                    username = usernames.get(chat_id, "Unknown")
                    print(f"Chat ID: {chat_id}")
                    print(f"Username: {username}")
                    print("-" * 40)
                
                # If only one chat ID, update .env
                if len(chat_ids) == 1:
                    chat_id = list(chat_ids)[0]
                    print(f"\n✅ Your Chat ID is: {chat_id}")
                    
                    # Update .env file
                    if env_file.exists():
                        update = input("\nUpdate .env file with this Chat ID? (y/n): ").lower()
                        if update == 'y':
                            with open(env_file, 'r') as f:
                                lines = f.readlines()
                            
                            updated = False
                            for i, line in enumerate(lines):
                                if line.startswith("TELEGRAM_CHAT_ID="):
                                    lines[i] = f"TELEGRAM_CHAT_ID={chat_id}\n"
                                    updated = True
                                    break
                            
                            if not updated:
                                lines.append(f"\nTELEGRAM_CHAT_ID={chat_id}\n")
                            
                            with open(env_file, 'w') as f:
                                f.writelines(lines)
                            
                            print("✅ .env file updated!")
                    
                    print("\n🎉 Setup complete! Your bot is ready to send notifications.")
                    
                    # Send test message
                    test = input("\nSend a test message? (y/n): ").lower()
                    if test == 'y':
                        test_url = f"{base_url}/sendMessage"
                        payload = {
                            "chat_id": chat_id,
                            "text": "🎉 *Quadrick Bot Connected!*\n\nYou will receive trading notifications here.\n\nMission: $15 → $100,000 🚀",
                            "parse_mode": "Markdown"
                        }
                        
                        async with session.post(test_url, json=payload) as response:
                            result = await response.json()
                            if result.get("ok"):
                                print("✅ Test message sent! Check your Telegram.")
                            else:
                                print(f"❌ Failed to send test: {result}")
                
                else:
                    print(f"\n⚠️  Multiple chat IDs found. Choose the correct one and add to .env:")
                    print("TELEGRAM_CHAT_ID=<your_chat_id>")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPossible issues:")
            print("1. Invalid bot token")
            print("2. Network connection issue")
            print("3. Bot not created properly")

if __name__ == "__main__":
    print("\n🤖 Quadrick Trading Bot - Telegram Setup\n")
    
    try:
        asyncio.run(get_telegram_chat_id())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n" + "=" * 60)
