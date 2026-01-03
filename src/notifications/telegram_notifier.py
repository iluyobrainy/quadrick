"""
Telegram Notification Module - Sends alerts and updates to Telegram
"""
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger


class TelegramNotifier:
    """Telegram notification handler"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("Telegram notifier initialized")
    
    async def initialize(self):
        """Initialize async session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Test connection
        try:
            await self.get_chat_id()
            logger.info("✅ Telegram connection verified")
            return True
        except Exception as e:
            logger.error(f"❌ Telegram connection failed: {e}")
            return False
    
    async def get_chat_id(self):
        """Get updates to find chat ID if not set"""
        if self.chat_id and self.chat_id != "YOUR_CHAT_ID_HERE":
            return self.chat_id
        
        try:
            url = f"{self.base_url}/getUpdates"
            async with self.session.get(url) as response:
                data = await response.json()
                
                if data["ok"] and data["result"]:
                    # Get the chat ID from the latest message
                    latest_update = data["result"][-1]
                    if "message" in latest_update:
                        chat_id = latest_update["message"]["chat"]["id"]
                        logger.info(f"Found chat ID: {chat_id}")
                        logger.info("Add this to your .env file: TELEGRAM_CHAT_ID={chat_id}")
                        return str(chat_id)
                
                logger.warning("No messages found. Send /start to your bot first!")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get chat ID: {e}")
            return None
    
    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Send a message to Telegram
        
        Args:
            text: Message text
            parse_mode: HTML or Markdown
            disable_notification: Silent notification
        
        Returns:
            True if sent successfully
        """
        if not self.session:
            await self.initialize()
        
        # Auto-detect chat ID if needed
        if self.chat_id == "YOUR_CHAT_ID_HERE":
            detected_id = await self.get_chat_id()
            if detected_id:
                self.chat_id = detected_id
            else:
                logger.error("Cannot send message - no chat ID available")
                return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }
            
            async with self.session.post(url, json=payload) as response:
                data = await response.json()
                
                if data["ok"]:
                    logger.debug("Message sent to Telegram")
                    return True
                else:
                    logger.error(f"Telegram error: {data}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def send_trade_alert(
        self,
        action: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        risk_pct: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reason: Optional[str] = None
    ):
        """Send trade alert"""
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        action_emoji = "📈" if action == "OPEN" else "📊"
        
        message = f"""
{action_emoji} <b>{action} POSITION</b>

{emoji} <b>{symbol}</b> - {side.upper()}
💰 Size: {size:.4f}
💵 Entry: ${price:,.2f}
⚠️ Risk: {risk_pct:.1f}%
        """
        
        if stop_loss:
            message += f"\n🛑 Stop Loss: ${stop_loss:,.2f}"
        if take_profit:
            message += f"\n🎯 Take Profit: ${take_profit:,.2f}"
        if reason:
            message += f"\n\n📝 Reason: {reason}"
        
        message += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        await self.send_message(message)
    
    async def send_pnl_update(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        balance: float,
        closed: bool = False
    ):
        """Send P&L update"""
        if pnl >= 0:
            emoji = "💚"
            status = "PROFIT"
        else:
            emoji = "💔"
            status = "LOSS"
        
        action = "CLOSED" if closed else "UPDATE"
        
        message = f"""
{emoji} <b>POSITION {action}</b>

📊 {symbol}
💰 PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%)
💼 Balance: ${balance:.2f}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        
        await self.send_message(message)
    
    async def send_milestone_alert(
        self,
        current_balance: float,
        milestone_reached: float,
        next_milestone: float,
        time_taken_hours: Optional[float] = None
    ):
        """Send milestone achievement alert"""
        message = f"""
🎉🎯 <b>MILESTONE REACHED!</b> 🎯🎉

✅ Achieved: ${milestone_reached:.0f}
💰 Current Balance: ${current_balance:.2f}
🎯 Next Target: ${next_milestone:.0f}
        """
        
        if time_taken_hours:
            days = int(time_taken_hours / 24)
            hours = int(time_taken_hours % 24)
            message += f"\n⏱️ Time Taken: {days}d {hours}h"
        
        progress = ((current_balance - milestone_reached) / (next_milestone - milestone_reached)) * 100
        message += f"\n📊 Progress to Next: {progress:.1f}%"
        
        message += f"\n\n🚀 Keep going! Next target: ${next_milestone:.0f}"
        
        await self.send_message(message)
    
    async def send_daily_summary(
        self,
        trades_count: int,
        wins: int,
        losses: int,
        total_pnl: float,
        starting_balance: float,
        ending_balance: float,
        best_trade: Optional[Dict[str, Any]] = None,
        worst_trade: Optional[Dict[str, Any]] = None
    ):
        """Send daily trading summary"""
        win_rate = (wins / trades_count * 100) if trades_count > 0 else 0
        daily_return = ((ending_balance - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0
        
        if total_pnl >= 0:
            emoji = "💚"
            result = "PROFITABLE"
        else:
            emoji = "💔"
            result = "LOSS"
        
        message = f"""
📊 <b>DAILY SUMMARY</b> {emoji}

📈 Result: {result} DAY
💰 P&L: ${total_pnl:+.2f} ({daily_return:+.1f}%)

📊 Statistics:
• Trades: {trades_count}
• Wins: {wins} | Losses: {losses}
• Win Rate: {win_rate:.1f}%

💼 Balance:
• Start: ${starting_balance:.2f}
• End: ${ending_balance:.2f}
        """
        
        if best_trade:
            message += f"\n\n🏆 Best Trade: {best_trade['symbol']} +${best_trade['pnl']:.2f}"
        if worst_trade:
            message += f"\n😢 Worst Trade: {worst_trade['symbol']} -${abs(worst_trade['pnl']):.2f}"
        
        message += f"\n\n⏰ {datetime.utcnow().strftime('%Y-%m-%d')} UTC"
        
        await self.send_message(message)
    
    async def send_warning(self, title: str, message: str):
        """Send warning message"""
        text = f"""
⚠️ <b>WARNING: {title}</b> ⚠️

{message}

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        await self.send_message(text)
    
    async def send_error(self, title: str, error: str):
        """Send error message"""
        text = f"""
🚨 <b>ERROR: {title}</b> 🚨

{error}

Please check the logs for details.

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        await self.send_message(text, disable_notification=False)
    
    async def send_startup_message(
        self,
        balance: float,
        mode: str,
        milestone_current: float,
        milestone_next: float
    ):
        """Send bot startup message"""
        message = f"""
🚀 <b>QUADRICK BOT STARTED</b> 🚀

🤖 Mode: {mode}
💰 Balance: ${balance:.2f}
🎯 Milestone: ${milestone_current:.0f} → ${milestone_next:.0f}

Mission: $15 → $100,000

Let's trade! 💪

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        await self.send_message(message)
    
    async def send_shutdown_message(
        self,
        balance: float,
        total_pnl: float,
        trades_today: int
    ):
        """Send bot shutdown message"""
        message = f"""
🛑 <b>BOT SHUTDOWN</b> 🛑

💰 Final Balance: ${balance:.2f}
📊 Session P&L: ${total_pnl:+.2f}
📈 Trades Today: {trades_today}

Bot stopped successfully.

⏰ {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """
        await self.send_message(message)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
            self.session = None
