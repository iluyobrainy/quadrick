# 📊 SUPABASE DATABASE SETUP INSTRUCTIONS

This guide will help you set up the database for your Quadrick AI Trading System.

## ✅ Your Supabase Configuration (Already in `.env`)

Your Supabase credentials are already configured:
- **Project URL**: https://srltdknslcxyxbjdrtaf.supabase.co
- **Project ID**: srltdknslcxyxbjdrtaf
- **Anon Key**: (Configured in `.env`)

## 📋 Step-by-Step Setup

### Step 1: Access Supabase Dashboard

1. Go to: https://supabase.com/dashboard
2. Sign in with your account
3. Select your project: `srltdknslcxyxbjdrtaf`

### Step 2: Open SQL Editor

1. In the left sidebar, click **SQL Editor**
2. Click **+ New Query** button

### Step 3: Run the Schema

1. Open the file `database/schema.sql` in this directory
2. Copy **ALL** the contents (Ctrl+A, Ctrl+C)
3. Paste into the Supabase SQL Editor
4. Click **Run** button (or press Ctrl+Enter)

**This will create:**
- ✅ 8 Tables (trades, decisions, positions, etc.)
- ✅ All indexes for fast queries
- ✅ Triggers for automatic timestamps
- ✅ Views for easy data querying
- ✅ Security policies

### Step 4: Verify Tables Were Created

1. In the left sidebar, click **Table Editor**
2. You should see these tables:
   - `trades` - All trading activity
   - `decisions` - AI decision logs
   - `positions` - Open positions
   - `account_snapshots` - Balance history
   - `market_data` - Market analysis
   - `performance_metrics` - Stats
   - `alerts` - System alerts
   - `system_logs` - Error logs

### Step 5: Test the Connection

Run the connection test:
```bash
python test_connection.py
```

This will verify:
- ✅ Supabase connection works
- ✅ Tables are accessible
- ✅ Database logging is enabled

## 📊 Database Tables Explained

### 1. **trades** - Complete Trade History
Records every trade you make:
- Entry/exit prices
- P&L (profit/loss)
- Risk percentage
- AI reasoning
- Strategy used

### 2. **decisions** - AI Decision Log
Every decision DeepSeek makes:
- What it wanted to trade
- Why it made that decision
- Risk assessment
- Whether it was executed

### 3. **positions** - Open Positions Tracker
Real-time tracking of open trades:
- Current prices
- Unrealized P&L
- Stop loss / Take profit levels

### 4. **account_snapshots** - Balance History
Snapshots of your account every few minutes:
- Total equity
- Available balance
- Daily P&L
- Milestone progress

### 5. **market_data** - Technical Analysis Data
Market analysis results:
- Price data (OHLCV)
- Technical indicators
- Patterns detected
- Trend analysis

### 6. **performance_metrics** - Performance Stats
Daily/weekly/monthly statistics:
- Win rate
- Total P&L
- Best/worst trades
- Strategy performance

### 7. **alerts** - System Notifications
All alerts sent:
- Trade alerts
- Milestone achievements
- Warnings
- Errors

### 8. **system_logs** - System Events
Technical logs:
- Errors
- Warnings
- Debug info

## 🔍 Useful Queries

Once your bot is running, you can query data in Supabase:

### View Recent Trades
```sql
SELECT * FROM v_recent_trades 
ORDER BY timestamp DESC 
LIMIT 10;
```

### View Current Open Positions
```sql
SELECT * FROM v_open_positions;
```

### View Daily Performance
```sql
SELECT * FROM v_daily_performance 
ORDER BY trade_date DESC;
```

### View Strategy Performance
```sql
SELECT * FROM v_strategy_performance 
ORDER BY total_pnl DESC;
```

### View Account Balance History
```sql
SELECT 
  timestamp,
  total_equity,
  total_pnl,
  total_pnl_pct,
  open_positions
FROM account_snapshots
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
```

### View AI Decisions
```sql
SELECT 
  timestamp,
  decision_type,
  symbol,
  side,
  risk_pct,
  confidence_score,
  executed,
  execution_result
FROM decisions
ORDER BY timestamp DESC
LIMIT 20;
```

### Calculate Win Rate
```sql
SELECT 
  COUNT(*) as total_trades,
  SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
  ROUND(
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / 
    COUNT(*)::NUMERIC * 100, 
    2
  ) as win_rate_pct
FROM trades
WHERE status = 'closed';
```

## 📱 Access Your Data

### Via Supabase Dashboard:
1. Go to https://supabase.com/dashboard
2. Select your project
3. Click **Table Editor** to browse data
4. Click **SQL Editor** to run queries

### Via API (Optional):
Your bot automatically saves data, but you can also:
- Build a custom dashboard
- Export data for analysis
- Create reports

## 🔐 Security Notes

Your database uses:
- **Row Level Security (RLS)** - Controls who can access data
- **Service Role** - Bot has full access
- **Anon Key** - Safe for client-side use

**Keep your Service Role key private!** (Not in this system, we use Anon key)

## 🛠️ Troubleshooting

### Error: "relation does not exist"
**Problem**: Tables weren't created
**Solution**: Run the schema.sql again in SQL Editor

### Error: "permission denied"
**Problem**: RLS blocking access
**Solution**: Make sure policies are created (they're in schema.sql)

### Bot says "Database logging disabled"
**Problem**: Supabase credentials not in .env
**Solution**: Check .env file has:
```
DATABASE_PROVIDER=supabase
SUPABASE_URL=https://srltdknslcxyxbjdrtaf.supabase.co
SUPABASE_ANON_KEY=your_key_here
```

### Can't connect to Supabase
**Problem**: Network or credentials issue
**Solution**: 
1. Check internet connection
2. Verify project URL is correct
3. Regenerate anon key if needed

## 📈 What Gets Logged

Your bot will automatically save:
- ✅ Every trade (open/close)
- ✅ Every AI decision
- ✅ Position updates
- ✅ Account balance (every 2 minutes)
- ✅ Performance metrics
- ✅ System alerts
- ✅ Errors and warnings

## 💡 Benefits of Database Logging

1. **Track Performance**: See exactly how you're doing
2. **Analyze Strategies**: Which strategies work best?
3. **Debug Issues**: Review what happened when
4. **Build Dashboard**: Create custom visualizations
5. **Historical Analysis**: Learn from past trades
6. **Compliance**: Keep records for taxes

## 🎯 Next Steps

After setting up the database:

1. ✅ Run the schema (done above)
2. ✅ Test connection: `python test_connection.py`
3. ✅ Start the bot: `python main.py`
4. 📊 Watch data populate in Supabase dashboard
5. 📈 Run queries to analyze performance

---

**Database setup complete!** Your bot will now log all activity to Supabase. 🚀

*Any questions? Check the Supabase docs: https://supabase.com/docs*
