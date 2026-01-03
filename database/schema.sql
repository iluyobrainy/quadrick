-- ================================================
-- QUADRICK AI TRADING SYSTEM - SUPABASE SCHEMA
-- ================================================
-- This file contains all table definitions for the trading system
-- Run this in your Supabase SQL Editor to create the database

-- ================================================
-- 1. TRADES TABLE - Record all trading activity
-- ================================================
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Trade Details
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL, -- Buy/Sell
    order_type VARCHAR(20) NOT NULL, -- Market/Limit
    category VARCHAR(20) NOT NULL DEFAULT 'linear',
    
    -- Position Info
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 2) NOT NULL,
    exit_price DECIMAL(20, 2),
    stop_loss DECIMAL(20, 2),
    take_profit DECIMAL(20, 2),
    
    -- Risk & Leverage
    leverage INTEGER NOT NULL,
    risk_pct DECIMAL(5, 2) NOT NULL,
    position_value DECIMAL(20, 2) NOT NULL,
    margin_used DECIMAL(20, 2),
    
    -- P&L
    realized_pnl DECIMAL(20, 2),
    realized_pnl_pct DECIMAL(10, 4),
    fees DECIMAL(20, 4),
    
    -- Trade Management
    status VARCHAR(20) NOT NULL DEFAULT 'open', -- open, closed, cancelled
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    duration_minutes INTEGER,
    
    -- Strategy & Decision
    strategy_tag VARCHAR(100),
    ai_confidence DECIMAL(5, 4),
    ai_reasoning TEXT,
    decision_id UUID,
    
    -- Account State
    balance_before DECIMAL(20, 2),
    balance_after DECIMAL(20, 2),
    
    -- Exchange Info
    exchange_order_id VARCHAR(100),
    exchange_response JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for trades
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_strategy ON trades(strategy_tag);
CREATE INDEX idx_trades_decision_id ON trades(decision_id);

-- ================================================
-- 2. DECISIONS TABLE - Record all AI decisions
-- ================================================
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Decision Type
    decision_type VARCHAR(50) NOT NULL, -- open_position, close_position, hold, wait
    
    -- Context Summary
    account_balance DECIMAL(20, 2) NOT NULL,
    open_positions INTEGER NOT NULL,
    daily_pnl DECIMAL(20, 2),
    current_milestone VARCHAR(50),
    
    -- Decision Details (if trading)
    symbol VARCHAR(50),
    side VARCHAR(10),
    risk_pct DECIMAL(5, 2),
    leverage INTEGER,
    entry_price_target DECIMAL(20, 2),
    stop_loss DECIMAL(20, 2),
    take_profit_1 DECIMAL(20, 2),
    take_profit_2 DECIMAL(20, 2),
    
    -- AI Analysis
    strategy_tag VARCHAR(100),
    confidence_score DECIMAL(5, 4),
    reasoning JSONB,
    risk_management JSONB,
    market_context JSONB,
    
    -- Execution
    executed BOOLEAN DEFAULT false,
    execution_result VARCHAR(50), -- success, rejected, failed
    rejection_reason TEXT,
    
    -- Performance Tracking
    model_version VARCHAR(50),
    processing_time_ms INTEGER,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for decisions
CREATE INDEX idx_decisions_timestamp ON decisions(timestamp DESC);
CREATE INDEX idx_decisions_type ON decisions(decision_type);
CREATE INDEX idx_decisions_executed ON decisions(executed);
CREATE INDEX idx_decisions_symbol ON decisions(symbol);

-- ================================================
-- 3. POSITIONS TABLE - Track open positions
-- ================================================
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Position Details
    symbol VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 2) NOT NULL,
    current_price DECIMAL(20, 2),
    mark_price DECIMAL(20, 2),
    
    -- Risk Management
    leverage INTEGER NOT NULL,
    stop_loss DECIMAL(20, 2),
    take_profit DECIMAL(20, 2),
    liquidation_price DECIMAL(20, 2),
    
    -- P&L
    unrealized_pnl DECIMAL(20, 2),
    unrealized_pnl_pct DECIMAL(10, 4),
    realized_pnl DECIMAL(20, 2),
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    opened_at TIMESTAMPTZ NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Links
    trade_id UUID REFERENCES trades(id),
    decision_id UUID REFERENCES decisions(id),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for positions
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);

-- ================================================
-- 4. ACCOUNT_SNAPSHOTS TABLE - Track account over time
-- ================================================
CREATE TABLE IF NOT EXISTS account_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Balance Info
    total_equity DECIMAL(20, 2) NOT NULL,
    available_balance DECIMAL(20, 2) NOT NULL,
    used_margin DECIMAL(20, 2) NOT NULL,
    unrealized_pnl DECIMAL(20, 2) NOT NULL,
    realized_pnl DECIMAL(20, 2) NOT NULL,
    
    -- Performance Metrics
    daily_pnl DECIMAL(20, 2),
    daily_pnl_pct DECIMAL(10, 4),
    total_pnl DECIMAL(20, 2),
    total_pnl_pct DECIMAL(10, 4),
    
    -- Trading Stats
    open_positions INTEGER NOT NULL DEFAULT 0,
    total_trades_today INTEGER NOT NULL DEFAULT 0,
    win_rate_today DECIMAL(5, 2),
    
    -- Milestone Progress
    current_milestone VARCHAR(50),
    milestone_progress_pct DECIMAL(5, 2),
    
    -- Risk Metrics
    current_drawdown_pct DECIMAL(5, 2),
    max_drawdown_today_pct DECIMAL(5, 2),
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for account snapshots
CREATE INDEX idx_snapshots_timestamp ON account_snapshots(timestamp DESC);

-- ================================================
-- 5. MARKET_DATA TABLE - Store analyzed market data
-- ================================================
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Symbol Info
    symbol VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Price Data
    open DECIMAL(20, 2) NOT NULL,
    high DECIMAL(20, 2) NOT NULL,
    low DECIMAL(20, 2) NOT NULL,
    close DECIMAL(20, 2) NOT NULL,
    volume DECIMAL(30, 8) NOT NULL,
    
    -- Technical Indicators
    rsi DECIMAL(5, 2),
    macd DECIMAL(20, 8),
    macd_signal DECIMAL(20, 8),
    atr DECIMAL(20, 2),
    bb_upper DECIMAL(20, 2),
    bb_lower DECIMAL(20, 2),
    
    -- Analysis Results
    trend VARCHAR(20),
    market_regime VARCHAR(50),
    patterns JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for market data
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timeframe ON market_data(timeframe);

-- ================================================
-- 6. PERFORMANCE_METRICS TABLE - Daily/Weekly/Monthly stats
-- ================================================
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_type VARCHAR(20) NOT NULL, -- daily, weekly, monthly
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    
    -- Trading Stats
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    losing_trades INTEGER NOT NULL,
    win_rate DECIMAL(5, 2) NOT NULL,
    
    -- P&L
    total_pnl DECIMAL(20, 2) NOT NULL,
    total_pnl_pct DECIMAL(10, 4) NOT NULL,
    avg_win DECIMAL(20, 2),
    avg_loss DECIMAL(20, 2),
    largest_win DECIMAL(20, 2),
    largest_loss DECIMAL(20, 2),
    
    -- Risk Metrics
    max_drawdown DECIMAL(5, 2),
    sharpe_ratio DECIMAL(10, 4),
    profit_factor DECIMAL(10, 4),
    
    -- Strategy Performance
    best_strategy VARCHAR(100),
    best_strategy_pnl DECIMAL(20, 2),
    worst_strategy VARCHAR(100),
    worst_strategy_pnl DECIMAL(20, 2),
    
    -- Account Growth
    starting_balance DECIMAL(20, 2) NOT NULL,
    ending_balance DECIMAL(20, 2) NOT NULL,
    balance_change_pct DECIMAL(10, 4) NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance metrics
CREATE INDEX idx_performance_period ON performance_metrics(period_type, period_start DESC);

-- ================================================
-- 7. ALERTS TABLE - System alerts and notifications
-- ================================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Alert Details
    alert_type VARCHAR(50) NOT NULL, -- trade, milestone, warning, error
    severity VARCHAR(20) NOT NULL, -- info, warning, critical
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    -- Context
    symbol VARCHAR(50),
    trade_id UUID REFERENCES trades(id),
    decision_id UUID REFERENCES decisions(id),
    
    -- Notification Status
    sent_telegram BOOLEAN DEFAULT false,
    sent_email BOOLEAN DEFAULT false,
    sent_discord BOOLEAN DEFAULT false,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for alerts
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);

-- ================================================
-- 8. SYSTEM_LOGS TABLE - System events and errors
-- ================================================
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Log Details
    log_level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    module VARCHAR(100) NOT NULL,
    function VARCHAR(100),
    message TEXT NOT NULL,
    
    -- Error Details (if applicable)
    error_type VARCHAR(100),
    error_traceback TEXT,
    
    -- Context
    context JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for system logs
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX idx_system_logs_level ON system_logs(log_level);
CREATE INDEX idx_system_logs_module ON system_logs(module);

-- ================================================
-- FUNCTIONS AND TRIGGERS
-- ================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to tables
CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================================
-- VIEWS FOR EASY QUERYING
-- ================================================

-- View: Recent trades with P&L
CREATE OR REPLACE VIEW v_recent_trades AS
SELECT 
    t.trade_id,
    t.timestamp,
    t.symbol,
    t.side,
    t.size,
    t.entry_price,
    t.exit_price,
    t.realized_pnl,
    t.realized_pnl_pct,
    t.strategy_tag,
    t.status,
    t.duration_minutes
FROM trades t
ORDER BY t.timestamp DESC
LIMIT 100;

-- View: Current open positions
CREATE OR REPLACE VIEW v_open_positions AS
SELECT 
    p.symbol,
    p.side,
    p.size,
    p.entry_price,
    p.current_price,
    p.unrealized_pnl,
    p.unrealized_pnl_pct,
    p.stop_loss,
    p.take_profit,
    p.opened_at,
    EXTRACT(EPOCH FROM (NOW() - p.opened_at))/60 as minutes_open
FROM positions p
WHERE p.status = 'open'
ORDER BY p.opened_at DESC;

-- View: Daily performance summary
CREATE OR REPLACE VIEW v_daily_performance AS
SELECT 
    DATE(timestamp) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) as win_rate,
    SUM(realized_pnl) as total_pnl,
    MAX(realized_pnl) as best_trade,
    MIN(realized_pnl) as worst_trade,
    AVG(realized_pnl) as avg_pnl
FROM trades
WHERE status = 'closed'
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

-- View: Strategy performance
CREATE OR REPLACE VIEW v_strategy_performance AS
SELECT 
    strategy_tag,
    COUNT(*) as total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) as win_rate,
    SUM(realized_pnl) as total_pnl,
    AVG(realized_pnl) as avg_pnl,
    MAX(realized_pnl) as best_trade,
    MIN(realized_pnl) as worst_trade
FROM trades
WHERE status = 'closed' AND strategy_tag IS NOT NULL
GROUP BY strategy_tag
ORDER BY total_pnl DESC;

-- ================================================
-- ROW LEVEL SECURITY (Optional - for multi-user)
-- ================================================

-- Enable RLS on all tables
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE account_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;

-- Create policies to allow service role full access
CREATE POLICY "Service role can do everything" ON trades
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON decisions
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON positions
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON account_snapshots
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON market_data
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON performance_metrics
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON alerts
    FOR ALL USING (true);

CREATE POLICY "Service role can do everything" ON system_logs
    FOR ALL USING (true);

-- ================================================
-- SETUP COMPLETE!
-- ================================================
-- All tables, indexes, triggers, and views have been created.
-- Your Quadrick AI Trading System database is ready!
-- ================================================
