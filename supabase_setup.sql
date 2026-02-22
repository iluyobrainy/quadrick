-- Migration 004: Full trading-schema reset (canonical Railway backend model)
-- WARNING: This migration drops legacy Quadrick trading tables.
-- Run database/backup_supabase.py before applying this migration in production.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop old objects first (views/functions/tables)
DROP FUNCTION IF EXISTS match_trade_memories(vector, double precision, integer);
DROP FUNCTION IF EXISTS match_trade_memories(vector(4), double precision, integer);
DROP FUNCTION IF EXISTS match_trade_memories(vector(8), double precision, integer);
DROP FUNCTION IF EXISTS match_trade_memories(vector(8), float, int);

DROP VIEW IF EXISTS v_recent_trades;

DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS market_data CASCADE;
DROP TABLE IF EXISTS account_snapshots CASCADE;
DROP TABLE IF EXISTS positions CASCADE;
DROP TABLE IF EXISTS trades CASCADE;
DROP TABLE IF EXISTS decisions CASCADE;
DROP TABLE IF EXISTS trade_memories CASCADE;
DROP TABLE IF EXISTS active_contexts CASCADE;
DROP TABLE IF EXISTS strategy_learning CASCADE;
DROP TABLE IF EXISTS bot_status CASCADE;
DROP TABLE IF EXISTS active_positions CASCADE;
DROP TABLE IF EXISTS system_logs CASCADE;

-- Canonical tables used by current runtime
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL DEFAULT 'Market',
    category TEXT NOT NULL DEFAULT 'linear',
    size NUMERIC(24, 8),
    entry_price NUMERIC(24, 8),
    exit_price NUMERIC(24, 8),
    stop_loss NUMERIC(24, 8),
    take_profit NUMERIC(24, 8),
    leverage INTEGER,
    risk_pct NUMERIC(10, 4),
    position_value NUMERIC(24, 8),
    margin_used NUMERIC(24, 8),
    realized_pnl NUMERIC(24, 8),
    realized_pnl_pct NUMERIC(12, 6),
    fees NUMERIC(24, 8),
    status TEXT NOT NULL DEFAULT 'open',
    opened_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ,
    duration_minutes INTEGER,
    strategy_tag TEXT,
    entry_tier TEXT,
    policy_state TEXT,
    policy_key TEXT,
    ai_confidence NUMERIC(10, 6),
    ai_reasoning TEXT,
    decision_id TEXT,
    balance_before NUMERIC(24, 8),
    balance_after NUMERIC(24, 8),
    exchange_order_id TEXT,
    exchange_response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decision_id TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_type TEXT NOT NULL,
    account_balance NUMERIC(24, 8) DEFAULT 0,
    open_positions INTEGER DEFAULT 0,
    daily_pnl NUMERIC(24, 8),
    current_milestone TEXT,
    symbol TEXT,
    side TEXT,
    risk_pct NUMERIC(10, 4),
    leverage INTEGER,
    entry_price_target NUMERIC(24, 8),
    stop_loss NUMERIC(24, 8),
    take_profit_1 NUMERIC(24, 8),
    take_profit_2 NUMERIC(24, 8),
    strategy_tag TEXT,
    confidence_score NUMERIC(10, 6),
    reasoning JSONB,
    risk_management JSONB,
    market_context JSONB,
    executed BOOLEAN DEFAULT FALSE,
    execution_result TEXT,
    rejection_reason TEXT,
    model_version TEXT,
    processing_time_ms INTEGER,
    error_message TEXT,
    error_traceback TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE trade_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    strategy TEXT,
    pnl_pct FLOAT,
    win BOOLEAN,
    market_context_json JSONB,
    market_vector VECTOR(8),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE active_contexts (
    symbol TEXT PRIMARY KEY,
    context JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE strategy_learning (
    id TEXT PRIMARY KEY,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE bot_status (
    id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    mode TEXT NOT NULL DEFAULT 'initialized',
    trading_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    pause_reason TEXT,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,
    cooldown_active BOOLEAN NOT NULL DEFAULT FALSE,
    cooldown_until TIMESTAMPTZ,
    total_balance NUMERIC(24, 8) NOT NULL DEFAULT 0,
    available_balance NUMERIC(24, 8) NOT NULL DEFAULT 0,
    unrealized_pnl NUMERIC(24, 8) NOT NULL DEFAULT 0,
    daily_pnl NUMERIC(24, 8) NOT NULL DEFAULT 0,
    daily_pnl_pct NUMERIC(12, 6) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE active_positions (
    symbol TEXT PRIMARY KEY,
    side TEXT NOT NULL,
    size NUMERIC(24, 8) NOT NULL DEFAULT 0,
    entry_price NUMERIC(24, 8) NOT NULL DEFAULT 0,
    current_price NUMERIC(24, 8),
    mark_price NUMERIC(24, 8),
    pnl NUMERIC(24, 8),
    pnl_pct NUMERIC(12, 6),
    unrealized_pnl NUMERIC(24, 8),
    leverage NUMERIC(10, 2),
    stop_loss NUMERIC(24, 8),
    take_profit NUMERIC(24, 8),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE system_logs (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    module TEXT,
    function TEXT
);

-- Indexes
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_strategy ON trades(strategy_tag);

CREATE INDEX idx_decisions_timestamp ON decisions(timestamp DESC);
CREATE INDEX idx_decisions_symbol ON decisions(symbol);
CREATE INDEX idx_decisions_type ON decisions(decision_type);
CREATE INDEX idx_decisions_executed ON decisions(executed);

CREATE INDEX idx_trade_memories_vector ON trade_memories
USING ivfflat (market_vector vector_cosine_ops)
WITH (lists = 100);
CREATE INDEX idx_trade_memories_symbol ON trade_memories(symbol);
CREATE INDEX idx_trade_memories_strategy ON trade_memories(strategy);
CREATE INDEX idx_trade_memories_win ON trade_memories(win);
CREATE INDEX idx_trade_memories_timestamp ON trade_memories(timestamp DESC);

CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX idx_system_logs_level ON system_logs(level);

-- RAG similarity RPC (8D vectors)
CREATE OR REPLACE FUNCTION match_trade_memories(
    query_embedding VECTOR(8),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    symbol TEXT,
    strategy TEXT,
    pnl_pct FLOAT,
    win BOOLEAN,
    similarity FLOAT,
    market_context_json JSONB
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        tm.id,
        tm.symbol,
        tm.strategy,
        tm.pnl_pct,
        tm.win,
        1 - (tm.market_vector <=> query_embedding) AS similarity,
        tm.market_context_json
    FROM trade_memories tm
    WHERE 1 - (tm.market_vector <=> query_embedding) > match_threshold
    ORDER BY tm.market_vector <=> query_embedding
    LIMIT match_count;
$$;

-- RLS + policies (service role full access, authenticated read on dashboard tables)
DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'trades',
        'decisions',
        'trade_memories',
        'active_contexts',
        'strategy_learning',
        'bot_status',
        'active_positions',
        'system_logs'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('DROP POLICY IF EXISTS service_role_all ON %I', t);
        EXECUTE format(
            'CREATE POLICY service_role_all ON %I FOR ALL USING (auth.role() = ''service_role'') WITH CHECK (auth.role() = ''service_role'')',
            t
        );
    END LOOP;

    FOREACH t IN ARRAY ARRAY[
        'bot_status',
        'active_positions',
        'system_logs',
        'decisions',
        'trades'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS authenticated_read ON %I', t);
        EXECUTE format(
            'CREATE POLICY authenticated_read ON %I FOR SELECT USING (auth.role() IN (''authenticated'', ''service_role''))',
            t
        );
    END LOOP;
END $$;

GRANT EXECUTE ON FUNCTION match_trade_memories(VECTOR(8), FLOAT, INT) TO authenticated;
GRANT EXECUTE ON FUNCTION match_trade_memories(VECTOR(8), FLOAT, INT) TO service_role;



-- Migration 005: Seed baseline state after full reset

INSERT INTO bot_status (
    id,
    mode,
    trading_allowed,
    pause_reason,
    consecutive_losses,
    cooldown_active,
    cooldown_until,
    total_balance,
    available_balance,
    unrealized_pnl,
    daily_pnl,
    daily_pnl_pct,
    updated_at
)
VALUES (
    1,
    'initialized',
    FALSE,
    NULL,
    0,
    FALSE,
    NULL,
    0,
    0,
    0,
    0,
    0,
    NOW()
)
ON CONFLICT (id) DO UPDATE SET
    mode = EXCLUDED.mode,
    trading_allowed = EXCLUDED.trading_allowed,
    pause_reason = EXCLUDED.pause_reason,
    consecutive_losses = EXCLUDED.consecutive_losses,
    cooldown_active = EXCLUDED.cooldown_active,
    cooldown_until = EXCLUDED.cooldown_until,
    total_balance = EXCLUDED.total_balance,
    available_balance = EXCLUDED.available_balance,
    unrealized_pnl = EXCLUDED.unrealized_pnl,
    daily_pnl = EXCLUDED.daily_pnl,
    daily_pnl_pct = EXCLUDED.daily_pnl_pct,
    updated_at = EXCLUDED.updated_at;

INSERT INTO strategy_learning (id, data, updated_at)
VALUES ('global_stats', '{}'::jsonb, NOW())
ON CONFLICT (id) DO NOTHING;


