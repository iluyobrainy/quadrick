-- ============================================
-- Quadrick RAG Vector Migration (4D → 8D)
-- ============================================
-- This migration expands the trade_memories vector column from 4 to 8 dimensions
-- for enhanced similarity matching in the RAG system.
--
-- Run this migration in your Supabase SQL Editor BEFORE starting the bot
-- after updating the codebase.
--
-- ⚠️ WARNING: This will clear existing trade memories. Back them up first if needed.
-- ============================================

-- Step 1: Backup existing data (optional but recommended)
-- CREATE TABLE trade_memories_backup AS SELECT * FROM trade_memories;

-- Step 2: Drop the existing table (if using pgvector, alter column dimension is not supported)
DROP TABLE IF EXISTS trade_memories;

-- Step 3: Recreate the table with 8-dimension vector
CREATE TABLE trade_memories (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    symbol TEXT NOT NULL,
    strategy TEXT,
    pnl_pct FLOAT,
    win BOOLEAN,
    
    -- 8-dimension vector for enhanced RAG similarity
    -- Dimensions:
    -- [0] RSI / 100 (0-1)
    -- [1] Trend (-1 to 1)
    -- [2] Volatility ATR% (0-1)
    -- [3] Momentum MACD (-1 to 1)
    -- [4] ADX / 100 (0-1) - NEW
    -- [5] Volume Ratio (0-1) - NEW
    -- [6] BB Position (-1 to 1) - NEW
    -- [7] Time of Day (0-1) - NEW
    market_vector vector(8),
    
    -- Store full context for debugging
    market_context_json JSONB,
    
    -- Indexing
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 4: Create the vector similarity index (IVFFlat for performance)
CREATE INDEX idx_trade_memories_vector ON trade_memories 
USING ivfflat (market_vector vector_cosine_ops)
WITH (lists = 100);

-- Step 5: Create additional indexes
CREATE INDEX idx_trade_memories_symbol ON trade_memories(symbol);
CREATE INDEX idx_trade_memories_strategy ON trade_memories(strategy);
CREATE INDEX idx_trade_memories_win ON trade_memories(win);
CREATE INDEX idx_trade_memories_timestamp ON trade_memories(timestamp DESC);

-- Step 6: Drop existing function first (required to change return type)
DROP FUNCTION IF EXISTS match_trade_memories(vector, double precision, integer);
DROP FUNCTION IF EXISTS match_trade_memories(vector(4), double precision, integer);

-- Step 7: Create the match_trade_memories function for 8-dimension vectors
CREATE OR REPLACE FUNCTION match_trade_memories(
    query_embedding vector(8),
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
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tm.id,
        tm.symbol,
        tm.strategy,
        tm.pnl_pct,
        tm.win,
        1 - (tm.market_vector <=> query_embedding) as similarity,
        tm.market_context_json
    FROM trade_memories tm
    WHERE 1 - (tm.market_vector <=> query_embedding) > match_threshold
    ORDER BY tm.market_vector <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Step 7: Grant permissions (adjust role name if needed)
GRANT ALL ON trade_memories TO authenticated;
GRANT ALL ON trade_memories TO service_role;
GRANT EXECUTE ON FUNCTION match_trade_memories TO authenticated;
GRANT EXECUTE ON FUNCTION match_trade_memories TO service_role;

-- ============================================
-- Migration Complete!
-- ============================================
-- The trade_memories table now uses 8-dimension vectors:
-- - RSI, Trend, Volatility, Momentum (original 4)
-- - ADX, Volume, BB Position, Time of Day (new 4)
--
-- This enables more precise similarity matching for RAG-based
-- trade recommendations.
-- ============================================
