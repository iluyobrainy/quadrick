-- Migration 003: Expand Decisions Table for Rich Logging
-- Execute this in your Supabase SQL Editor

ALTER TABLE decisions 
ADD COLUMN IF NOT EXISTS account_balance numeric,
ADD COLUMN IF NOT EXISTS open_positions int,
ADD COLUMN IF NOT EXISTS daily_pnl numeric,
ADD COLUMN IF NOT EXISTS current_milestone text,
ADD COLUMN IF NOT EXISTS side text,
ADD COLUMN IF NOT EXISTS risk_pct numeric,
ADD COLUMN IF NOT EXISTS leverage numeric,
ADD COLUMN IF NOT EXISTS entry_price_target numeric,
ADD COLUMN IF NOT EXISTS stop_loss numeric,
ADD COLUMN IF NOT EXISTS take_profit_1 numeric,
ADD COLUMN IF NOT EXISTS take_profit_2 numeric,
ADD COLUMN IF NOT EXISTS strategy_tag text,
ADD COLUMN IF NOT EXISTS confidence_score numeric,
ADD COLUMN IF NOT EXISTS risk_management jsonb,
ADD COLUMN IF NOT EXISTS model_version text,
ADD COLUMN IF NOT EXISTS processing_time_ms numeric,
ADD COLUMN IF NOT EXISTS executed boolean default false,
ADD COLUMN IF NOT EXISTS error_message text,
ADD COLUMN IF NOT EXISTS error_traceback text,
ADD COLUMN IF NOT EXISTS timestamp timestamptz default now();

-- Update existing column if needed
COMMENT ON COLUMN decisions.reasoning IS 'AI thought process or trade justification';
