-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store trade execution history (standard logging)
create table if not exists trades (
  id bigint primary key generated always as identity,
  trade_id text not null,
  symbol text not null,
  side text not null,
  size numeric,
  entry_price numeric,
  pnl numeric,
  strategy_tag text,
  ai_confidence numeric,
  created_at timestamptz default now()
);

-- Create a table to store LLM decisions (standard logging)
create table if not exists decisions (
  id bigint primary key generated always as identity,
  decision_id text not null,
  decision_type text not null,
  symbol text,
  side text,
  risk_pct numeric,
  leverage numeric,
  account_balance numeric,
  open_positions int,
  daily_pnl numeric,
  current_milestone text,
  entry_price_target numeric,
  stop_loss numeric,
  take_profit_1 numeric,
  take_profit_2 numeric,
  strategy_tag text,
  confidence_score numeric,
  reasoning jsonb,
  risk_management jsonb,
  model_version text,
  processing_time_ms numeric,
  executed boolean default false,
  error_message text,
  error_traceback text,
  timestamp timestamptz default now(),
  created_at timestamptz default now()
);

-- Create the RAG Memory table
-- This stores the "Market Context" as a vector and the "Outcome" of the trade
create table if not exists trade_memories (
  id bigint primary key generated always as identity,
  timestamp timestamptz default now(),
  symbol text not null,
  strategy text not null,
  pnl_pct numeric,
  win boolean,
  market_context_json jsonb, -- Store raw context for debugging
  market_vector vector(4)    -- The 4-dimensional vector we defined in Python
);

-- Create a similarity search function
-- This allows us to find past trades with similar market conditions
create or replace function match_trade_memories (
  query_embedding vector(4),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  symbol text,
  strategy text,
  pnl_pct numeric,
  win boolean,
  similarity float
)
language plpgsql
as $$
begin
  return query(
    select
      trade_memories.id,
      trade_memories.symbol,
      trade_memories.strategy,
      trade_memories.pnl_pct,
      trade_memories.win,
      1 - (trade_memories.market_vector <=> query_embedding) as similarity
    from trade_memories
    where 1 - (trade_memories.market_vector <=> query_embedding) > match_threshold
    order by trade_memories.market_vector <=> query_embedding
    limit match_count
  );
end;
$$;
