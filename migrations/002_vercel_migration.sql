-- Migration 002: Vercel/Serverless State Storage

-- Bot Status Table (Single row state)
create table if not exists bot_status (
    id int primary key default 1,
    mode text default 'active',
    trading_allowed boolean default true,
    total_balance numeric default 0,
    available_balance numeric default 0,
    unrealized_pnl numeric default 0,
    daily_pnl numeric default 0,
    daily_pnl_pct numeric default 0,
    updated_at timestamptz default now(),
    constraint single_row check (id = 1)
);

-- Active Positions Table
create table if not exists active_positions (
    symbol text primary key,
    side text not null,
    size numeric not null,
    entry_price numeric not null,
    mark_price numeric not null,
    unrealized_pnl numeric not null,
    leverage numeric,
    stop_loss numeric,
    take_profit numeric,
    updated_at timestamptz default now()
);

-- System Logs Table (Streaming)
create table if not exists system_logs (
    id bigint primary key generated always as identity,
    timestamp timestamptz default now(),
    level text not null,
    message text not null
);

-- Decisions and Trades already exist from 001
