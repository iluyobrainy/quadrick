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

