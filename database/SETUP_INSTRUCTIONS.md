# Quadrick Supabase Setup (Canonical)

This project uses one canonical trading schema aligned to runtime code in `main.py`, `src/database/supabase_client.py`, and `src/database/supabase_bridge.py`.

## Required Environment Variables

Set these in backend `.env`:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_ANON_KEY=your_anon_key
```

Runtime write access should use `SUPABASE_SERVICE_ROLE_KEY`.

## Backup Before Reset

Before changing schema, export existing data:

```bash
python database/backup_supabase.py
```

Backups are written to `database/backups/supabase_<timestamp>/` as JSON files.

## Apply Canonical Reset

Run these SQL files in Supabase SQL editor (in order):

1. `migrations/004_full_reset.sql`
2. `migrations/005_seed_state.sql`

`database/schema.sql` and `supabase_setup.sql` mirror the same canonical SQL for convenience.

## Canonical Tables

The runtime requires these tables:

- `trades`
- `decisions`
- `trade_memories`
- `active_contexts`
- `strategy_learning`
- `bot_status`
- `active_positions`
- `system_logs`

## RAG Vector Contract

- Column: `trade_memories.market_vector vector(8)`
- RPC: `match_trade_memories(query_embedding vector(8), match_threshold float, match_count int)`

Quick check:

```sql
select *
from match_trade_memories(
  '[0,0,0,0,0,0,0,0]'::vector(8),
  0.0,
  3
);
```

## RLS and Roles

- RLS is enabled on canonical tables.
- Policy `service_role_all` allows full access for `auth.role() = 'service_role'`.
- Policy `authenticated_read` allows select access on dashboard-facing tables.

## Post-Reset Validation

Run these sanity queries:

```sql
select count(*) from bot_status;
select count(*) from strategy_learning;
select table_name
from information_schema.tables
where table_schema = 'public'
order by table_name;
```

Expected seed state:

- `bot_status` contains row `id = 1`
- `strategy_learning` contains row `id = 'global_stats'`

## Notes

- Use service-role key in backend only; never expose it in frontend.
- Frontend (`quadend`) should call backend via `NEXT_PUBLIC_API_URL`.
- Backend remains Railway/VPS runtime; Vercel is frontend-only.
