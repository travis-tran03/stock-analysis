-- Run once in Supabase SQL editor when using Discover + Supabase persistence.
create table if not exists recommendation_runs (
  id uuid primary key default gen_random_uuid(),
  horizon text not null,
  results_json jsonb not null,
  created_at timestamptz default now()
);

create index if not exists recommendation_runs_horizon_created
  on recommendation_runs (horizon, created_at desc);
