# Business database migrations

Alembic owns only the application tables (`tasks`, `task_events`, and
`task_artifacts`). LangGraph's `PostgresSaver.setup()` owns its checkpoint
schema independently.

Set `RISE_DATABASE_URL` to a PostgreSQL URL using psycopg 3, then run:

```bash
alembic upgrade head
```
