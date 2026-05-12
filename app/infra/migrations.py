from dataclasses import dataclass
from threading import RLock

import psycopg2


@dataclass(frozen=True)
class SqlMigration:
    migration_id: str
    statements: tuple[str, ...]


APP_DATABASE_MIGRATIONS: tuple[SqlMigration, ...] = (
    SqlMigration(
        migration_id="001_create_prediction_logs",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id BIGSERIAL PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT,
                run_id TEXT,
                request_id TEXT,
                target_type TEXT NOT NULL,
                target_id TEXT,
                qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
                predicted_at TIMESTAMPTZ NOT NULL,
                target_timestamp TIMESTAMPTZ NOT NULL,
                predicted_value JSONB NOT NULL,
                actual_value JSONB,
                error_value DOUBLE PRECISION,
                error_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                input_features JSONB NOT NULL DEFAULT '{}'::jsonb,
                output_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_predicted_at
            ON prediction_logs (model_name, predicted_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_logs_request_id
            ON prediction_logs (request_id)
            """,
        ),
    ),
    SqlMigration(
        migration_id="002_create_training_jobs",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                context JSONB NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                result JSONB,
                error_message TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created_at
            ON training_jobs (status, created_at DESC)
            """,
        ),
    ),
    SqlMigration(
        migration_id="003_create_mlops_events",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS mlops_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                occurred_at TIMESTAMPTZ NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mlops_events_type_occurred_at
            ON mlops_events (event_type, occurred_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mlops_events_occurred_at
            ON mlops_events (occurred_at DESC)
            """,
        ),
    ),
)

_migration_lock = RLock()
_migrated_database_urls: set[str] = set()


def run_app_database_migrations(database_url: str) -> None:
    if database_url in _migrated_database_urls:
        return
    with _migration_lock:
        if database_url in _migrated_database_urls:
            return
        with psycopg2.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_schema_migrations (
                        migration_id TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                for migration in APP_DATABASE_MIGRATIONS:
                    cursor.execute(
                        "SELECT 1 FROM app_schema_migrations WHERE migration_id = %s",
                        (migration.migration_id,),
                    )
                    if cursor.fetchone() is not None:
                        continue
                    for statement in migration.statements:
                        cursor.execute(statement)
                    cursor.execute(
                        "INSERT INTO app_schema_migrations (migration_id) VALUES (%s)",
                        (migration.migration_id,),
                    )
        _migrated_database_urls.add(database_url)
