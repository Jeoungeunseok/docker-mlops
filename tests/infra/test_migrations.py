from app.infra import migrations


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.params: list[object] | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def execute(self, statement: str, params: list[object] | tuple[object, ...] | None = None) -> None:
        self.statements.append(statement)
        self.params = list(params) if params is not None else None

    def fetchone(self) -> None:
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def cursor(self) -> FakeCursor:
        return self._cursor


def test_run_app_database_migrations_applies_known_migrations(monkeypatch) -> None:
    cursor = FakeCursor()
    monkeypatch.setattr(migrations, "_migrated_database_urls", set())
    monkeypatch.setattr(migrations.psycopg2, "connect", lambda database_url: FakeConnection(cursor))

    migrations.run_app_database_migrations("postgresql://app:app_pass@localhost:5434/app")

    joined_sql = "\n".join(cursor.statements)
    assert "CREATE TABLE IF NOT EXISTS app_schema_migrations" in joined_sql
    assert "CREATE TABLE IF NOT EXISTS prediction_logs" in joined_sql
    assert "CREATE TABLE IF NOT EXISTS training_jobs" in joined_sql
    assert "CREATE TABLE IF NOT EXISTS mlops_events" in joined_sql
    assert joined_sql.count("INSERT INTO app_schema_migrations") == len(migrations.APP_DATABASE_MIGRATIONS)


def test_run_app_database_migrations_runs_once_per_database_url(monkeypatch) -> None:
    cursor = FakeCursor()
    monkeypatch.setattr(migrations, "_migrated_database_urls", set())
    monkeypatch.setattr(migrations.psycopg2, "connect", lambda database_url: FakeConnection(cursor))

    migrations.run_app_database_migrations("postgresql://app:app_pass@localhost:5434/app")
    statement_count = len(cursor.statements)
    migrations.run_app_database_migrations("postgresql://app:app_pass@localhost:5434/app")

    assert len(cursor.statements) == statement_count
