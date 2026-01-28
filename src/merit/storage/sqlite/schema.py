"""SQLite schema definitions for Merit storage."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    start_time          TEXT NOT NULL,
    end_time            TEXT,
    total_duration_ms   REAL NOT NULL,

    passed              INTEGER NOT NULL,
    failed              INTEGER NOT NULL,
    errors              INTEGER NOT NULL,
    skipped             INTEGER NOT NULL,
    xfailed             INTEGER NOT NULL,
    xpassed             INTEGER NOT NULL,
    total               INTEGER NOT NULL,
    stopped_early       INTEGER NOT NULL,

    environment_json    TEXT
);

CREATE TABLE IF NOT EXISTS test_executions (
    execution_id        TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_id           TEXT REFERENCES test_executions(execution_id) ON DELETE CASCADE,

    test_name           TEXT NOT NULL,
    file_path           TEXT,
    class_name          TEXT,
    case_id             TEXT,
    id_suffix           TEXT,
    trace_id            TEXT,
    tags_json           TEXT,
    skip_reason         TEXT,
    xfail_reason        TEXT,

    status              TEXT NOT NULL,
    duration_ms         REAL NOT NULL,
    error_message       TEXT,
    error_traceback     TEXT
);

CREATE INDEX IF NOT EXISTS idx_tests_run ON test_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_tests_parent ON test_executions(parent_id);
CREATE INDEX IF NOT EXISTS idx_tests_name ON test_executions(test_name);
CREATE INDEX IF NOT EXISTS idx_tests_status ON test_executions(status);
CREATE INDEX IF NOT EXISTS idx_tests_trace ON test_executions(trace_id);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_execution_id   TEXT REFERENCES test_executions(execution_id) ON DELETE CASCADE,

    name                TEXT NOT NULL,
    scope               TEXT NOT NULL,
    value               REAL,
    value_json          TEXT,

    first_recorded_at   TEXT,
    last_recorded_at    TEXT,
    collected_from_merits_json      TEXT,
    collected_from_resources_json   TEXT,
    collected_from_cases_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_execution ON metrics(test_execution_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);

CREATE TABLE IF NOT EXISTS assertions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    test_execution_id   TEXT REFERENCES test_executions(execution_id) ON DELETE CASCADE,
    metric_id           INTEGER REFERENCES metrics(metric_id) ON DELETE SET NULL,

    expression_repr     TEXT NOT NULL,
    passed              INTEGER NOT NULL,
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS predicates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    assertion_id        INTEGER REFERENCES assertions(id) ON DELETE CASCADE,

    predicate_name      TEXT,
    actual              TEXT,
    reference           TEXT,
    strict              INTEGER,
    confidence          REAL,
    value               INTEGER,
    message             TEXT
);

CREATE INDEX IF NOT EXISTS idx_assertions_run ON assertions(run_id);
CREATE INDEX IF NOT EXISTS idx_assertions_execution ON assertions(test_execution_id);
CREATE INDEX IF NOT EXISTS idx_assertions_metric ON assertions(metric_id);
CREATE INDEX IF NOT EXISTS idx_predicates_assertion ON predicates(assertion_id);
CREATE INDEX IF NOT EXISTS idx_predicates_run ON predicates(run_id);
CREATE INDEX IF NOT EXISTS idx_predicates_name ON predicates(predicate_name);
"""
