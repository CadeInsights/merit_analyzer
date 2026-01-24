"""SQLite storage backend for Merit test runs."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from merit.assertions.base import AssertionResult
from merit.metrics_.base import CalculatedValue, MetricMetadata, MetricResult
from merit.predicates.base import PredicateResult
from merit.resources import Scope
from merit.storage.base import Store
from merit.storage.sqlite.schema import SCHEMA
from merit.testing.models.definition import MeritTestDefinition
from merit.testing.models.result import TestExecution, TestResult, TestStatus
from merit.testing.models.run import MeritRun, RunEnvironment, RunResult


DEFAULT_DB_NAME = ".merit/merit.db"
SCHEMA_VERSION = 1

RUN_INSERT_SQL = """
    INSERT INTO runs (
        run_id, start_time, end_time, total_duration_ms,
        passed, failed, errors, skipped, xfailed, xpassed,
        total, stopped_early, environment_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

EXECUTION_INSERT_SQL = """
    INSERT INTO test_executions (
        execution_id, run_id, parent_id, test_name, file_path, class_name,
        case_id, id_suffix, trace_id, tags_json, skip_reason, xfail_reason,
        status, duration_ms, error_message
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

METRIC_INSERT_SQL = """
    INSERT INTO metrics (
        run_id, test_execution_id, name, scope, value, value_json,
        first_recorded_at, last_recorded_at,
        collected_from_merits_json, collected_from_resources_json, collected_from_cases_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

ASSERTION_INSERT_SQL = """
    INSERT INTO assertions (
        run_id, test_execution_id, metric_id, expression_repr, passed,
        error_message
    ) VALUES (?, ?, ?, ?, ?, ?)
"""

PREDICATE_INSERT_SQL = """
    INSERT INTO predicates (
        predicate_id, run_id, assertion_id, case_id, predicate_name,
        merit_name, actual, reference, strict, confidence, value, message
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def find_project_root() -> Path:
    """Find project root by searching for pyproject.toml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


class SQLiteStore(Store):
    """SQLite-based storage for Merit test runs."""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = find_project_root() / DEFAULT_DB_NAME
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]
            if current_version < SCHEMA_VERSION:
                conn.executescript(SCHEMA)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def save_run(self, run: MeritRun) -> None:
        """Save a complete test run."""
        with self._connect() as conn:
            run_id = str(run.run_id)
            conn.execute(
                RUN_INSERT_SQL,
                (
                    run_id,
                    run.start_time.isoformat(),
                    run.end_time.isoformat() if run.end_time else None,
                    run.result.total_duration_ms,
                    run.result.passed,
                    run.result.failed,
                    run.result.errors,
                    run.result.skipped,
                    run.result.xfailed,
                    run.result.xpassed,
                    run.result.total,
                    int(run.result.stopped_early),
                    json.dumps(run.environment.to_dict()),
                ),
            )

            execution_rows: list[tuple[object, ...]] = []
            stack: list[tuple[TestExecution, str | None]] = [
                (execution, None) for execution in run.result.executions
            ]
            while stack:
                execution, parent_id = stack.pop()
                defn = execution.definition
                error_msg = str(execution.result.error) if execution.result.error else None
                test_name = getattr(defn, "name", str(defn))
                module_path = getattr(defn, "module_path", None)
                file_path = str(module_path) if module_path else None
                class_name = getattr(defn, "class_name", None)
                id_suffix = getattr(defn, "id_suffix", None)
                tags: set[str] = getattr(defn, "tags", set())
                skip_reason = getattr(defn, "skip_reason", None)
                xfail_reason = getattr(defn, "xfail_reason", None)
                case_id = self._parse_case_id(id_suffix)
                execution_id = str(execution.execution_id)

                execution_rows.append(
                    (
                        execution_id,
                        run_id,
                        parent_id,
                        test_name,
                        file_path,
                        class_name,
                        str(case_id) if case_id else None,
                        id_suffix,
                        execution.trace_id,
                        json.dumps(list(tags)) if tags else None,
                        skip_reason,
                        xfail_reason,
                        execution.status.value,
                        execution.duration_ms,
                        error_msg,
                    )
                )

                for sub in execution.sub_executions:
                    stack.append((sub, execution_id))

            if execution_rows:
                conn.executemany(EXECUTION_INSERT_SQL, execution_rows)

            for metric_result in run.result.metric_results:
                metric_id = self._save_metric(conn, run.run_id, metric_result)
                self._save_metric_assertions(conn, run.run_id, metric_result, metric_id)

            for execution in run.result.executions:
                self._save_assertions_for_execution(conn, run.run_id, execution)

    def _save_assertion(
        self,
        conn: sqlite3.Connection,
        run_id: UUID,
        execution_id: UUID | None,
        metric_id: int | None,
        assertion: AssertionResult,
    ) -> int:
        cursor = conn.execute(
            ASSERTION_INSERT_SQL,
            (
                str(run_id),
                str(execution_id) if execution_id else None,
                metric_id,
                assertion.expression_repr.expr,
                int(assertion.passed),
                assertion.error_message,
            ),
        )
        return cast("int", cursor.lastrowid)

    def _save_predicate(
        self,
        conn: sqlite3.Connection,
        run_id: UUID,
        assertion_id: int,
        predicate: PredicateResult,
    ) -> None:
        meta = predicate.predicate_metadata
        conn.execute(
            PREDICATE_INSERT_SQL,
            (
                str(predicate.id),
                str(run_id),
                assertion_id,
                str(predicate.case_id) if predicate.case_id else None,
                meta.predicate_name,
                meta.merit_name,
                meta.actual,
                meta.reference,
                int(meta.strict) if meta.strict is not None else None,
                predicate.confidence if predicate.confidence is not None else None,
                int(predicate.value),
                predicate.message,
            ),
        )

    def _save_assertions_for_execution(
        self,
        conn: sqlite3.Connection,
        run_id: UUID,
        execution: TestExecution,
    ) -> None:
        for assertion in execution.result.assertion_results:
            assertion_id = self._save_assertion(
                conn, run_id, execution.execution_id, None, assertion
            )
            for predicate in assertion.predicate_results:
                self._save_predicate(conn, run_id, assertion_id, predicate)

        for sub in execution.sub_executions:
            self._save_assertions_for_execution(conn, run_id, sub)

    def _save_metric(
        self,
        conn: sqlite3.Connection,
        run_id: UUID,
        metric: MetricResult,
    ) -> int:
        value = metric.value
        value_real: float | None = None
        value_json: str | None = None

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value_real = float(value)
        else:
            value_json = json.dumps(value)

        meta = metric.metadata
        cursor = conn.execute(
            METRIC_INSERT_SQL,
            (
                str(run_id),
                str(metric.execution_id) if metric.execution_id else None,
                metric.name,
                meta.scope.value if isinstance(meta.scope, Scope) else str(meta.scope),
                value_real,
                value_json,
                meta.first_item_recorded_at.isoformat() if meta.first_item_recorded_at else None,
                meta.last_item_recorded_at.isoformat() if meta.last_item_recorded_at else None,
                self._to_json_list(meta.collected_from_merits),
                self._to_json_list(meta.collected_from_resources),
                self._to_json_list(meta.collected_from_cases),
            ),
        )
        return cast("int", cursor.lastrowid)

    def _save_metric_assertions(
        self,
        conn: sqlite3.Connection,
        run_id: UUID,
        metric: MetricResult,
        metric_id: int,
    ) -> None:
        """Save assertion results linked to a metric, reusing _save_assertion."""
        for assertion in metric.assertion_results:
            self._save_assertion(
                conn=conn,
                run_id=run_id,
                execution_id=None,
                metric_id=metric_id,
                assertion=assertion,
            )
            for predicate in assertion.predicate_results:
                self._save_predicate(
                    conn=conn,
                    run_id=run_id,
                    assertion_id=metric_id,
                    predicate=predicate,
                )

    def _to_json_list(self, items: set[str]) -> str | None:
        return json.dumps(list(items)) if items else None

    def _parse_case_id(self, id_suffix: str | None) -> UUID | None:
        """Extract case_id from id_suffix if it's a valid UUID."""
        if not id_suffix:
            return None
        try:
            return UUID(id_suffix)
        except ValueError:
            return None

    def get_run(self, run_id: UUID) -> MeritRun | None:
        """Retrieve a test run by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()

            if not row:
                return None

            return self._row_to_run(conn, row)

    def list_runs(self, limit: int = 10) -> list[MeritRun]:
        """List recent runs, ordered by start_time descending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
            ).fetchall()

            return [self._row_to_run(conn, row) for row in rows]

    def get_metrics_for_execution(self, execution_id: UUID) -> list[MetricResult]:
        """Get all metrics for a specific test execution."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE test_execution_id = ?",
                (str(execution_id),),
            ).fetchall()
            return [self._row_to_metric(row) for row in rows]

    def get_assertions_for_execution(self, execution_id: UUID) -> list[dict]:
        """Get all assertions for a specific test execution."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assertions WHERE test_execution_id = ?",
                (str(execution_id),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_assertions_for_run(self, run_id: UUID) -> list[dict]:
        """Get all assertions for a specific run."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assertions WHERE run_id = ?",
                (str(run_id),),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_predicates_for_assertion(self, assertion_id: int) -> list[dict]:
        """Get all predicates for a specific assertion."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predicates WHERE assertion_id = ?",
                (assertion_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def _row_to_run(self, conn: sqlite3.Connection, row: sqlite3.Row) -> MeritRun:
        env_data = json.loads(row["environment_json"]) if row["environment_json"] else {}
        environment = RunEnvironment(
            commit_hash=env_data.get("commit_hash"),
            branch=env_data.get("branch"),
            dirty=env_data.get("dirty"),
            python_version=env_data.get("python_version", ""),
            platform=env_data.get("platform", ""),
            hostname=env_data.get("hostname", ""),
            working_directory=env_data.get("working_directory", ""),
            merit_version=env_data.get("merit_version", ""),
            env_vars=env_data.get("env_vars", {}),
        )

        exec_rows = conn.execute(
            "SELECT * FROM test_executions WHERE run_id = ?",
            (row["run_id"],),
        ).fetchall()

        executions_by_id: dict[str, TestExecution] = {}
        for exec_row in exec_rows:
            executions_by_id[exec_row["execution_id"]] = self._row_to_execution(exec_row)

        executions: list[TestExecution] = []
        for exec_row in exec_rows:
            execution = executions_by_id[exec_row["execution_id"]]
            parent_id = exec_row["parent_id"]
            if parent_id:
                executions_by_id[parent_id].sub_executions.append(execution)
            else:
                executions.append(execution)

        metric_rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ?",
            (row["run_id"],),
        ).fetchall()
        metric_results = [self._row_to_metric(r) for r in metric_rows]

        result = RunResult(
            executions=executions,
            metric_results=metric_results,
            total_duration_ms=row["total_duration_ms"],
            stopped_early=bool(row["stopped_early"]),
        )

        end_time = None
        if row["end_time"]:
            end_time = datetime.fromisoformat(row["end_time"])

        return MeritRun(
            run_id=UUID(row["run_id"]),
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=end_time,
            environment=environment,
            result=result,
        )

    def _row_to_execution(self, row: sqlite3.Row) -> TestExecution:
        error = Exception(row["error_message"]) if row["error_message"] else None
        tags_json = row["tags_json"]
        tags = set(json.loads(tags_json)) if tags_json else set()

        definition = MeritTestDefinition(
            name=row["test_name"],
            fn=lambda: None,
            module_path=Path(row["file_path"]) if row["file_path"] else Path(),
            is_async=False,
            class_name=row["class_name"],
            tags=tags,
            skip_reason=row["skip_reason"],
            xfail_reason=row["xfail_reason"],
            id_suffix=row["id_suffix"],
        )

        result = TestResult(
            status=TestStatus(row["status"]),
            duration_ms=row["duration_ms"],
            error=error,
        )

        return TestExecution(
            definition=definition,
            result=result,
            execution_id=UUID(row["execution_id"]),
            trace_id=row["trace_id"],
        )

    def _row_to_metric(self, row: sqlite3.Row) -> MetricResult:
        if row["value"] is not None:
            value = cast("CalculatedValue", row["value"])
        elif row["value_json"]:
            value = cast("CalculatedValue", json.loads(row["value_json"]))
        else:
            value = float("nan")

        scope_str = row["scope"]
        scope = Scope(scope_str) if scope_str in {s.value for s in Scope} else Scope.SESSION

        first_at = row["first_recorded_at"]
        last_at = row["last_recorded_at"]
        metadata = MetricMetadata(
            first_item_recorded_at=datetime.fromisoformat(first_at) if first_at else None,
            last_item_recorded_at=datetime.fromisoformat(last_at) if last_at else None,
            scope=scope,
            collected_from_merits=self._from_json_set(row["collected_from_merits_json"]),
            collected_from_resources=self._from_json_set(row["collected_from_resources_json"]),
            collected_from_cases=self._from_json_set(row["collected_from_cases_json"]),
        )

        exec_id = row["test_execution_id"]
        return MetricResult(
            name=row["name"],
            metadata=metadata,
            assertion_results=[],
            value=value,
            execution_id=UUID(exec_id) if exec_id else None,
        )

    def _from_json_set(self, json_str: str | None) -> set[str]:
        return set(json.loads(json_str)) if json_str else set()
