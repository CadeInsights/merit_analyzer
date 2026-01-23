from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from merit.assertions.base import AssertionRepr, AssertionResult
from merit.metrics_.base import MetricMetadata, MetricResult
from merit.predicates.base import PredicateMetadata, PredicateResult
from merit.resources import Scope
from merit.storage.sqlite import SQLiteStore
from merit.testing.models.definition import MeritTestDefinition
from merit.testing.models.result import TestExecution, TestResult, TestStatus
from merit.testing.models.run import MeritRun, RunEnvironment, RunResult


def test_sqlite_store_save_and_get_run(tmp_path: Path) -> None:
    db_path = tmp_path / "merit.db"
    store = SQLiteStore(db_path)

    execution_id = uuid4()
    sub_execution_id = uuid4()
    start_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    end_time = datetime(2024, 1, 1, 12, 5, tzinfo=UTC)

    definition = MeritTestDefinition(
        name="test_one",
        fn=lambda: None,
        module_path=Path("tests/test_sample.py"),
        is_async=False,
        tags={"smoke"},
        id_suffix=str(uuid4()),
    )
    sub_definition = MeritTestDefinition(
        name="test_sub",
        fn=lambda: None,
        module_path=Path("tests/test_sample.py"),
        is_async=False,
    )

    sub_execution = TestExecution(
        definition=sub_definition,
        result=TestResult(status=TestStatus.PASSED, duration_ms=10.0),
        execution_id=sub_execution_id,
    )
    execution = TestExecution(
        definition=definition,
        result=TestResult(
            status=TestStatus.FAILED,
            duration_ms=20.0,
            error=Exception("boom"),
        ),
        execution_id=execution_id,
        trace_id="trace-1",
        sub_executions=[sub_execution],
    )

    metric_metadata = MetricMetadata(
        first_item_recorded_at=start_time,
        last_item_recorded_at=end_time,
        scope=Scope.CASE,
        collected_from_merits={"merit_test"},
        collected_from_resources={"resource"},
        collected_from_cases={"case-1"},
    )
    metric_result = MetricResult(
        name="latency_ms",
        metadata=metric_metadata,
        assertion_results=[],
        value=12.5,
        execution_id=execution_id,
    )

    environment = RunEnvironment(
        commit_hash="abc123",
        branch="main",
        dirty=False,
        python_version="3.12.0",
        platform="darwin",
        hostname="host",
        working_directory="/tmp/project",
        merit_version="1.0.0",
        env_vars={"ENV": "1"},
    )

    run = MeritRun(
        run_id=uuid4(),
        start_time=start_time,
        end_time=end_time,
        environment=environment,
        result=RunResult(
            executions=[execution],
            metric_results=[metric_result],
            total_duration_ms=123.4,
            stopped_early=False,
        ),
    )

    store.save_run(run)
    loaded = store.get_run(run.run_id)

    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.start_time == start_time
    assert loaded.end_time == end_time
    assert loaded.environment.branch == "main"
    assert loaded.environment.commit_hash == "abc123"
    assert loaded.result.total_duration_ms == 123.4
    assert loaded.result.failed == 1
    assert loaded.result.total == 1
    assert len(loaded.result.executions) == 1
    assert loaded.result.executions[0].trace_id == "trace-1"
    assert len(loaded.result.executions[0].sub_executions) == 1
    assert len(loaded.result.metric_results) == 1
    assert loaded.result.metric_results[0].name == "latency_ms"
    assert loaded.result.metric_results[0].value == 12.5
    assert loaded.result.metric_results[0].metadata.scope == Scope.CASE


def test_sqlite_store_list_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "merit.db"
    store = SQLiteStore(db_path)

    run_one = MeritRun(
        run_id=uuid4(),
        start_time=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 1, 10, 10, tzinfo=UTC),
        environment=RunEnvironment(merit_version="1.0.0"),
        result=RunResult(),
    )
    run_two = MeritRun(
        run_id=uuid4(),
        start_time=datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
        end_time=datetime(2024, 1, 2, 10, 10, tzinfo=UTC),
        environment=RunEnvironment(merit_version="1.0.0"),
        result=RunResult(),
    )

    store.save_run(run_one)
    store.save_run(run_two)

    runs = store.list_runs(limit=1)

    assert len(runs) == 1
    assert runs[0].run_id == run_two.run_id


def test_sqlite_store_assertions_and_predicates(tmp_path: Path) -> None:
    db_path = tmp_path / "merit.db"
    store = SQLiteStore(db_path)

    execution_id = uuid4()
    predicate_meta = PredicateMetadata(
        actual="actual",
        reference="reference",
        strict=True,
        predicate_name="equals",
        merit_name="merit_test",
    )
    predicate_result = PredicateResult(
        predicate_metadata=predicate_meta,
        value=False,
        confidence=0.3,
        message="nope",
        case_id=uuid4(),
    )
    assertion = AssertionResult(
        expression_repr=AssertionRepr(
            expr="x == y",
            lines_above="",
            lines_below="",
            resolved_args={},
        ),
        passed=False,
        error_message="bad",
        predicate_results=[predicate_result],
    )

    definition = MeritTestDefinition(
        name="test_one",
        fn=lambda: None,
        module_path=Path("tests/test_sample.py"),
        is_async=False,
    )
    execution = TestExecution(
        definition=definition,
        result=TestResult(
            status=TestStatus.FAILED,
            duration_ms=12.0,
            error=Exception("boom"),
            assertion_results=[assertion],
        ),
        execution_id=execution_id,
    )

    metric_metadata = MetricMetadata(scope=Scope.SESSION)
    metric_assertion = AssertionResult(
        expression_repr=AssertionRepr(
            expr="metric > 0",
            lines_above="",
            lines_below="",
            resolved_args={},
        ),
        passed=True,
    )
    metric_result = MetricResult(
        name="count",
        metadata=metric_metadata,
        assertion_results=[metric_assertion],
        value=[1, 2, 3],
        execution_id=None,
    )

    run = MeritRun(
        run_id=uuid4(),
        start_time=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        environment=RunEnvironment(merit_version="1.0.0"),
        result=RunResult(
            executions=[execution],
            metric_results=[metric_result],
            total_duration_ms=10.0,
        ),
    )

    store.save_run(run)

    assertions = store.get_assertions_for_execution(execution_id)
    assert len(assertions) == 1
    assert assertions[0]["expression_repr"] == "x == y"

    predicates = store.get_predicates_for_assertion(assertions[0]["id"])
    assert len(predicates) == 1
    assert predicates[0]["predicate_name"] == "equals"
    assert predicates[0]["merit_name"] == "merit_test"

    run_assertions = store.get_assertions_for_run(run.run_id)
    assert any(row["metric_id"] is not None for row in run_assertions)
    assert any(row["test_execution_id"] == str(execution_id) for row in run_assertions)
    assert any(row["expression_repr"] == "metric > 0" for row in run_assertions)
