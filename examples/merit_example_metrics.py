import merit

from merit.metrics.base import Metric

from merit.testing.case import Case

from typing import Generator, Any

from dotenv import load_dotenv

load_dotenv()

# Metrics work on different scopes

@merit.metric(scope="case")
def hallucination_count() -> Metric:
    metric = Metric(name="hallucination_count")
    return metric


@merit.metric
def avg_hallucinations_per_case() -> Generator[Metric, Any, Any]:
    metric = Metric(name="avg_hallucinations_per_case")
    yield metric

    assert metric.mean == 15
    metric.push_to_dashboard()


@merit.parametrize("hallucinations,id_suffix", [([1,2,3], "test1"), ([4,5,6], "test2"), ([7,8,9], "test3")])
def merit_hallucinations_test(
    hallucinations: int, 
    id_suffix: str, 
    hallucination_count: Metric, 
    avg_hallucinations_per_case: Metric
    ) -> None:
    # Record the hallucinations to the case-level metric and push to dashboard
    hallucination_count.record_values(hallucinations)
    hallucination_count.metadata = {"id_suffix": id_suffix}
    hallucination_count.metric_value = hallucination_count.sum
    hallucination_count.push_to_dashboard()

    # Record the hallucinations to the session-level metric to aggregate it later
    avg_hallucinations_per_case.record_values(hallucination_count.metric_value)


# Aggregated metrics

@merit.metric
def false_positives() -> Generator[Metric, Any, Any]:
    metric = Metric(name="false_positives")
    yield metric

    metric.metric_value = metric.frequency_count[False]
    assert metric.metric_value == 3

@merit.metric
def false_negatives() -> Generator[Metric, Any, Any]:
    metric = Metric(name="false_negatives")
    yield metric

    metric.metric_value = metric.frequency_count[False]
    assert metric.metric_value == 1

@merit.metric
def accuracy(false_positives, false_negatives) -> Generator[Metric, Any, Any]:
    metric = Metric(name="accuracy")
    yield metric

    metric.record_values(false_positives.raw_values + false_negatives.raw_values)
    metric.metric_value = metric.frequency_share[True]
    assert metric.metric_value > 0.25




@merit.parametrize("pos", [False, True, True])
def merit_positives_test(pos: bool, false_negatives: Metric):
    false_negatives.record_values(pos)

@merit.parametrize("neg", [False, False, False])
def merit_negatives_test(neg: bool, false_positives: Metric):
    false_positives.record_values(neg)
