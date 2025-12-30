import merit

from merit.metrics.base import Metric

from dotenv import load_dotenv

load_dotenv()


# Composite metrics


@merit.metric
def accuracy():
    metric = Metric()
    yield metric

    assert metric.distribution[True] == 0.5


@merit.metric
def false_positives(accuracy: Metric):
    metric = Metric()
    yield metric

    accuracy.add_record(metric.raw_values)  # writes once collected all values
    assert metric.counter[False] == 2


@merit.metric
def false_negatives(accuracy: Metric):
    metric = Metric()
    yield metric

    accuracy.add_record(metric.raw_values)  # writes once collected all values
    assert metric.counter[False] == 1


@merit.parametrize("pos", [False, True, True])
def merit_positives_test(pos: bool, false_negatives: Metric):
    false_negatives.add_record(pos)  # writes per each iterated case


@merit.parametrize("neg", [False, False, True])
def merit_negatives_test(neg: bool, false_positives: Metric):
    false_positives.add_record(neg)  # writes per each iterated case


# Case level vs session level


@merit.metric(scope="session")
def hallucinations_per_case():
    metric = Metric()
    yield metric

    assert metric.mean == 10


@merit.metric(scope="case")
def hallucinations_counter(hallucinations_per_case: Metric):
    metric = Metric()
    yield metric

    hallucinations_per_case.add_record(metric.raw_values)  # writes per each iterated case


@merit.parametrize("h", [5, 10, 15])
def merit_hallucinations_test(h: int, hallucinations_counter: Metric):
    hallucinations_counter.add_record(h)  # writes per each iterated case
