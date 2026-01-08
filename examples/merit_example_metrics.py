import merit

from merit import Metric, metrics


# Use dependency injection to stack metrics
# Example:
# - Track false positives and false negatives results separately to understand what fixes are needed
# - Propagate values from both metrics into an accuracy metric to check if AI system is ready for production


def band_quality_classifier(query: str) -> bool:
    if "Metallica" in query:
        return True
    elif "Whitesnake" in query:
        return True
    elif "Led Zeppelin" in query:
        return False
    elif "Megadeth" in query:
        return True
    elif "Nickelback" in query:
        return True
    elif "Limp Bizkit" in query:
        return False
    else:
        raise ValueError(f"Unknown query: {query}")


@merit.metric
def accuracy():
    metric = Metric()
    yield metric

    # Check accuracy metric is 50%
    assert metric.distribution[True] == 0.5


@merit.metric
def false_positives(accuracy: Metric):
    metric = Metric()
    yield metric

    # Write false positives to accuracy metric
    accuracy.add_record(metric.raw_values)

    # Check the metric registered two false positive results
    assert metric.counter[False] == 2


@merit.metric
def false_negatives(accuracy: Metric):
    metric = Metric()
    yield metric

    # Write false negatives to accuracy metric
    accuracy.add_record(metric.raw_values)

    # Check the metric registered one false negative result
    assert metric.counter[False] == 1


@merit.parametrize("good_band_name", ["Metallica", "Whitesnake", "Led Zeppelin"])
def merit_expected_true(good_band_name: str, false_negatives: Metric):
    """
    Test that the classifier returns True for good band names.
    If AI returns False: register it as a false negative.
    """
    is_good = band_quality_classifier(good_band_name)
    with metrics([false_negatives]):
        assert is_good


@merit.parametrize("horrible_band_name", ["Megadeth", "Nickelback", "Limp Bizkit"])
def merit_expected_false(horrible_band_name: str, false_positives: Metric):
    """
    Test that the classifier returns False for horrible band names.
    If AI returns True: register it as a false positive.
    """
    is_good = band_quality_classifier(horrible_band_name)
    with metrics([false_positives]):
        assert not is_good


# Use different scopes to track local and global statistics
# Example:
# - Track hallucinations number for each case to understand which cases require debugging
# - Calculate average hallucinations number across all cases to understand the overall system performance


def geography_bot(query: str) -> str:
    if "San Francisco" in query:
        return "California, USA"
    elif "Boston" in query:
        return "Massachusetts, Canada"
    elif "Chicago" in query:
        return "Washington, USA"
    elif "Seattle" in query:
        return "Washington, USA"
    elif "Miami" in query:
        return "California, USA"
    elif "Houston" in query:
        return "California, Uzbekistan"
    elif "Washington" in query:
        return "California, Canada"
    else:
        raise ValueError(f"Unknown query: {query}")


@merit.metric(scope="session")
def average_hallucinations_per_case():
    metric = Metric()
    yield metric

    # Check that the average number of hallucinations per case is 1
    assert metric.mean == 1


@merit.metric(scope="case")
def case_hallucinations_count(average_hallucinations_per_case: Metric):
    metric = Metric()
    yield metric

    # Write the number of hallucinations for the case to the average metric
    average_hallucinations_per_case.add_record(metric.counter[False])


@merit.parametrize(
    "city,expected_state,expected_country",
    [
        ("San Francisco", "California", "USA"),
        ("Boston", "Massachusetts", "USA"),
        ("Chicago", "Illinois", "USA"),
        ("Seattle", "Washington", "USA"),
        ("Miami", "Florida", "USA"),
        ("Houston", "Texas", "USA"),
        ("Washington", "District of Columbia", "USA"),
    ],
)
def merit_hallucinations_test(
    city: str, 
    expected_state: str, 
    expected_country: str, 
    case_hallucinations_count: Metric
    ):
    """
    Test that the geography bot returns the correct state and country for the given city.
    If AI returns a different state or country: register it as a hallucination for the case.
    """
    result = geography_bot(city)
    with metrics([case_hallucinations_count]):
        assert expected_state in result
        assert expected_country in result