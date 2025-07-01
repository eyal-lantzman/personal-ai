import pytest
import logging
from agent.web_search import identify_region, simple_search

logger = logging.getLogger(__name__)

TEST_RUNS = 3

identify_region_test_cases = [
    ("What are the top 3 thanksgiving dishes with yams?", ["us-en"]),
    ("What colours considered posh this year?", ["uk-en"]),
    ("Which are the top 3 place to visit in Canada?", ["ca-en", "us-en"]),
    ("What is the projected MSFT price in 3 years from now? ", ["wt-wt", "us-en"]),
]
@pytest.mark.parametrize("run", range(TEST_RUNS))
@pytest.mark.parametrize("query,expected_results", identify_region_test_cases)
def test_identify_region(query, expected_results, run):
    assert identify_region(query) in expected_results

@pytest.mark.parametrize("run", range(TEST_RUNS))
def test_simple_search(run):
    results = simple_search(run, query="What's new?", num_results=run+1)
    logger.info(results)
    assert len(results.web_research_results) == run+1
    assert len(results.sources_gathered) == run+1
    for i, source in enumerate(results.sources_gathered):
        assert source.startswith(f"[{run}.{i+1}](")
        assert source.endswith(")")
