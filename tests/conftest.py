import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", help="run tests that call the real SEC APIs")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls data.sec.gov / www.sec.gov (needs SEC_USER_AGENT)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live SEC test: run with  pytest --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
