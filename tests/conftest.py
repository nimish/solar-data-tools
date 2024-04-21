import pytest

from pathlib import Path


@pytest.fixture
def fixtures_dir() -> Path:
    return Path("tests/fixtures")
