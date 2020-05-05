from pathlib import Path
import pandas as pd

import pytest


@pytest.fixture(scope="session")
def project_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    return Path(str(fn))

@pytest.fixture
def sample_data_dir():
    return Path("./tests/").absolute().joinpath("sample_data")
