from pathlib import Path
import pandas as pd

import pytest

ROOTDIR = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data")
    print(fn)
    return Path(str(fn))

@pytest.fixture
def sample_data_dir():
    return ROOTDIR / "tests" / "sample_data"
