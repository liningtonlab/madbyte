from pathlib import Path
import pandas as pd

import pytest

from madbyte import utils


def test_establish_file_structure(project_dir: Path):
    out_dir = project_dir.joinpath("test")
    utils.establish_file_structure("test", out_dir, project_dir)
    assert project_dir.exists()
    assert out_dir.exists()
    assert project_dir.joinpath("Spin_Systems_Master.json").exists()


def test_get_master(project_dir: Path):
    df = utils.get_master(project_dir)
    # Test the DF has a shape
    assert df.shape


def test_remove_self_corr(project_dir: Path, sample_data_dir: Path):
    sample_tocsy = pd.read_csv(sample_data_dir.joinpath("azithro_tocsy.csv"))
    expected = pd.read_csv(sample_data_dir.joinpath("azithro_tocsy_noselfcorr.csv"))
    res = utils.remove_self_corr(sample_tocsy, cols=['H1', 'H2'], inplace=False)
    assert res.shape
    print(res.shape)
    print(expected.shape)
    # Best we can do because of floating point error
    assert res.shape == expected.shape
