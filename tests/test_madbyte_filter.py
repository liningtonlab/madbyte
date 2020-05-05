import pandas as pd
import pytest

from madbyte.filter import Filter
from madbyte.utils import apply_filters

sample_data = pd.DataFrame(
    data=[[0.5, 0.5], [0,0], [0.7, 1.5], [2,2]],
    columns=['H_PPM', 'C_PPM'],
)

def test_create_filter():
    f = Filter(["H_PPM"], [(0.0, 1.0)])
    assert f

def test_create_bad_filter():
    # intentionally make bad columns != values
    with pytest.raises(AssertionError):
        f = Filter(["H_PPM", "C_PPM"], [(0.0, 1.0)])

def test_filter_columns():
    f = Filter(["H_PPM"], [(0.0, 1.0)])
    assert f.columns == ["H_PPM"]

def test_filter_colmax():
    f = Filter(["H_PPM"], [(0.0, 1.0)])
    assert f.col_max("H_PPM") == 1.0

def test_filter_colmin():
    f = Filter(["H_PPM"], [(0.0, 1.0)])
    assert f.col_min("H_PPM") == 0.0

def test_filter_shape_stripe():
    f = Filter(["H_PPM"], [(0.0, 1.0)])
    assert f.shape == "stripe"

def test_filter_shape_block():
    f = Filter(["H_PPM", "C_PPM"], [(0.0, 1.0), (0.0, 1.0)])
    assert f.shape == "block"

def test_filter_shape_undef():
    f = Filter([],[])
    assert f.shape == "undefined"

def test_apply_stripe_filter():
    expected = pd.DataFrame(
        data=[[2.0,2.0]],
        columns=['H_PPM', 'C_PPM'],
    )
    f = [Filter(["H_PPM"], [(0.0, 1.0)])]
    print(sample_data)
    res = apply_filters(sample_data, f, inplace=False)
    res.reset_index(drop=True, inplace=True)
    print(res)
    assert expected.equals(res)

def test_apply_block_filter():
    expected = pd.DataFrame(
        data=[[0.7, 1.5], [2,2]],
        columns=['H_PPM', 'C_PPM'],
    )
    f = [Filter(["H_PPM", "C_PPM"], [(0.0, 1.0), (0.0, 1.0)])]
    print(sample_data)
    res = apply_filters(sample_data, f, inplace=False)
    res.reset_index(drop=True, inplace=True)
    print(res)
    assert expected.equals(res)

def test_apply_multi_filters():
    expected = pd.DataFrame(
        data=[[0.7, 1.5]],
        columns=['H_PPM', 'C_PPM'],
    )
    f = [
            Filter(["H_PPM", "C_PPM"], [(0.0, 1.0), (0.0, 1.0)]),
            Filter(["H_PPM"], [(1.9, 2.1)]),
        ]
    print(sample_data)
    res = apply_filters(sample_data, f, inplace=False)
    res.reset_index(drop=True, inplace=True)
    print(res)
    assert expected.equals(res)
