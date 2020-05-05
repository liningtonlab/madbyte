import json
import os

import numpy as np
import pandas as pd
import rtree

from madbyte.logging import get_logger


logger = get_logger("MADByTE")


CONFIG = {
    "ColsToMatch": ["H_PPM", "C_PPM", "PHASE"],
    "Tolerances": {
        "H_PPM": 0.03, 
        "C_PPM": 0.4, 
        "PHASE": 0.1,
    },
}


def simple_replicate_merge(df, config=None):
    """Take dataframe and perform Rtree comparison to replicate

    Args:
        df (pd.DataFrame): Data to work on

    Returns:
        pd.DataFrame: New dataframe with averaged data
    """
    if not config:
        config = CONFIG
    gen_error_cols(df, config)
    rects = get_rects(df, config)
    rtree = build_rtree(rects, config)
    con_comps = gen_con_comps(rtree, rects)
    new_data = []
    for c in con_comps:
        c_df = df.iloc[list(c)]
        if len(c_df) > 1:
            logger.debug(f"Averaging {c_df[config['ColsToMatch']].to_dict()}")
        new_data.append(average_data(c_df, config=config))
    return pd.DataFrame(new_data).round(4)


def gen_error_cols(df, config):
    """
    Uses the errorinfo dict to generate
    error windows for each of the columns.
    Mutates dataframe inplace for some memory conservation.

    Args:
        df (pandas.DataFrame): input dataframe to calc error windows (modified in place)
        errorinfo (dict): dict of error information
    """

    for dcol, evalue in config["Tolerances"].items():
        col = df[dcol]
        efunc = lambda x:evalue
        errors = col.apply(efunc)
        df[f"{dcol}_low"] = df[dcol] - errors
        df[f"{dcol}_high"] = df[dcol] + errors


def gen_con_comps(rtree: rtree.index.Index, rects: np.ndarray) -> set:
    """
    Generate connected components subgraphs for a graph where nodes are hyperrectangles
    and edges are overlapping hyperrectangles. This is done using the rtree index and
    a depth first search.
    """
    seen = set()

    for i, _ in enumerate(rects):
        if i in seen:
            continue
        search_idxs = [i]
        c = {i}
        while search_idxs:
            search = search_idxs.pop()
            try:
                neighbors = set(rtree.intersection(rects[search]))
            except Exception as e:
                print(e)
                print(rects[search])
                raise e

            for n in neighbors - seen:  # set math
                c.add(n)
                search_idxs.append(n)
                seen.add(n)
        yield c


def build_rtree(rects: np.ndarray, config) -> rtree.index.Index:
    """
    Build RTree index for rectangles for fast range queries.
    df needs errors cols pre-calculated
    """
    dims = len(config["ColsToMatch"])
    p = rtree.index.Property()
    p.dimension = dims
    p.interleaved = False
    rgen = ((i, r, None) for i, r in enumerate(rects))
    idx = rtree.index.Index(rgen, properties=p)
    return idx


def get_rects(df: pd.DataFrame, config) -> np.ndarray:
    """
    Get the error portions of df
    """
    ecols = [f"{c}_low" for c in config["ColsToMatch"]]
    ecols = ecols + [f"{c}_high" for c in config["ColsToMatch"]]

    return df[ecols].values


def average_data(df: pd.DataFrame, config) -> dict:
    """
    Takes conncect component DF and return average of compared values
    as dict for appending to list for new DF construction
    """
    dat = df[config["ColsToMatch"]].mean()
    if "Identity" in df.columns:
        dat["Identity"] = df.iloc[0]["Identity"]
    return dat
