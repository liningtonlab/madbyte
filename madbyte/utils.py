import json
import math
import os
from collections import defaultdict
from functools import reduce
from itertools import chain, combinations
from pathlib import Path
from random import choice

import networkx as nx
import numpy as np
import pandas as pd
import rtree
from joblib import Parallel, delayed
from tqdm import tqdm

from madbyte import dereplicator
import madbyte.plotting as plot
from madbyte.logging import get_logger


logger = get_logger("MADByTE")


def establish_file_structure(project_dir):
    """Create output data directory and empty spin system master if required
     for MADByTE

    Args:
        output_dir (str or Path): desired output directory
        project_dir (str or Path): project directory
    """
    project_dir = Path(project_dir)

    master = project_dir.joinpath("Spin_Systems_Master.json")
    if not master.exists():
        master_df = pd.DataFrame(
            columns=["Spin_System_ID","Members","Found_In"],
        )
        master_df.to_json(master)
        logger.info("Required master file not found. One has been created.")


def apply_filters(df, filters, name="MADByTE", inplace=True):
    """Take dataframe and apply list of custom filter objects

    Args:
        df (pd.DataFrame): Data to work on
        filters (list): List of custom Filter objects
        name (str, optional): Name of sample for logging
        inplace (bool, optional): Apply filtering inplace. Defaults to True.
    """
    # Going to construct a flexible pandas mask
    mask_list = []
    for f in filters:
        inner_mask = []
        for col in f.columns:
            inner_mask.append(
                df[col].between(f.col_min(col), f.col_max(col))
            )
        mask_list.append(np.logical_and.reduce(inner_mask))
    mask = np.logical_or.reduce(mask_list)
    # If you want reporting you will have to do it this way instead of
    # just doing -mask
    bad_points = df[mask]
    logger.info(f"{name} : Found {bad_points.shape[0]} points to drop by filters")

    # Apply function and return None if inplace=True
    # Otherwise return DF
    return df.drop(bad_points.index, inplace=inplace)


def remove_self_corr(df, cols=['Ha', 'Hb'], tolerance=0.05, name="MADByTE", inplace=True):
    """Take dataframe and remove self correlated points in defined columns

    Args:
        df (pd.DataFrame): Data to work on
        cols (list, optional): Columns to compare. Defaults to ['Ha', 'Hb'].
        tolerance (float, optional): Point comparison tolerance. Defaults to 0.05.
        name (str, optional): Name of sample for logging
        inplace (bool, optional): Apply filtering inplace. Defaults to True.
    """
    # This function will only work for two columns
    assert len(cols) == 2
    bad_points = df[(df[cols[0]]-df[cols[1]]).abs() < tolerance]
    if not bad_points.empty:
        logger.info(f"{name} : Found {bad_points.shape[0]} points to dropped for self correlation")
    else:
        logger.info(f"{name} : No self correlating points found")

    # Apply function and return None if inplace=True
    # Otherwise return DF
    return df.drop(bad_points.index, inplace=inplace)


# Helpers for mutliplet merger
CONFIG = {
    "ColsToMatch": ["H_PPM", "C_PPM", "PHASE"],
    "Tolerances": {
        "H_PPM": 0.03/2,
        "C_PPM": 0.4/2,
        "PHASE": 0.1,
    },
}


def mutliplet_merger(df, config=CONFIG, name="MADByTE"):
    """Take dataframe and perform Rtree comparison to dereplicate

    Args:
        df (pd.DataFrame): Data to work on
        name (str, optional): Name of sample for logging

    Returns:
        pd.DataFrame: New dataframe with averaged data
    """
    logger.info(f"{name}: Peaklist before merger has {len(df)} points")
    gen_phase_col(df)
    new_df = dereplicator.simple_replicate_merge(df, config=config)
    new_df = new_df.round(3)
    logger.info(f"{name} : Peaklist after merger has {len(new_df)} points")
    return new_df


def gen_phase_col(df: pd.DataFrame) -> None:
    df["PHASE"] = df["Intensity"].apply(np.sign)


def tocsy_in_hsqc(tocsy, hsqc, tolerance=0.05, name="MADByTE", inplace=True):
    """Takes TOCSY + HSQC dataframes and removes protons from
    TOCSY not in HSQC

    Args:
        tocsy (pd.DataFrame): TOCSY dataframe
        hsqc (pd.DataFrame): HSQC dataframe
        name (str, optional): Name of sample for logging
        inplace (bool, optional): Apply filtering inplace. Defaults to True.
    """
    hsqc_points = hsqc.H_PPM.values
    bad_points = set()
    for idx, row in tocsy.iterrows():
        if not any([math.isclose(row.Ha, x, abs_tol=tolerance) for x in hsqc_points]):
            logger.debug(f"{name} : TOCSY cleanup {row.Ha} as `Ha` was not found in the HSQC data")
            bad_points.add(idx)
        if not any([math.isclose(row.Hb, x, abs_tol=tolerance) for x in hsqc_points]):
            logger.debug(f"{name} : TOCSY cleanup {row.Hb} as `Hb` was not found in the HSQC data")
            bad_points.add(idx)

    logger.info(f"{name}: There are {len(list(bad_points))} points to remove the the TOCSY peak list that are not in the HSQC")

    # Apply function and return None if inplace=True
    # Otherwise return DF
    return tocsy.drop(bad_points, inplace=inplace)

## DEPRECATED - not used in pipeline
def align_tocsy_hsqc(tocsy, hsqc):
    """Takes TOCSY + HSQC dataframes and aligns TOCSY with HSQC protons

    Args:
        tocsy (pd.DataFrame): TOCSY dataframe
        hsqc (pd.DataFrame): HSQC dataframe
        name (str, optional): Name of sample for logging
    """
    hsqc_points = hsqc.H_PPM.values
    ### Adjusted for analysis of points being re-assigned ###
    try:
        TOCSY_Reassginment_DF = pd.read_csv('TOCSY_Point_assignment.csv',index_col=0)#
    except:
        TOCSY_Reassginment_DF = pd.DataFrame(columns=['TOCSY_point','HSQC_point'])#
    
    for idx, row in tocsy.iterrows():
        # Get closest values
        closest_ha = hsqc_points[np.abs(hsqc_points-row.Ha).argmin()]
        closest_hb = hsqc_points[np.abs(hsqc_points-row.Hb).argmin()]
        # Set closest values
        if not math.isclose(row.Ha, closest_ha, abs_tol=0.001):
            logger.debug(f"{name} : Ha: {row.Ha}, {closest_ha}")
            TOCSY_Reassginment_DF = DF.append({'TOCSY_point':row.Ha,'HSQC_point':closest_ha},ignore_index=True)#
            tocsy.loc[idx, 'Ha'] = closest_ha
        if not math.isclose(row.Hb, closest_hb, abs_tol=0.001):
            logger.debug(f"{name} : Hb: {row.Hb}, {closest_hb}")
            TOCSY_Reassginment_DF = DF.append({'TOCSY_point':row.Hb,'HSQC_point':closest_hb},ignore_index=True)#
            tocsy.loc[idx, 'Hb'] = closest_hb
    # drop duplicates
    tocsy.drop_duplicates(["Ha", "Hb"], inplace=True)
    TOCSY_Reassginment_DF.to_csv('TOCSY_Point_assignment.csv')#

def self_align_tocsy(tocsy, tolerance=0.02, inplace=True, name="MADByTE"):
    """Take the TOCSY data and align Hb with data in Ha.
    Ha (F2) has better resolution than Hb (F1) so aligning to Ha forces
    "real" data.

    Args:
        tocsy (pd.DataFrame): TOCSY dataframe
        inplace (bool, optional): Apply inplace. Defaults to True.
    """
    # Should have some tolerance here too!
    if not inplace:
        df = tocsy.copy()
    else:
        df = tocsy
    unique_ha = np.array(list(set(df.Ha.values)))
    for idx, row in df.iterrows():
        closest_ha = unique_ha[np.abs(row.Hb-unique_ha).argmin()]
        if np.abs(closest_ha-row.Hb) > tolerance:
            logger.debug(f"{name} - No matching Ha for Hb = {row.Hb}")
        else:
            logger.debug(f"{name} - Closest Ha = {closest_ha} for Hb = {row.Hb}")
            df.loc[idx, 'Hb'] = closest_ha

    # Apply function and return None if inplace=True impliclity
    # Otherwise return DF
    df.drop_duplicates(["Ha", "Hb"], inplace=True)
    if not inplace:
        return df


def gen_subgraphs(G, min_size=2):
    """Take a Networkx Graph and return generator of subgraphs filtered by minsize

    Args:
        G (nx.Graph): Graph to find connected components of
        min_size (int, optional): Min number of nodes in Subgraph. Defaults to 2.

    Returns:
        (generator): Generator of subgraphs
    """
    if G.is_directed():
        return (G.subgraph(c).copy() for c in nx.weakly_connected_components(G) if len(c) >= min_size)
    else:
        return (G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) >= min_size)


def del_min_directed(G, min_size=2):
    assert G.is_directed()
    while True:
        altered = False
        # make a new copy of the subgraph after each iter
        for n in G.copy().nodes:
            if len(G.in_edges(n))+len(G.out_edges(n)) < min_size:
                G.remove_node(n)
                altered = True
        if not altered:
            break


def del_min_undirected(G, min_size=2):
    assert not G.is_directed()
    while True:
        altered = False
        # make a new copy of the subgraph after each iter
        for n in G.copy().nodes:
            if len(G.edges(n)) < min_size:
                G.remove_node(n)
                altered = True
        if not altered:
            break


def delete_min_connected(G, min_size=2):
    if G.is_directed():
        del_min_directed(G, min_size=min_size)
    else:
        del_min_undirected(G, min_size=min_size)


def create_graph(df, cols=["Ha", "Hb"], name="MADByTE", graph="di"):
    assert graph in ["di", "multi"]
    if graph == "multi":
        G = nx.MultiGraph()
    else:
        G = nx.DiGraph()
    G.add_edges_from(df[cols].values)
    logger.info(f"{name} : Initial spin network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def is_clique(G):
    return not any(node_needs_splitting(G, n) for n in G)


def node_needs_splitting(sg, n):
    nneigh = len(set(sg.predecessors(n)) | set(sg.successors(n)))
    nedge = len(list(sg.in_edges(n))+list(sg.out_edges(n)))
    return nedge < 2 * nneigh


# @timer(logger=logger)
def chop_subgraph(G):
    if len(G.nodes) == 2 or is_clique(G):
        return [G]

    # Create set of edges and randomly try
    # deleting and see if we get broken clique
    # move good subgraphs to a list and keep trying
    good_subgraphs = []
    edges = set(G.edges)
    while edges:
        H = G.copy()
        random_edge = choice(list(edges))
        edges.remove(random_edge)
        H.remove_edge(*random_edge)
        subgraphs = list(gen_subgraphs(H))

        if len(subgraphs) == 1:
            _ = [delete_min_connected(x) for x in subgraphs]
            good = [x for x in subgraphs if is_clique(x)]
            if good:
                good_subgraphs.extend(good)
                bad = [x for x in subgraphs if not is_clique(x)]
                if not bad:
                    break
                G = nx.compose_all(bad)
                edges = set(G.edges)
        # Add recursive call when large subgraphs get split
        elif len(subgraphs) > 1:
            _ = [delete_min_connected(x) for x in subgraphs]
            sgs = []
            for g in subgraphs:
                sgs.extend(chop_subgraph(g))
            good_subgraphs.extend(sgs)
    # If any successful splitting, return
    # Else, graph is valid
    return good_subgraphs or [G]


def find_tocsy_spin_systems(tocsy, name=None, out_dir=None, cols=["Ha", "Hb"], graph="di",):
    G = create_graph(tocsy, cols, name=name, graph=graph)
    if name:
        out_dir = Path(out_dir) if out_dir else Path()
        # Create file if needed
        nx.write_graphml(G, out_dir.joinpath(f"{name}_TOCSY_{graph}_raw.graphml"))
    # list of subgraph spin systems
    spin_graphs = []
    for sg in gen_subgraphs(G, min_size=2):
        # Need to introduce some logic to differentiant
        delete_min_connected(sg)
        # can eliminate single nodes here
        if len(sg.nodes()) == 1:
            continue
        # chop_subgraph returns list
        spin_graphs.extend(chop_subgraph(sg))

    if spin_graphs:
        H = nx.compose_all(spin_graphs)
        # compute removed protons
        diff_spin_networks(initial=G, final=H, name=name)
    else:
        logger.warning(f"Warning! {name} returns no coherent spin systems")
        logger.warning("Writing an empty graph output anyway...")
        H = nx.DiGraph()
    logger.info(f"{name} : Final spin network has {H.number_of_nodes()} nodes and {H.number_of_edges()} edges")
    if name:
        nx.write_graphml(H, out_dir.joinpath(f"{name}_TOCSY_{graph}_processed.graphml"))
    return H


def diff_spin_networks(initial, final, name="MADByTE"):
    diff = initial.nodes - final.nodes
    for n in diff:
        logger.debug(f"{name} : {n} removed from spin network")


def build_spin_systems(tocsy, hsqc, name, out_dir, tolerance=0.05):
    tocsy_G = find_tocsy_spin_systems(tocsy, out_dir=out_dir, name=name)
    spin_systems = {}
    for idx, sg in enumerate(gen_subgraphs(tocsy_G)):
        spin_systems[f"{name}_{idx}"] = get_hsqc_in_tocsy_system(sg, hsqc, tolerance, name=name)

    with out_dir.joinpath(f"{name}_spin_systems.json").open("w") as f:
        f.write(json.dumps(spin_systems, indent=4))
    return spin_systems


def add_spin_systems_to_master(name, spin_systems, master):
    # Save some time by returning early if no spin systems
    if not spin_systems:
        return master
    data = pd.DataFrame(
        [
            {"Found_In": name, "Spin_System_ID": k, "Members": v}
            for k,v in spin_systems.items()
        ]
    )
    output = master.append(data, ignore_index=True, sort=False).\
        drop_duplicates("Spin_System_ID", keep="last").\
        reset_index(drop=True)
    return output


def get_hsqc_in_tocsy_system(G, hsqc, tolerance=0.05, name="MADByTE"):
    # G is spin subgraph which contains protons
    system = set() # Ensure's not duplicates within a spin system
    for prot in G.nodes():
        # Want to find the closest proton possible and assign carbons from TOCSY
        # If two values are equidistant, then just assign both (should be rare)
        slc = hsqc[hsqc.H_PPM.between(prot-tolerance, prot+tolerance)]
        slc_protons = slc.H_PPM.unique()
        diffs = np.abs(prot-slc_protons)
        min_diff = diffs.min()
        minima = slc_protons[[np.isclose(x, min_diff) for x in diffs]]
        logger.debug(f"{name} : Found {len(minima)} HSQC protons to match TOCSY proton {prot}")
        if len(minima) == 0:
            logger.warning(f"{name} : No HSQC protons near TOCSY proton {prot}")
            continue
        elif len(minima) > 1:
            sub_slc = slc
        else:
            sub_slc = slc.loc[slc.H_PPM==minima[0]]
        system.update([(prot, carb) for carb in sub_slc.C_PPM.values])
    return sorted(system)


def load_spin_systems(project_dir):
    project_dir = Path(project_dir)
    all_spin_systems = {}
    for d in project_dir.glob("*"):
        if not d.is_dir():
            continue
        ss_file = d.joinpath(f"{d.name}_spin_systems.json")
        if not ss_file.exists():
            raise FileNotFoundError
        all_spin_systems.update(json.load(ss_file.open()))
    df = pd.DataFrame(
        [{"ID": k, "H_ppm": x[0], "C_ppm": x[1]} for k,v in all_spin_systems.items() for x in v]
    )
    return df


def compute_corr_matrix(df, h_tol=0.05, c_tol=0.5):
    idxs = df["ID"].unique()
    mat = pd.DataFrame(columns=idxs, index=idxs)

    def do_ratio(x, idx, y, idy):
        ratio = ratio_two_systems(idx, idy, df, h_tol, c_tol)
        return (x, y, ratio)


    order = [
        (x,idx,y,idy) for x, idx in enumerate(idxs) for y, idy in enumerate(idxs)
    ]

    results = Parallel(n_jobs=-1)(
            delayed(do_ratio)(*o) for o in tqdm(order)
        )

    # Old methods without parallelization
    # results = [do_ratio(*o) for o in tqdm(order)]
    # for x, idx in enumerate(tqdm(idxs)):
    #     for y, idy in enumerate(idxs):
    #         ratio = ratio_two_systems(idx, idy, df, h_tol, c_tol)
    #         mat.iloc[x,y] = ratio

    for r in results:
        mat.iloc[r[0], r[1]] = r[2]

    return mat


def ratio_two_systems(idx, idy, df, h_tol, c_tol):
    if idx == idy:
        return 1.0
    x_df = df[df.ID==idx]
    y_df = df[df.ID==idy]
    count = 0
    protons = x_df.groupby("H_ppm")
    denom = len(protons)
    for h, grp in protons:
        matching = -y_df[(y_df.H_ppm.between(h-h_tol,h+h_tol,inclusive=False))
                     &(np.logical_or.reduce(
                        [y_df.C_ppm.between(c-c_tol,c+c_tol,inclusive=False) for c in grp.C_ppm]
                        ))
                     ].empty
        if not matching:
            count += 1
    return count/denom


def partition(p, iter_):
    'Use a predicate to partition entries into true entries and false entries'
    return reduce(lambda x, y: x[not p(y)].append(y) or x, iter_, ([], []))


def trim_associations(G):
    H = G.copy()
    for n, d in H.copy().nodes.items():
        if d['_type'] == "spin" and len(H.edges(n)) < 2:
            H.remove_node(n)
    for n, d in H.copy().nodes.items():
        if d['_type'] != "spin" and len(H.edges(n)) == 0:
            H.remove_node(n)
    return H
def filter_singletons(G):
    H = G.copy()
    for n, d in H.copy().nodes.items():
        if d['_type'] == "spin" and len(eval(d.get('members', '[]')))< 2:
            H.remove_node(n)
    for n, d in H.copy().nodes.items():
        if d['_type'] != "spin" and len(H.edges(n)) == 0:
            H.remove_node(n)
    return H


def association_network(
    project_dir,
    fname,
    corr_mat,
    master,
    colors,
    cutoff=0.5,
    hppm_error=0.05,
    cppm_error=0.5,
    extract_size=15,
    feature_size=10,
    max_system_size=20,
):
    master = master.loc[master['Members'].apply(lambda x: len(x) < max_system_size)]
    idxs = master.Spin_System_ID.tolist()
    systems = [(row.Spin_System_ID, {"members": str(row.Members)}) for row in master.itertuples()]
    extracts = master.Found_In.unique()
    standards, samples = partition(lambda x: x.startswith("HND_"), extracts)
    idx_master = master.set_index("Spin_System_ID", drop=True)

    def get_weight(x,y):
        # Get max value from two ids
        return float(max(corr_mat.loc[x,y], corr_mat.loc[y,x]))

    def same_sample(x,y):
        # tells whether two spin system are from the sample sample
        # based on matching of first 7 char of id
        return idx_master.loc[x, 'Found_In'] == idx_master.loc[y, 'Found_In']

    # Determine edges
    # Edges from extract to systems
    extract_edges = [(row.Spin_System_ID, row.Found_In) for row in master.itertuples()]
    spinsystem_edges = [
        (x,y,get_weight(x,y)) for x,y in combinations(idxs, r=2)
        if corr_mat.loc[x,y] >= cutoff and not same_sample(x,y)
    ]

    # network building
    G = nx.Graph()
    G.add_nodes_from(systems, _color=colors['spin'], _type="spin")
    G.add_nodes_from(samples, _color=colors['extract'], _type="extract")
    G.add_nodes_from(standards, _color=colors['standard'], _type="standard")
    G.add_edges_from(extract_edges, weight=1.0)
    G.add_weighted_edges_from(spinsystem_edges)
    plot.create_bokeh(G, "MADByTE Full Association Network - All Spin Systems", project_dir.joinpath(f"{fname}_association_network_all.html"),
        extract_size=extract_size,feature_size=feature_size,)
    nx.write_graphml(G, project_dir.joinpath(f"{fname}_association_network_all.graphml"))

    # Filters unconnected
    H = trim_associations(G)
    # nx.write_graphml(G, project_dir.joinpath("test_2.graphml"))
    plot.create_bokeh(H, "MADByTE Similarity Network - Trimmed Spin Systems", project_dir.joinpath(f"{fname}_similarity_network.html"),
        extract_size=extract_size,feature_size=feature_size,)
    nx.write_graphml(H, project_dir.joinpath(f"{fname}_similarity_network_network.graphml"))

    # Join connected associations
    J = hybridize_network(H,idx_master, colors, hppm_error, cppm_error)
    plot.create_bokeh(J, "MADByTE Hybrid Association Network", project_dir.joinpath(f"{fname}_hybrid_network.html"),
        extract_size=extract_size,feature_size=feature_size,)
    nx.write_graphml(J, project_dir.joinpath(f"{fname}_hybrid_network.graphml"))


def neighbour_spins(G, n, samples):
    return list(filter(lambda x: x not in samples, G.neighbors(n)))


def get_node_combos(G, samples):
    neighbours = {n: neighbour_spins(G, n, samples) for n in G}
    new_nodes = set()
    for n, neigh in neighbours.items():
        if G.nodes[n]['_type'] != "spin":
            continue
        these = set((n, *neigh))
        for m in neigh:
            those = set((m, *neighbours[m]))
            if these == those:
                new_nodes.add(frozenset(those))
            else:
                new_nodes.add(frozenset((m, n)))
    return new_nodes


def comp_spins(sig, others, hppm_error, cppm_error):
    hppm_match = lambda x,y: abs(x-y) < hppm_error
    cppm_match = lambda x,y: abs(x-y) < cppm_error
    matching_sig = [other_sig
        for other_sig in chain.from_iterable(others)
        if hppm_match(sig[0], other_sig[0]) and cppm_match(sig[1], other_sig[1])
    ]
    return [round(sum(y) / len(y), 3) for y in zip(*matching_sig)]


def hybridize_spin(G, n, hppm_error, cppm_error):
    get_spins = lambda n: eval(G.nodes[n]['members'])
    systems = [get_spins(m) for m in n]
    min_idx = systems.index(min(systems, key=len))
    min_sys = systems.pop(min_idx)
    # print(n, min_sys)
    # print(systems)
    return list(filter(lambda x: x, [comp_spins(sig, systems, hppm_error, cppm_error) for sig in min_sys]))


def hybridize_network(G, master, colors, hppm_error, cppm_error):
    samples = [n for n, d in G.nodes.items() if d['_type'] != 'spin']
    sample_nodes = [(n,d) for n, d in G.nodes(data=True) if d['_type'] != 'spin']
    node_combos = get_node_combos(G, samples)
    new_nodes = [
        ("/".join(m), {"members": str(hybridize_spin(G, m, hppm_error, cppm_error))})
        for m in node_combos
    ]
    new_edges = []
    for n in new_nodes:
        sys = n[0].split("/")
        for s in sys:
            extract = master.loc[s, 'Found_In']
            new_edges.append((n[0], extract))

    H = nx.Graph()
    H.add_nodes_from(sample_nodes)
    H.add_nodes_from(new_nodes, _color=colors['spin'], _type="spin")
    H.add_edges_from(new_edges, weight=1.0)
    K = filter_singletons(H)
    return K


def get_master(project_dir):
    master_path = project_dir.joinpath("Spin_Systems_Master.json")
    return pd.read_json(master_path)


def save_master(project_dir, master):
    master_path = project_dir.joinpath("Spin_Systems_Master.json")
    master_path_csv = project_dir.joinpath("Spin_Systems_Master.csv")
    master.to_json(master_path)
    master.to_csv(master_path_csv, index=False)


def outside_window(c_df, tol=0.01, col="Ha"):
    """
    Take connected component dataframe and determine if the connected components
    span larger than the allowed bin width
    """
    c_range = c_df[col].max()-c_df[col].min()
    return c_range > tol


def split_bin(c_df, config, tol=0.01, col="Ha", name="MADByTE"):
    """
    Take connected component DF and split it into the appropriate number of bins
    To be called if outside_window == True
    """
    c_df = c_df.copy()
    c_range = c_df[col].max()-c_df[col].min()
    num_bins = int(np.ceil(c_range / tol))
    bins = pd.cut(c_df[col], num_bins, labels=False)
    logger.warning(f"{name} - Splitting into {num_bins} bins")
    c_df["bins"] = bins
    groups = c_df.groupby("bins")
    return [df for _, df in groups]

def tocsy_dereplicate_protons(df, col="Ha", tol=0.01, name="MADByTE"):
    """Take dataframe and perform Rtree comparison to replicate

    Args:
        df (pd.DataFrame): Data to work on

    Returns:
        pd.DataFrame: New dataframe with averaged data
    """
    # Create a DF for R-Tree comparison
    protons = df[[col]].drop_duplicates().reset_index(drop=True)
    logger.info(f"{name} - Initially {len(protons)} {col} protons in TOCSY")
    protons["PHASE"] = 1.0
    config = {
        "ColsToMatch": [col, "PHASE"],
        "Tolerances": {
            f"{col}": tol/2.0,
            "PHASE": 0.1,
        }
    }
    dereplicator.gen_error_cols(protons, config)
    rects = dereplicator.get_rects(protons, config)
    rt = dereplicator.build_rtree(rects, config)
    con_comps = dereplicator.gen_con_comps(rt, rects)
    new_data = []
    for c in con_comps:
        c_df = protons.iloc[list(c)]
        if outside_window(c_df, tol=tol, col=col):
            logger.warning(f"{name} - BIN IS LARGER THAN ALLOWED TOLERANCE: {c_df[config['ColsToMatch']].to_dict()}")
            split_bins = split_bin(c_df, config=config, col=col, name=name)
            [new_data.append(dereplicator.average_data(b, config=config)) for b in split_bins]
        else:
            new_data.append(dereplicator.average_data(c_df, config=config))
    unique_prot = pd.DataFrame(new_data).round(3)

    points = unique_prot[col].values
    logger.info(f"{name} - Reduced to {len(points)} {col} protons in TOCSY")
    for idx, row in df.iterrows():
        closest = points[np.abs(points-row[col]).argmin()]
        # Double check in log
        if np.abs(closest - row[col]) > tol:
            logger.warning("ERROR - reassignment is outside of tolerance")
        logger.debug(f"{name} : {col} reassignment - inital value: {row[col]} - final value: {closest}")
        df.loc[idx, col] = closest
