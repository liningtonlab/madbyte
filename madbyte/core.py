import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

import madbyte.utils as utils
from madbyte.filter import Filter, SOLVENT_DICT
from madbyte.logging import get_logger
from madbyte.parsers import acd_sim_parser, mestrenova_parser, topspin_parser

logger = get_logger("MADByTE")


def construct_spin_system(
    name,
    input_dir,
    project_dir,
    nmr_data_type="Bruker",
    entity="Extract",
    hppm_error=0.05,
    tocsy_error=0.05,
    merge_multiplets=True,
    restart=True,
    solvent="dmso",
):
    assert solvent in SOLVENT_DICT.keys()
    input_dir = Path(input_dir)
    project_dir = Path(project_dir)
    output_dir = project_dir.joinpath(name)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        logger.info(f"Data directory created for {name}")
    except FileExistsError:
        logger.debug(f"Data directory already exists : {name}")
        pass

    master = utils.get_master(project_dir)

    # Speed up already processed unless explicitly told to reprocess
    if name in list(master['Found_In'].unique()) and not restart:
        logger.info(f"Already processed {name}")
        return

    if entity == "Simulacrum":
        ##Important note: the simulacrum method is structured to be specific to ACD formatted data.##
        hsqc_data, tocsy_data = acd_sim_parser(name, input_dir, output_dir)

    elif entity == "Extract":
        if nmr_data_type == "Bruker":
            hsqc_data, tocsy_data = topspin_parser(name, input_dir, output_dir)
        elif nmr_data_type == "Mestrenova":
            hsqc_data, tocsy_data = mestrenova_parser(name,input_dir,output_dir)
        else:
            raise Exception("nmr_data_type not valid!")

    else:
        raise Exception("entity is invalid!")

    # initial data stats
    start_hsqc = len(hsqc_data)
    start_tocsy = len(tocsy_data)

    ## HSQC Filtration step
    ## Define filters
    # These represent stripes of Hppm which have been forbidden
    solvent_filters = SOLVENT_DICT[solvent]

    restricted_zones = [
        Filter(["H_PPM", "C_PPM"], [(0.0, 2.4), (100.0,  210.0)]), # Solvent band and under, square 1
        Filter(["H_PPM", "C_PPM"], [(0.0, 7.0), (170.0, 210.0)]), # Region of atypical shift patterns
        Filter(["H_PPM", "C_PPM"], [(7.0, 13.0), (0.0, 50.0)]), # Upper left corner of the spectra
    ]

    # Apply filters
    logger.info(f"{name} : Applying HSQC filters")
    hsqc_filters = solvent_filters + restricted_zones
    utils.apply_filters(hsqc_data, hsqc_filters)

    ## TOCSY Filtration step
    tocsy_filters = [
        Filter(["Ha", "Hb"], [(0.0, 2.5), (0.0, 2.5)]) # Very noisy data in general
    ]
    # Remove self-correlated points
    logger.info(f"{name} : Applying TOCSY filters")
    utils.apply_filters(tocsy_data, tocsy_filters, name=name)
    utils.remove_self_corr(tocsy_data, tolerance=tocsy_error, name=name)

    ## HSQC Point Consensus
    # Save temp files
    hsqc_data.to_json(output_dir.joinpath(f"{name}_HSQC_Pre_Decoupling.json"), orient="records")
    tocsy_data.to_json(output_dir.joinpath(f"{name}_TOCSY_Pre_Decoupling.json"), orient="records")

    if merge_multiplets:
        logger.info(f"{name} : Merging multiplets in HSQC data")
        hsqc_data = utils.mutliplet_merger(hsqc_data, name=name)
        hsqc_data.to_json(output_dir.joinpath(f"{name}_HSQC_MultipletMerged.json"), orient="records")

    ## TOCSY Peak Consensus
    # Merge F2 protons
    utils.tocsy_dereplicate_protons(tocsy_data, col="Ha", name=name)
    # Align F1 to F2
    utils.self_align_tocsy(tocsy_data, tolerance=0.02, name=name)
    # Re-merge F1 (should only merge non-aligned signals)
    utils.tocsy_dereplicate_protons(tocsy_data, col="Hb", name=name)

    utils.tocsy_in_hsqc(tocsy_data, hsqc_data, tolerance=hppm_error, name=name)

    ## Quick summary of how much data was reduced from consideration
    # final data stats
    final_hsqc = len(hsqc_data)
    final_tocsy = len(tocsy_data)
    logger.info(f"{name} : HSQC reduced from {start_hsqc} peaks to {final_hsqc} peaks")
    logger.info(f"{name} : TOCSY reduced from {start_tocsy} peaks to {final_tocsy} peaks")

    # Save temp files
    hsqc_data.to_json(output_dir.joinpath(f"{name}_HSQC_Data.json"), orient="records")
    tocsy_data.to_json(output_dir.joinpath(f"{name}_TOCSY_Data.json"), orient="records")

    ## Spin System Construction
    spin_systems = utils.build_spin_systems(tocsy_data, hsqc_data, name, output_dir, tolerance=hppm_error)
    return name, spin_systems


def construct_correlation_matrix(project_dir, hppm_error=0.05, cppm_error=0.5):
    project_dir = Path(project_dir)
    ss_df = utils.load_spin_systems(project_dir)
    corr_mat = utils.compute_corr_matrix(ss_df, hppm_error, cppm_error)
    corr_mat.to_json(project_dir.joinpath("correlation_matrix.json"))
    corr_mat.to_csv(project_dir.joinpath("correlation_matrix.csv"))


def create_outputs(
    project_dir,
    fname="MADByTE",
    threshold=0.5,
    hppm_error=0.05,
    cppm_error=0.5,
    colors=None,
    extract_size=15,
    feature_size=10,
    max_system_size=20,
):
    if not colors:
        colors = {
            "spin": "#009999", # GREY
            "extract": "#ff3333", # RED
            "standard": "#0FFBFF", # Black
        }
    project_dir = Path(project_dir)
    corr_mat = pd.read_json(project_dir.joinpath("correlation_matrix.json"))
    master = pd.read_json(project_dir.joinpath("Spin_Systems_Master.json"), precise_float=True)
    utils.association_network(project_dir, fname, corr_mat, master, colors, cutoff=threshold, hppm_error=hppm_error,
        cppm_error=cppm_error,extract_size=extract_size,feature_size=feature_size,max_system_size=max_system_size,)
