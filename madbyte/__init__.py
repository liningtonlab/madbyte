from pathlib import Path
from typing import List, Optional

from joblib import Parallel, delayed

from madbyte import core, utils
from madbyte.logging import get_logger

__version__ = '1.0.0'

logger = get_logger("MADByTE")

### Wrapper for implementation in GUI
def spin_system_construction(
    sample_list: List[str],
    input_dir: str,
    nmr_data_type : str,
    entity: str,
    hppm_error: float,
    tocsy_error: float,
    project_dir: str,
    merge_multiplets: bool = True,
    restart: bool = False,
    n_jobs: int = -1,
):
    logger.info(f"Processing {len(sample_list)} samples...")
    project_dir = Path(project_dir)
    utils.establish_file_structure(project_dir)
    def do_calc(n):
        logger = get_logger("MADByTE")
        try:
            result = core.construct_spin_system(
                name=n,
                input_dir=input_dir,
                project_dir=project_dir,
                nmr_data_type=nmr_data_type,
                entity=entity,
                hppm_error=hppm_error,
                tocsy_error=tocsy_error,
                merge_multiplets=merge_multiplets,
                restart=restart,
            )
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc(e))
            logger.error(f"{n} failed...")
        else:
            logger.info(f"{n} finished processing")
            return result

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(do_calc)(n) for n in sample_list
    )
    master = utils.get_master(project_dir)
    for (name, ss) in results:
        master = utils.add_spin_systems_to_master(name, ss, master)
    utils.save_master(project_dir, master)


def correlation_matrix_generation(
    hppm_error: float,
    cppm_error: float,
    project_dir: str,
):
    logger.info("Generating correlation network...")
    core.construct_correlation_matrix(
        project_dir=project_dir,
        hppm_error=hppm_error,
        cppm_error=cppm_error,
    )


def generate_network(
    project_dir: str,
    threshold: float,
    fname: str,
    cppm_error: float,
    hppm_error: float,
    colors: Optional[dict] = None,
    extract_size: int = 15,
    feature_size: int = 10,
    max_system_size: int = 20,
):
    logger.info("Generating association network outputs...")
    core.create_outputs(
        project_dir=project_dir,
        fname=fname,
        threshold=threshold,
        hppm_error=hppm_error,
        cppm_error=cppm_error,
        colors=colors,
        extract_size=extract_size,
        feature_size=feature_size,
        max_system_size=max_system_size,
    )
