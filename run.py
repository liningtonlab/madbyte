import shutil
from pathlib import Path
import logging

from madbyte import (correlation_matrix_generation, generate_network,
                     spin_system_construction)
from madbyte.logging import setup_logging


def main():
    # Change this
    input_dir = Path("./example_data")
    assert input_dir.exists()
    # project_dir = Path("example_output")
    project_dir = input_dir.joinpath("Output")
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(exist_ok=True)
    setup_logging("MADByTE_Log.txt", fpath=project_dir, level=logging.DEBUG)
    sample_list = []
    for d in input_dir.glob("*"):
        if not d.is_dir() or d == project_dir:
            continue
        sample_list.append(d.name)
    spin_system_construction(
        sample_list=sample_list,
        input_dir=input_dir,
        project_dir=project_dir,
        nmr_data_type="Bruker",
        entity="Extract",
        hppm_error=0.05,
        tocsy_error=0.03,
        n_jobs=-1,
    )

    correlation_matrix_generation(hppm_error=0.05, cppm_error=0.3, project_dir=project_dir)
    generate_network(
            project_dir=project_dir, 
            threshold=0.5, 
            fname="MADByTE",
            hppm_error=0.05,
            cppm_error=0.4,
        )


if __name__ == "__main__":
    main()
