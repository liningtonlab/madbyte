# MADByTE

MADByTE stands for Metabolomics And Dereplication By Two-dimensional Experiments.

MADByTE allows for comparative analysis of NMR spectra from large sample sets, simultaneously, deriving shared structural features from heteronuclear and homonuclear experiments. Using large sample sets, the common features between each sample can be visualized to aid in sample prioritization and structure characterization of scaffolds present.

https://www.madbyte.org/


## Setup

For a stable version, please download the latest release. Alternatively, you can clone this repo development.

Full documentation can be found in `Documentation/MADByTE_Install_Guide.pdf`.

The easiest way to install madbyte is using the Anaconda Python distribution. To install dependencies in a virtual environment:

```
conda create -y -n madbyte python=3.7
conda install -y -n madbyte -c conda-forge libspatialindex rtree joblib networkx bokeh\>=1.0 pandas matplotlib tqdm pyqt pyqtgraph
conda install -y -n madbyte -c bioconda nmrglue
```

**On Windows** You can simply run the `madbyte_install.bat` script and then use the `madbyte.bat` script to launch the GUI.


## To run

A full user guide and examples can be found in the `Documentation` directory.

If you have already installed the madbyte virtual envrionment, then running the GUI may be as simple as:

`conda activate madbyte && python madbyte_gui.py`




## Tests

Make sure you have pytest installed `conda install pytest`

Then from the root git directory (the one with this README), simple run the `pytest` command. It will autodiscover and run all the tests.
