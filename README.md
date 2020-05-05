# MADByTE

MADByTE stands for Metabolomics And Dereplication By Two-dimensional Experiments.

MADByTE allows for comparative analysis of NMR spectra from large sample sets, simultaneously, deriving shared structural features from heteronuclear and homonuclear experiments. Using large sample sets, the common features between each sample can be visualized to aid in sample prioritization and structure characterization of scaffolds present.

https://www.madbyte.org/


## Setup


The easiest way to install madbyte is using the Anaconda Python distribution. To install dependencies in a virtual environment:

```
conda create -y -n madbyte python=3.7
conda install -y -n madbyte -c conda-forge libspatialindex rtree joblib networkx bokeh>=1.0 pandas matplotlib tqdm pyqt pyqtgraph
conda install -y -n madbyte -c bioconda nmrglue
```

**On Windows** You can simply run the `madbyte_install.bat` script and then use the `madbyte.bat` script to launch the GUI.


## To run

`conda activate madbyte && python madbyte_gui.py`


## Tests

Make sure you have pytest installed `conda install pytest`

Then from the root git directory (the one with this README), simple run the `pytest` command. It will autodiscover and run all the tests.
