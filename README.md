# MADByTE

Code for Joe's project

## To run

`conda activate madbyte && python madbyte_gui.py`

## Setup
Install dependencies:

```
conda create -y -n madbyte python=3.7
conda install -y -n madbyte -c conda-forge libspatialindex rtree joblib networkx bokeh>=1.0 pandas matplotlib tqdm pyqt pyqtgraph
conda install -y -n madbyte -c bioconda nmrglue
```

**On Windows** You can simply run the `madbyte_install.bat` script and then use the `madbyte.bat` script to launch the GUI.


## Tests

Make sure you have pytest installed `conda install pytest`

Then from the root git directory (the on with this README), simple run the `pytest` command. It will autodiscover and run all the tests.
