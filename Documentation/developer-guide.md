### TODO: Fill this out

The easiest way to install madbyte is using the Anaconda Python distribution. To install dependencies in a virtual environment:

```bash
conda env create -f environment.yml
```

**On Windows** You can simply run the `madbyte_install.bat` script and then use the `madbyte.bat` script to launch the GUI.

## To run

A full user guide and examples can be found in the `Documentation` directory.

If you have already installed the madbyte virtual envrionment, then running the GUI may be as simple as:

`conda activate madbyte && python madbyte_gui.py`

## Tests

Make sure you have pytest installed `conda install pytest`

Then from the root git directory (the one with this README), simple run the `pytest` command. It will autodiscover and run all the tests.
