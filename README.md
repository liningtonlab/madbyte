# MADByTE

MADByTE stands for **M**etabolomics **A**nd **D**ereplication **By** **T**wo-dimensional **E**xperiments.

MADByTE allows for comparative analysis of NMR spectra from large sample sets, simultaneously, deriving shared structural features from heteronuclear and homonuclear experiments. Using large sample sets, the common features between each sample can be visualized to aid in sample prioritization and structure characterization of scaffolds present.

If you use this tool, please cite it:

[DOI: 10.1021/acs.jnatprod.0c01076](https://doi.org/10.1021/acs.jnatprod.0c01076)

More information about the MADByTE program, including news, examples, and detailed tutorials can be found at our website:

https://www.madbytenmr.com/

Documentation for usage and installation can be found in `Documentation/MADByTE_User_Guide.pdf`

Please download the latest release and follow the instructions in the documentation for installation.

### Basic Usage

#### DO NOT INSTALL MADBYTE USING ANACONDA NAVIGATOR - IT WILL FAIL INSTALLATION. 

We **Highly** recommend installing through the included .bat script or installing manually with `conda env create -f environment.yml`. 

If you have followed the installation guide and setup the MADByTE Python virtual environment, then navigate to the root directory of the code using your console/terminal (the directory this `README` is located in).

Ensure your virtual environment is activated and run the launcher script.

```bash
conda activate madbyte
python madbyte_gui.py
```

