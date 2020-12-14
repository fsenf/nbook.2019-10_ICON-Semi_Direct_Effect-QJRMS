# Jupyter Notebooks for Plotting Analysis of the "Semi-Direct Effect over Germany with ICON-LEM" study

## Paper
This is a collection of jupyter notebooks that have been used to prepare plots for the paper project

Senf, Quaas and Tegen, **Absorbing aerosol decreases cloud cover in cloud-resolving simulations over Germany**, QJRMS, revision submitted in Dec. 2020.

A pre-print version of the manuscript can be downloaded at https://doi.org/10.1002/essoar.10505373.1


## Overview
An **overview** is provided [here](nbooks/00-Overview.ipynb) as starting point. All figures from the main manuscript is listed and links to the corresponding notebooks are given. 


## Underlying Data
The underlying post-processed data have been archived in the DKRZ database CERA. A TAR file can be downloaded from  [data url](http://cera-www.dkrz.de/WDCC/ui/Compact.jsp?acronym=DKRZ_LTA_1174_ds00001). CERA requires registration.


## Dependencies
All notebooks were run with a python 3.7 kernel on DKRZ mistral. 

### Main Python Packages

* `numpy`
* `pylab`
* `xarray`
* `tropy`
* `nawdex_analysis`

### Comments on Own Packages and Tools
Some Notebooks depend on **my python package** `nawdex_analysis` that was developed for another analysis (CRE over the North Atlantic). The package is found [https://github.com/fsenf/proj.nawdex_analysis](https://github.com/fsenf/proj.nawdex_analysis). This package further depends on my former tool box: `tropy` package which are collected here [https://github.com/fsenf/proj.tropy](https://github.com/fsenf/proj.tropy).

Some local **tools** have been used which were not collected into a package. These tools have been copied to the `../tools` directory of this repository. To use the tool modules, please adjust the `tools_dir` in the concerning notebooks.



