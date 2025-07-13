# CMIP6-ICESat2-SeaIce-Petty-2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15849469.svg)](https://doi.org/10.5281/zenodo.15849469)

This repository contains the analysis code and data processing for [Petty et al. (2025)](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-766/), including comparisons of CMIP6 model outputs with ICESat-2 sea ice freeboard and thickness and OSI SAF sea ice concentration.

Contact: Chris Cardinale | ccardina@umd.edu

## Overview

This repository provides Jupyter Notebooks that demonstrate the primary aspects of our analysis workflow, including:

- **Data Access**: Code to access wrangled/regridded CMIP6 data and observations from AWS S3
- **Statistical Analysis**: Code to estimate internal variability and model plausibility indices  
- **Visualization**: Code to produce key figures shown in the manuscript

## Getting Started

### Prerequisites & Environment Setup

Different options for setting up the required Python environment:

#### Using Conda
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate seaice-env
```

#### Using pip
```bash
# Create a new virtual environment with uv
uv venv
# Activate the virtual environment
source .venv/bin/activate  
# Install packages from requirements.txt
uv pip install -r requirements.txt
```
The environment.yml and requirements.txt are untested and changes may be needed.

## Notebook Descriptions

### Results.ipynb
- **Purpose**: Quick figure reproduction using preprocessed data
- **Runtime**: A few minutes
- **Data**: Loads precomputed climatologies, error statistics, and internal variability
- **Use Case**: Best for reproducing manuscript figures efficiently

### Data-processing_results.ipynb
- **Purpose**: Complete data processing workflow from raw CMIP6 outputs
- **Runtime**: Full CMIP6 download: ~25 minutes - 2 hours (if not loading preprocessed internal variability data)
- **Data**: Processes raw CMIP6 data, computes all statistics from scratch
- **Use Case**: Best for understanding the full methodology or modifying analysis parameters

## Important Notes
- **Data Availability**: Due to shifting availability on ESGF data nodes, some figures may not precisely match the published manuscript
  - **EC-Earth3**:
    - 7 additional members that include sea ice concentration
    - 8 additional members that include both snow thickness and sea ice thickness 
  - **EC-Earth3-CC**:
    - 11 additional members that include sea ice concentration
    - 10 additional members that include sea ice thickness
    - 9 additional members that include snow thickness
    - Impacts the CMIP6 mean internal variability, with minimal changes to the plausibility index values
  - **EC-Earth3-Veg**:
    - 2 additional members that include sea ice concentration
    - 2 additional members that include sea ice thickness 
  - **NorESM2-LM**:
    - 1 additional members that include sea ice concentration
    - 7 additional members that include sea ice thickness
    - 8 additional members that include snow thickness
  - **AWI-CM-1-1-MR**
    - September Arctic total freeboard is notably higher in the latest load from ESGF potentially resulting from missing data that was filled with zeros in our data processing (the plausibility value for September Arctic total freeboard increased by 1).
- **Memory Requirements**: Loading all models requires significant RAM; consider using model subsets for initial exploration or Results.ipynb
- **Cloud Access**: All data is accessed from AWS S3; no local storage required
- **Regridding**: Model regridding is not shown in this repository, contact for more information

## Repository Structure

```
├── Results.ipynb                 # Main figure reproduction notebook (preprocessed data)
├── Data-processing_results.ipynb # Full data processing workflow
├── functions.py                  # Custom analysis functions
├── environment.yml               # Conda environment file
├── requirements.txt              # Pip requirements file
├── files/                        # Supporting data files
│   ├── NSIDC-0780_SeaIceRegions_NH_v1.0.*
│   ├── NSIDC-0780_SeaIceRegions_SH-NASA_v1.0.*
│   ├── NSIDC0771_CellArea_PS_N25km_v1.0.nc
│   └── NSIDC0771_CellArea_PS_S25km_v1.0.nc
```
