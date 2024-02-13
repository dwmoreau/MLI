# unit_cell_ML

## Dependencies
1: CCDC API
You need access through a subscription. These are the relevant links for LBL:

https://software.chem.ucla.edu/CSD/

https://software.chem.ucla.edu/CSD/protected/distribution/release_install_2022_3.pdf

I would recommend installing mamba, adding programs on top of the CCDC's python package is incredibly slow

2: mpi4py

I am running this on a M1 mac. I just install mpi4py and mpich through conda-forge

3: pandas & pyarrow

I am saving my dataframes to parquet format. It saves so much time over json format.

4: tensorflow

The basic ML uses tensorflow. At no point do you need the CCDC's api and tensorflow at the same time. So I install this is in a separate environment

## Dataset generation

### Step 1: Run convert_ccdc_mpi.py

This goes through all the entries in the CSD and does a quality control check and duplicate removal check. The output file is stored in ...data/unique_entries.parquet The QC checks for the following:

  1: No chemical formula

  2: unknown lattice centering

  3: Unable to read spacegroup number and setting

  4: Space group number > 230

  5: No listed crystal system

  6: Crystal system and space group number are consistent

  7: Unit cell is consistent with crystal system

  8: Chemical formula is parseable

  9: Unit cell volume is within an order of magnitude of 18 X number of non hydrogen atoms in the unit cell

Each entry and relevant information is stored in pandas data frames. There will be one data frame for each MPI rank.

A duplicate removal step is then performed. A duplicate is defined as:

1: Belongs to the same crystal family

2: Has unit cell volume within 5%

The duplicate removal step was a stand alone function. This was recently combined into convert_ccdc_mpi.py to reduce the number of steps. The stand alone function works, the combination has not been bug tested.

### Step 2: Run GenerateDataset_mpi.py

This uses the unique_entries.parquet file from Step 1 to generate a data set. This data set includes the following components for each entry:
- identifier
- unit_cell: a, b, c, alpha, beta, gamma
- pattern: diffraction pattern normalized so np.linalg.norm(pattern) = 1
- volume: unit cell volume
- crystal_family
- centering
- s6 and g6: these have not been verified for accuracy.
- peak lists.

Peak list description. There are 6 different peak lists.
  - all: These are all the peaks reported by the CCDC api.
  - sa: These are all the peaks with the systematic absences removed.
  - strong: The weak peaks are removed. This is based on the powder diffraction pattern being greater than 0.001. The pattern is normalized such that $\sqrt{\sum_i I_i^2} = 1$.
  - 2der: There must be a second derivative present in the diffraction pattern
  - overlaps: If multiple peaks are within overlap_threshold, only the peak at the largest value of the diffraction pattern is kept
  - intersect: Intersection of all the above

For each peak list, the following information is kept.
- theta2
- d_spacing
- h
- k
- l
- order: This is the index for this peak in the 'all' peak list.

To access the d_spacings for the intersect peak list you would use dataframe['d_spacing_all'] for example

There are a couple of editable parameters within the file.

1: entries_per_group: This is the maximum number of entries to include in the data set for each crystal family

2: peak_length: This saves this many d-spacings for the points format

3: Diffraction pattern generation parameters
  - fwhm: full width half max in degrees
  - theta2_min / theta2_max: Minimum and maximum theta2 that the diffraction pattern is generated to.
    
4: overlap_threshold: maximum distance between two peaks to consider then to be overlaps

A couple of details:
- I have it hard coded to only use the entries with primitive centering
- The outputs are grouped by crystal family, so the output files are 'dataset_cubic.parquet' for example.

## Machine Learning

This implements very basic ML models with emphasis placed on visualization of the results. The data management is kept flexible and general for reuse.

All of the implemented ML models are based on points data sets. These are in PointsRegression.py and PointsClassification.py. These two inherit from basic_model.py

