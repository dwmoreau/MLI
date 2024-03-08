# MLI

## Dependencies
Two separate python environments are used, one for dataset generation and another for ML training. Dataset generation requires the CCDC python API. It is a fairly complex library and I felt it was best to keep it separate from the ML libraries.

### For the dataset generation environment:
1: CCDC API
You need access through a subscription. These are the relevant links for LBL:

https://software.chem.ucla.edu/CSD/

Command to activate the csd python environment: "source /Path/To/CCDCInstallation/ccdc-software/csd-python-api/miniconda/bin/activate base"

I would recommend installing mamba, adding programs on top of the CCDC's python package is incredibly slow

Additional python libraries, all can be installed through conda-forge

- mpi4py & mpich/openmpi

- pandas & pyarrow

I am saving my dataframes to parquet format which requires the pyarrow library. It saves so much time over json format. In retrospect, saving to an hdf5 format would probably be best, but I don't feel like it would be worth refactoring the code.

- gemmi

This is a crystallography library that I use to extract information from cif files and parse and convert spacegroup symbols.

- ipython & jupyterlab

I don't believe these are strictly neccessary. I do a lot of testing in Jupyter notebooks.

### For the ML training environment:

- tensorflow

- sklearn

- pandas & pyarrow

- gemmi
  
- ipython & jupyterlab

- mpi4py & mpich/openmpi


## Dataset generation

### Step 1: Prepare the CSD database.

In the CCDI environment, run ParseCSD.py. This is setup to be run parallelized with mpi, to run with 8 ranks for example: mpiexec -n 8 python ParseCSD.py. This goes through all the entries in the CSD and does a quality control check. Each entry and relevant information is stored in pandas data frames. The outputs from each rank are saved individually to data/csd_rank.parquet - There will be one data frame for each MPI rank. The QC checks for the following:

  1: No chemical formula

  2: unknown lattice centering

  3: Unable to read spacegroup number and setting

  4: Space group number > 230

  5: No listed crystal system

  6: Crystal system and space group number are consistent

  7: Unit cell is consistent with crystal system

  8: Chemical formula is parseable

  9: Unit cell volume is within an order of magnitude of 18 X number of non hydrogen atoms in the unit cell

Entries that do not pass the QC check are stored in files data/failed_read_csd_rank.parquet.

### Step 2: Prepare the COD database.

The COD database can be obtained via rsync. The command can be found at .../data/download_cod.sh. This is a single line rsync command, just change the destination directory.

Next, run ParseCOD.py. This is essentially the same processes as ParseCSD.py, also run it in parallel.

### Step 3: Duplicate removal and database combination.
A duplicate is defined as:

1: Belongs to the same crystal family

2: Has unit cell volume within 5%

The program RemoveDuplicates.py removes duplicates from the CSD and COD databases separately and needs to be run twice, once for the CSD and COD. Edit the first two lines to specify the database and the number of ranks used to run ParseCSD.py or ParseCOD.py. This reads each file resulting from ParseCSD.py or ParseCOD.py and removes duplicates within a database. The unique entries are save to data/unique_entries_csd.parquet or data/unique_entries_cod.parquet. This program is not parallelized. It must be run serially.

Run CombineDatabases.py to combine the unique CSD and COD entries. This file is stored at data/unique_cod_entries_not_in_csd.parquet. This program is not parallelized. It must be run serially.

### Step 4: Run GenerateDataset.py

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

