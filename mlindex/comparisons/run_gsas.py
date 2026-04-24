import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import io
import contextlib
from candidate_cells import Script
import sys


file_name = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/opxrd/CNRS_output_data_verified_final3.json'
df = pd.read_json(file_name)
gsas_dir = 'results/gsas'

for index in tqdm(range(len(df))):
    try:
	    entry = df.iloc[index]
	    tag = Path(entry.file_name).name.split('.json')[0]
	    dat_file_name = os.path.join(
	        '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/comparisons/results/gsas/',
	        f'{tag}',
	        f'{tag}.dat'
	    )
	    if Path(dat_file_name.replace('.dat', '.log')).exists():
	        continue
	
	    if entry.lattice_system == 'triclinic':
	        n_searches = '2 2 2 4 4 4 4 6 6 6 6 12 12 24'
	    elif entry.lattice_system == 'monoclinic':
	        n_searches = '2 2 2 4 4 4 4 6 6 6 6 12 12 1'
	    else:
	        n_searches = '2 2 2 4 4 4 4 6 6 6 6 1 1 1'
	    sys.argv = [
	        "run_gsas.py",
	        f"input.peak_list={dat_file_name}",
	        "multiprocessing.nproc=24",
	        f"search.n_searches='{n_searches}'"
	    ]
	    script = Script()
	    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
	        script.run()
    except:
        print('FAILED')