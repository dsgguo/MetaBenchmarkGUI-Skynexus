from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import Benchmark

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metabci_utils import (
        create_p300_paradigm, p300_channel_selection,)

    from metabci.brainda.datasets import Cattan_P300



# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Cattan_P300"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'duration': [i for i in np.arange(0.3, 0.7, 0.1)],  
        'subject': [i for i in range(1, 13)] 
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Constant parameters
        srate = 500

        # Initialize the dataset
        dataset = Cattan_P300()

        channels = dataset.channels

        # create the paradigm
        paradigm = create_p300_paradigm(
            channels,
        )

        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[self.subject],
            return_concat=True,
            n_jobs=8,
            verbose=False
        )
        
        if dataset.dataset_code == "Cattan_P300":
            special_data = dataset
        else:
            special_data = None

            
        return dict(
            X=X,                         
            y=y,                         
            meta=meta,                   
            srate=srate,                 
            duration=self.duration,      
            dataset_name=dataset.dataset_code,  
            special_data=special_data,     
        )
