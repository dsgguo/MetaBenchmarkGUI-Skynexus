from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import Benchmark

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metabci_utils import create_mi_paradigm, channel_selection
    from metabci.brainda.datasets import BNCI2014001
    # This is a mi dataset, we add the attach Yf template and the filterbank
    # to the dataset
    from metabci.brainda.algorithms.decomposition import (
        generate_cca_references
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "bnci2014001"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'duration': [i for i in np.arange(2, 6, 0.2)],
        'subject': [i for i in range(1, 10)]  # 1 to 9
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Constant parameters
        srate = 250

        # Initialize the dataset
        dataset = BNCI2014001()

        channels = dataset.channels

        # create the paradigm
        paradigm = create_mi_paradigm(
            channels=channels,
            events=['left_hand', 'right_hand', 'feet', 'tongue'],
            interval=[(2, 6)], 
            srate=srate
        )


        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[self.subject],
            return_concat=True,
            n_jobs=None,
            verbose=False
        )

        if dataset.dataset_code == "bnci2014001":
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
