from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import Benchmark

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metabci_utils import create_ssvep_paradigm, channel_selection
    from metabci.brainda.datasets import BETA
    # This is a ssvep dataset, we add the attach Yf template and the filterbank
    # to the dataset
    from metabci.brainda.algorithms.decomposition import (
        generate_cca_references
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BETA"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'channel': ['occipital_9', 'all'],
        'duration': [i for i in np.arange(0.2, 2, 0.2)],'subject': [i for i in range(1, 16)],
        'duration': [i for i in np.arange(0.2, 3, 0.2)],'subject': [i for i in range(16, 71)],

    }
    
    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Constant parameters
        visual_delay = 0.14
        srate = 250

        # Initialize the dataset
        dataset = BETA()
        events = sorted(list(dataset.events.keys()))

        if self.channel == 'all':
            channels = dataset.channels
        elif self.channel == 'occipital_9':
            channels = channel_selection('occipital_9')

        # create the paradigm
        start_pnt = dataset.events[events[0]][1][0]
        paradigm = create_ssvep_paradigm(
            srate, channels,
            [(start_pnt+visual_delay, start_pnt+self.duration+visual_delay)],
            events
        )

        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[self.subject],
            return_concat=True,
            n_jobs=8,
            verbose=False
        )

        freqs = [dataset.get_freq(event) for event in events]
        phases = [dataset.get_phase(event) for event in events]

        Yf = generate_cca_references(
            freqs,
            srate,
            self.duration,
            phases=phases,
            n_harmonics=5)
        
        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            X=X,                         
            y=y,                         
            meta=meta,                   
            srate=srate,                 
            duration=self.duration,      
            dataset_name=dataset.dataset_code,  
            special_data=dict(Yf=Yf)     
        )

