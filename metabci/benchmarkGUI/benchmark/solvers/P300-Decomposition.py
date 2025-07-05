import importlib
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import NoCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from metabci.brainda.algorithms.decomposition.base import TimeDecodeTool
    from benchmark_utils.metabci_utils import gen_p300_filterbank
    from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_char_indices,
    match_char_kfold_indices
    )
    # import your reusable functions here
    import metabci.brainda.algorithms.decomposition as decomposition
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    @property
    def _stopping_criterion(self):
        """Use NoCriterion to run for fixed iterations."""
        return NoCriterion()

    # Name to select the solver in the CLI and to display the results.
    name = 'P300-docomposition-algo'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "model": [
            "LDA",
            "SKLDA",
            "STDA",
            "DCPM",
            None
        ],
        "custom_model": [None],  
        "module_name": ["decomposition", "algorithm"]  
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []
    
    sampling_strategy = "iteration"


    def set_objective(self, X, y, srate, special_data, meta):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        self.special_data = special_data
        self.meta = meta

        self.current_k = 0
        self.k_loo = 12

        self.X_test_current = None
        self.X_train_current = None
        self.y_train_current = None
        self.label_test = None

        set_random_seeds(38)
        self.indices = generate_char_indices(self.meta, kfold=self.k_loo)
        self.time_decode_tool = TimeDecodeTool(dataset=self.special_data)

        try:
            if self.module_name == "decomposition":
                module = importlib.import_module("metabci.brainda.algorithms.decomposition")
            elif self.module_name == "algorithm":
                module = importlib.import_module("metabci.benchmarkGUI.benchmark.algorithm")  
            else:
                raise ValueError(f"模块 {self.module_name} 不支持，请选择 'decomposition' 或 'algorithm'。")

            model_to_load = self.custom_model or self.model

            if model_to_load[:2] == "FB":
                filterbank, filterweights = gen_p300_filterbank(srate=srate)
                self.clf = getattr(module, model_to_load)(
                    filterbank=filterbank, filterweights=filterweights)
            else:
                if model_to_load == "LDA":
                    self.clf = LinearDiscriminantAnalysis()
                elif model_to_load == "DCPM":
                    self.clf = getattr(module, model_to_load)(n_components=8,)
                elif model_to_load == "SKLDA":
                    self.clf = getattr(module, model_to_load)()
                else: # STDA
                    self.clf = getattr(module, model_to_load)()

        except (AttributeError, ImportError, ValueError) as e:
            raise ValueError(f"模型 {model_to_load} 在模块 {self.module_name} 中未找到，请确认是否已正确定义。") from e

    def _prepare_iteration_data(self, k):
        """Prepare the data for the kth iteration"""
        train_id, val_id, test_id = match_char_kfold_indices(k, self.meta, self.indices)
        train_ind = np.concatenate((train_id, val_id))
        self.test_ind = test_id
        
        X_train_t = [self.X[i] for i in train_ind]
        y_train_t = [self.y[i] for i in train_ind]
        Key = self.meta.event[train_ind]
        
        y_train_tar = self.time_decode_tool.target_calibrate(y_train_t, Key)
        self.label_test = list(self.meta.event[test_id])
        
        self.X_train_current = np.concatenate(X_train_t)
        self.y_train_current = np.concatenate(y_train_tar)
        
        X_test_t = [self.X[i] for i in test_id]
        y_test_t = [self.y[i] for i in test_id]
        X_test_sort, y_test_sort = self.time_decode_tool.epoch_sort(X_test_t, y_test_t)
        
        self.X_test_current = X_test_sort[0]  
        self.label_test = list(self.meta.event[test_id])

        if self.model == "LDA" or self.model == "SKLDA":
            self.X_train_current = np.asarray(self.X_train_current)
            self.X_test_current = np.asarray(self.X_test_current)
            n_trials, n_channels, n_times = self.X_train_current.shape
            self.X_train_current = self.X_train_current.reshape(n_trials, n_channels*n_times)
            n_trials, n_channels, n_times = self.X_test_current.shape
            self.X_test_current = self.X_test_current.reshape(n_trials, n_channels*n_times)
        elif self.model == "STDA":

            pos_idx = self.y_train_current == 2  
            neg_idx = self.y_train_current == 1  
            
            min_samples = min(np.sum(pos_idx), np.sum(neg_idx))
            
            pos_samples = self.X_train_current[pos_idx]
            neg_samples = self.X_train_current[neg_idx]

            # Add small random noise for regularization
            noise_level = 1e-5
            pos_samples = pos_samples + np.random.normal(0, noise_level, pos_samples.shape)
            neg_samples = neg_samples + np.random.normal(0, noise_level, neg_samples.shape)

            if len(pos_samples) >= min_samples:
                np.random.seed(38)  
                pos_indices = np.random.choice(len(pos_samples), min_samples, replace=False)
                pos_samples = pos_samples[pos_indices]
            
            if len(neg_samples) >= min_samples:
                np.random.seed(38)
                neg_indices = np.random.choice(len(neg_samples), min_samples, replace=False)
                neg_samples = neg_samples[neg_indices]
            
            # Recombine training data
            self.X_train_current = np.concatenate([pos_samples, neg_samples])
            self.y_train_current = np.array([2]*len(pos_samples) + [1]*len(neg_samples))
        
        print(self.label_test)

    def run(self, n_iter):

        self._prepare_iteration_data(self.current_k)

        self.clf.fit(X=self.X_train_current, y=self.y_train_current-1)
        self.current_k += 1

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            estimator=self.clf,
            solver_state=dict(
                current_k=self.current_k,
                X_test=self.X_test_current,
                label_test=self.label_test,
                k_loo=self.k_loo,
                model_name=self.model,
            )
        )

