import importlib
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import NoCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from metabci.brainda.algorithms.decomposition.base import TimeDecodeTool
    from benchmark_utils.metabci_utils import gen_mi_filterbank
    from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_char_indices,
    match_char_kfold_indices
    )
    # import your reusable functions here
    import metabci.brainda.algorithms.decomposition as decomposition
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    @property
    def _stopping_criterion(self):
        """Use NoCriterion to run for fixed iterations."""
        return NoCriterion()

    # Name to select the solver in the CLI and to display the results.
    name = 'MI-docomposition-algo'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "model": [
            "CSP", "FBCSP", "MultiCSP", "FBMultiCSP",
            "DSP", "FBDSP",
            "SSCOR", "FBSSCOR",
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
        self.k_loo = 5 

        self.X_test_current = None
        self.X_train_current = None
        self.y_train_current = None
        self.y_test_current = None

        set_random_seeds(38)
        self.indices = generate_char_indices(self.meta, kfold=self.k_loo)

        try:

            if self.module_name == "decomposition":
                module = importlib.import_module("metabci.brainda.algorithms.decomposition")
            elif self.module_name == "algorithm":
                module = importlib.import_module("metabci.benchmarkGUI.benchmark.algorithm")  
            else:
                raise ValueError(f"模块 {self.module_name} 不支持，请选择 'decomposition' 或 'algorithm'。")

            model_to_load = self.custom_model or self.model

            if model_to_load[:2] == "FB":
                filterbank, filterweights = gen_mi_filterbank(srate=srate)

                if model_to_load == "FBCSP":
                    fb_model = getattr(module, model_to_load)(
                    n_components=5,
                    n_mutualinfo_components=4,
                    filterbank=filterbank,)
                    self.clf = make_pipeline(fb_model, SVC())  
                elif model_to_load == "FBMultiCSP":
                    fb_model = getattr(module, model_to_load)(
                    n_components=2, 
                    multiclass= 'ovo',
                    filterbank=filterbank,)
                    self.clf = make_pipeline(fb_model, SVC())
                else:
                    self.clf = getattr(module, model_to_load)(
                    filterbank=filterbank, filterweights=filterweights)

            else:
                if model_to_load == "CSP":
                    model = getattr(module, model_to_load)(
                    n_components=5,#
                    )
                    self.clf = make_pipeline(model, SVC())
                elif model_to_load == "MultiCSP":
                    model = getattr(module, model_to_load)(
                    n_components=2,
                    multiclass= 'ovo',)
                    self.clf = make_pipeline(model, SVC())
                else:
                    self.clf = getattr(module, model_to_load)()

        except (AttributeError, ImportError, ValueError) as e:
            raise ValueError(f"模型 {model_to_load} 在模块 {self.module_name} 中未找到，请确认是否已正确定义。") from e

    def _prepare_iteration_data(self, k):
        """准备第k次迭代的数据"""
        train_id, val_id, test_id = match_char_kfold_indices(k, self.meta, self.indices)
        train_ind = np.concatenate((train_id, val_id))
        self.test_ind = test_id

        self.X_train_current = self.X[train_ind]
        self.y_train_current = self.y[train_ind]

        self.X_test_current = self.X[self.test_ind]
        self.y_test_current = self.y[self.test_ind]

    def run(self, n_iter):

        self._prepare_iteration_data(self.current_k)

        self.clf.fit(X=self.X_train_current, y=self.y_train_current)
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
                X_train=self.X_train_current,
                y_train=self.y_train_current,
                X_test=self.X_test_current,
                y_test=self.y_test_current,
                k_loo=self.k_loo,
                model_name=self.model,
            )
        )
