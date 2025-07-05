import importlib
from benchopt import BaseSolver, safe_import_context

# 应用CCA修复补丁
try:
    from ..cca_patch import apply_cca_fix
    apply_cca_fix()
except ImportError:
    print("Warning: CCA patch not found, using original implementation")

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.metabci_utils import gen_ssvep_filterbank
    # import your reusable functions here


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'SSVEP-docomposition-algo'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "model": [
            "ECCA","FBECCA",
            "SCCA","FBSCCA",
            "ItCCA","FBItCCA",
            "TtCCA","FBTtCCA",
            "MsetCCA","FBMsetCCA",
            "MsetCCAR","FBMsetCCAR"
            "TRCA","FBTRCA",
            "TDCA", "FBTDCA"
            "TRCAR","FBTRCAR",
            None
        ],
        "padding_len": [None,0, 1, 2, 3, 4, 5],
        "custom_model": [None],  
        "module_name": ["decomposition", "algorithm"]  

    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    sampling_strategy = "run_once"

    def set_objective(self, X, y, srate, special_data=None):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        self.special_data = special_data

        try:
            # Dynamically import modules based on module_name
            if self.module_name == "decomposition":
                module = importlib.import_module("metabci.brainda.algorithms.decomposition")
            elif self.module_name == "algorithm":
                module = importlib.import_module("metabci.benchmarkGUI.benchmark.algorithm")
            else:
                raise ValueError(f"模块 {self.module_name} 不支持，请选择 'decomposition' 或 'algorithm'。")

            model_to_load = self.custom_model or self.model

            if model_to_load[:2] == "FB":
                filterbank, filterweights = gen_ssvep_filterbank(srate=srate)
                if model_to_load == "FBTDCA":
                    self.clf = getattr(module, model_to_load)(
                    filterbank=filterbank, padding_len=self.padding_len, filterweights=filterweights)
                else:
                    self.clf = getattr(module, model_to_load)(
                        filterbank=filterbank, filterweights=filterweights)
            else:
                if model_to_load == "TDCA":
                    self.clf = getattr(module, model_to_load)(padding_len=self.padding_len)
                else:
                    self.clf = getattr(module, model_to_load)()
        except (AttributeError, ImportError, ValueError) as e:
            raise ValueError(f"模型 {model_to_load} 在模块 {self.module_name} 中未找到，请确认是否已正确定义。") from e

    def run(self, iteration):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        self.clf.fit(X=self.X, y=self.y, Yf=self.special_data['Yf'])

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(estimator=self.clf)
