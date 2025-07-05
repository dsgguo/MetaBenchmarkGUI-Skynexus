from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from metabci.brainda.algorithms.utils.model_selection import (
        EnhancedStratifiedKFold,)
    from metabci.brainda.algorithms.decomposition.base import (
    TimeDecodeTool
    )
    from metabci.brainda.utils.performance import Performance


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "MetaBCI Benchmark"

    # URL of the main repo for this benchmark.
    url = "https://github.com/ch-MEIJIE/MetaBCI-Benchmark"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'random_state': [0],
        'intra_subject': [False],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = ["numpy"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(self, X, y, meta, srate, duration, dataset_name, special_data):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y
        self.meta = meta
        self.srate = srate
        self.duration = duration
        self.dataset_name = dataset_name
        self.special_data = special_data


        # Obtain the sample size for each category
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples = np.min(class_counts)
        
        # Adjust the fold based on the minimum category sample size
        n_splits = min(5, min_samples)

        # Initialize a 50% split layered cross validator
        self.cv = EnhancedStratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            return_validate=False, 
            random_state=self.random_state, 
        )

        if n_splits < 5:
            print(f"Warning: Reduced n_splits to {n_splits} due to limited samples")
            print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")

    def evaluate_result(self, estimator, solver_state=None):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.
        if self.dataset_name == "Cattan_P300":
            self.time_decode_tool = TimeDecodeTool(dataset=self.special_data)

            current_k = solver_state['current_k']
            X_test = solver_state['X_test']
            label_test = solver_state['label_test']
            k_loo = solver_state['k_loo']
            model_name = solver_state['model_name']

            if X_test.ndim == 4:
                X_test = X_test.reshape(-1, *X_test.shape[2:])

            test_feature = estimator.transform(X_test)

            #Based on the label, features, and paradigm of the current character, infer the final recognized character
            decode_key = self.time_decode_tool.decode(
                label_test[0],
                test_feature,
                fold_num=5,
                paradigm=self.special_data.paradigm)

            right_count = 1 if decode_key == label_test[0] else 0 
            current_acc = right_count / len(label_test)
            self.all_results.append(current_acc)

            mean_acc = np.mean(self.all_results) 

            # If all iterations are completed, return the average accuracy
            print(f"Current k: {current_k}, k_loo: {k_loo}")
            if current_k >= k_loo:
                final_accuracy = np.mean(self.all_results)
                return dict(
                    value=current_acc,
                    single_acc=current_acc,
                    mean_acc=mean_acc,
                    final_acc=final_accuracy,
                )
            else:
                return dict(value=current_acc, single_acc=current_acc, mean_acc=mean_acc,)
        
        elif self.dataset_name in {"alexeeg", "zhou2016", "bnci2014001", "cbcic2019001", "cbcic2019004", "cho2017", "munichmi", "eegbci", "schirrmeister2017", "weibo2014"}:
            current_k = solver_state['current_k']
            X_test = solver_state['X_test']
            y_test = solver_state['y_test']
            k_loo = solver_state['k_loo']
            model_name = solver_state['model_name']

            if X_test.ndim == 4:
                X_test = X_test.reshape(-1, *X_test.shape[2:])

            if model_name == "SSCOR" or model_name == "FBSSCOR":
                X_train = solver_state['X_train']
                y_train = solver_state['y_train']
                train_feature = estimator.transform(X_train)
                test_feature = estimator.transform(X_test)

                from sklearn.svm import SVC
                classifier = SVC()
                classifier.fit(train_feature, y_train)

                y_pred = classifier.predict(test_feature)
                performance = Performance(estimators_list=["Acc", "tITR"], Tw=self.duration)
                result = performance.evaluate(
                    y_true=y_test, y_pred=y_pred)
                
                accuracy = result["Acc"]
                itr = result["tITR"]
                self.all_results.append(accuracy)
                mean_acc = np.mean(self.all_results)
            else:
                y_pred = estimator.predict(X_test)
                performance = Performance(estimators_list=["Acc", "tITR"], Tw=self.duration)
                result = performance.evaluate(
                    y_true=y_test, y_pred=y_pred)

                accuracy = result["Acc"]
                itr = result["tITR"]
                self.all_results.append(accuracy)
                mean_acc = np.mean(self.all_results)

            if current_k >= k_loo:
                final_accuracy = np.mean(self.all_results)
                return dict(
                    value=accuracy,
                    single_acc=accuracy,
                    mean_acc=mean_acc,
                    final_acc=final_accuracy,
                    ITR=itr)
            return dict(
                value=accuracy,
                single_acc=accuracy,
                mean_acc=mean_acc,
                ITR=itr)
        
        elif self.dataset_name == "Xu_aVEP_min_aVEP":
            self.time_decode_tool = TimeDecodeTool(dataset=self.special_data)

            current_k = solver_state['current_k']
            X_test = solver_state['X_test']
            y_test = solver_state['y_test']
            label_test = solver_state['label_test']
            k_loo = solver_state['k_loo']
            model_name = solver_state['model_name']

            if X_test.ndim == 4:
                X_test = X_test.reshape(-1, *X_test.shape[2:])

            if model_name == "DCPM":
                epoch_accs = []
                p_labels = estimator.predict(X_test)
                epoch_accs.append(np.mean(p_labels == y_test-1))

            test_feature = estimator.transform(X_test)

            decode_key = self.time_decode_tool.decode(
                label_test[0],
                test_feature,
                fold_num=6,
                paradigm=self.special_data.paradigm)

            right_count = 1 if decode_key == label_test[0] else 0 
            current_acc = right_count / len(label_test)
            self.all_results.append(current_acc)

            mean_acc = np.mean(self.all_results) 

            if current_k >= k_loo:
                final_accuracy = np.mean(self.all_results)
                return dict(
                    value=final_accuracy,
                    mean_acc=final_accuracy
                )

            return dict(value=current_acc, epoch_accs=epoch_accs, single_acc=current_acc, mean_acc=mean_acc,)
        
        else:
            y_pred = estimator.predict(self.X_test)
            
            # Create a performance evaluator object to calculate classification accuracy (Acc) 
            # and information transfer rate (tITR), and specify window duration Tw.
            performance = Performance(estimators_list=["Acc", "tITR"], Tw=self.duration)

            result = performance.evaluate(
                y_true=self.y_test, y_pred=y_pred
            )

            accuracy = result["Acc"]
            itr = result["tITR"]

            # This method can return many metrics in a dictionary. One of these
            # metrics needs to be `value` for convergence detection purposes.
            return dict(
                value=accuracy,
                ACC=accuracy,
                ITR=itr
            )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        clf = DummyClassifier()
        clf.fit(self.X_train, self.y_train)
        return dict(model=clf)


    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
    
        if self.dataset_name == "Cattan_P300":
            # Initialization result storage
            self.all_results = []
            self.X_train = self.X
            self.y_train = self.y

            return dict(
            X=self.X_train,
            y=self.y_train,
            srate=self.srate,
            special_data=self.special_data,
            meta = self.meta,
            )

        elif self.dataset_name in {"alexeeg", "zhou2016", "bnci2014001", "cbcic2019001", "cbcic2019004", "cho2017", "munichmi", "eegbci", "schirrmeister2017", "weibo2014"}:

            self.all_results = []
            self.X_train = self.X
            self.y_train = self.y

            return dict(
            X=self.X_train,
            y=self.y_train,
            srate=self.srate,
            special_data=self.special_data,
            meta = self.meta,
            )

        elif self.dataset_name == "Xu_aVEP_min_aVEP":

            self.all_results = []
            self.X_train = self.X
            self.y_train = self.y

            return dict(
            X=self.X_train,
            y=self.y_train,
            srate=self.srate,
            special_data=self.special_data,
            meta = self.meta,
            )
        
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = self.get_split(
                self.X, self.y
            )
            return dict(
            X=self.X_train,
            y=self.y_train,
            srate=self.srate,
            special_data=self.special_data,
        )

        
