from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from metabci.brainda.algorithms.decomposition.base import TimeDecodeTool
    from benchmark_utils.metabci_utils import gen_aVEP_filterbank
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

    # Name to select the solver in the CLI and to display the results.
    name = 'aVEP-docomposition-algo'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "model": [
            "DCPM",
        ],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []
    
    # 在 benchopt 框架中，sampling_strategy 用于控制每个 solver 的运行方式。
    # "run_once" 表示：对于每组输入数据和参数组合，这个 solver 只会被调用一次（即只训练/评估一次）。
    # 适用于确定性算法或不需要多次采样/重复的算法。不需要像 "callback" 或 "iteration" 那样多次采样、画收敛曲线。
    # sampling_strategy = "run_once"#本 solver（算法）在每组参数下只运行一次，不需要多次采样或多次重复实验
    sampling_strategy = "iteration"
    stopping_strategy = "iterations"

    def set_objective(self, X, y, srate, special_data, meta):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        # print("X shape:", self.X.shape)
        # print("y shape:", self.y.shape)
        self.special_data = special_data
        self.meta = meta

        self.current_k = 0
        self.k_loo = 3

        # 初始化数据存储属性，避免get_result时出现属性不存在的错误
        self.X_test_current = None
        self.X_train_current = None
        self.y_train_current = None
        self.label_test = None

        set_random_seeds(38)
        self.indices = generate_char_indices(self.meta, kfold=6)
        self.time_decode_tool = TimeDecodeTool(dataset=self.special_data)

        if self.model[:2] == "FB":
            filterbank, filterweights = gen_aVEP_filterbank(srate=srate)#生成滤波器组及其权重，供 filter bank 类算法使用。
            # print(filterbank, filterweights)
            # getattr(decomposition, self.model)：等价于 decomposition.FBTRCA，即根据字符串动态获取对应的类
            self.clf = getattr(decomposition, self.model)(#：动态获取 decomposition 模块下的算法类（如 TRCA、FBTRCA、ECCA）
                filterbank=filterbank, filterweights=filterweights)
            # 这种写法适合支持多种算法统一接口，只需更改参数即可切换不同解码器，无需写多余的 if-else。
        else:
            if self.model == "LDA":
                self.clf = LinearDiscriminantAnalysis()
            elif self.model == "DCPM":
                self.clf = getattr(decomposition, self.model)(n_components=8,)
            elif self.model == "SKLDA":
                self.clf = getattr(decomposition, self.model)()
            else: # STDA
                self.clf = getattr(decomposition, self.model)()

    def _prepare_iteration_data(self, k):
        """准备第k次迭代的数据"""
        train_id, val_id, test_id = match_char_kfold_indices(k, self.meta, self.indices)
        train_ind = np.concatenate((train_id, val_id))
        self.test_ind = test_id
        # print("train_ind", train_ind)
        # print("test_id", test_id)
        
        # 准备训练数据
        X_train_t = [self.X[i] for i in train_ind]
        y_train_t = [self.y[i] for i in train_ind]
        Key = self.meta.event[train_ind]
        
        y_train_tar = self.time_decode_tool.target_calibrate(y_train_t, Key)
        self.label_test = list(self.meta.event[test_id])
        
        # 准备当前迭代的训练数据
        self.X_train_current = np.concatenate(X_train_t)
        self.y_train_current = np.concatenate(y_train_tar)
        
        # 准备测试数据并进行epoch_sort处理
        X_test_t = [self.X[i] for i in test_id]
        y_test_t = [self.y[i] for i in test_id]
        X_test_sort, y_test_sort = self.time_decode_tool.epoch_sort(X_test_t, y_test_t)
        
        self.X_test_current = X_test_sort[0]  # 取第一组排序后的数据
        self.y_test_current = y_test_sort[0]  # 取第一组排序后的标签
        self.label_test = list(self.meta.event[test_id])
        # print("X_test_sort", X_test_sort[0])
        if self.model == "LDA" or self.model == "SKLDA":
            self.X_train_current = np.asarray(self.X_train_current)
            self.X_test_current = np.asarray(self.X_test_current)
            n_trials, n_channels, n_times = self.X_train_current.shape
            self.X_train_current = self.X_train_current.reshape(n_trials, n_channels*n_times)
            n_trials, n_channels, n_times = self.X_test_current.shape
            self.X_test_current = self.X_test_current.reshape(n_trials, n_channels*n_times)
        elif self.model == "STDA":
            # STDA需要保持3D数据，但需要确保两个类别的样本数相同
            # print(self.y_train_current)
            pos_idx = self.y_train_current == 2  # 目标刺激
            neg_idx = self.y_train_current == 1  # 非目标刺激
            
            # 获取较小类别的样本数
            min_samples = min(np.sum(pos_idx), np.sum(neg_idx))
            
            # 随机采样使两个类别样本数相同
            pos_samples = self.X_train_current[pos_idx]
            neg_samples = self.X_train_current[neg_idx]
            # print("pos_samples", pos_samples.shape)
            # print("neg_samples", neg_samples.shape)
            # 添加微小的随机噪声进行正则化
            noise_level = 1e-5
            pos_samples = pos_samples + np.random.normal(0, noise_level, pos_samples.shape)
            neg_samples = neg_samples + np.random.normal(0, noise_level, neg_samples.shape)

            if len(pos_samples) >= min_samples:
                # print("len(pos_samples)", len(pos_samples))
                np.random.seed(38)  # 保持随机性一致
                pos_indices = np.random.choice(len(pos_samples), min_samples, replace=False)
                pos_samples = pos_samples[pos_indices]
            
            if len(neg_samples) >= min_samples:
                # print("len(neg_samples)", len(neg_samples))
                np.random.seed(38)
                neg_indices = np.random.choice(len(neg_samples), min_samples, replace=False)
                neg_samples = neg_samples[neg_indices]
            
            # 重新组合训练数据
            self.X_train_current = np.concatenate([pos_samples, neg_samples])
            self.y_train_current = np.array([2]*len(pos_samples) + [1]*len(neg_samples))
        
        print(self.label_test)

    def run(self, n_iter):
        # 每次run都处理一次迭代
        if self.current_k < self.k_loo:
            self._prepare_iteration_data(self.current_k)
            # 大多数分类器（包括 DCPM 和 LDA）期望标签从 0 开始，需要将标签映射为 0 和 1，0 表示非目标刺激，1 表示目标刺激
            # self.X_train_current = np.asarray(self.X_train_current)

            # print("X_train_current", self.X_train_current.shape)#(660, 16, 60)
            # print("y_train_current", self.y_train_current)
            self.clf.fit(X=self.X_train_current, y=self.y_train_current-1)
            self.current_k += 1
            return True  # 继续迭代
        return False    # 停止迭代

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        # 这段代码的核心作用是把训练好的模型对象打包返回，方便后续自动评测
        return dict(
            estimator=self.clf,
            solver_state=dict(
                current_k=self.current_k,
                X_test=self.X_test_current,
                y_test=self.y_test_current,
                label_test=self.label_test,
                k_loo=self.k_loo,
                model_name=self.model,
            )
        )
        #dict(model=self.clf)：构造一个字典，键名为 'model'，值为训练好的模型对象。
