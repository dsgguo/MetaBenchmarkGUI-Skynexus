from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import Benchmark

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metabci_utils import create_mi_paradigm, channel_selection
    from metabci.brainda.datasets import MunichMI
    # This is a mi dataset, we add the attach Yf template and the filterbank
    # to the dataset
    from metabci.brainda.algorithms.decomposition import (
        generate_cca_references
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "munichmi"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        # 'channel': ['occipital_9', 'all'],
        'duration': [i for i in np.arange(0, 7, 0.5)],
        'subject': [i for i in range(1, 11)]  # 1 to 10
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        # 该函数返回的字典会作为关键字参数传递给 Objective.set_data
        # 这定义了 benchmark 的数据传递接口，可以根据需求自定义

        # Constant parameters
        # visual_delay = 0.14
        srate = 500

        # Initialize the dataset
        dataset = MunichMI()
        # events = sorted(list(dataset.events.keys()))#取出所有事件的键,把这些键转成列表,并对这些字符串进行排序，得到有序的事件列表

        channels = dataset.channels
        # if self.channel == 'all':
        #     channels = dataset.channels
        # elif self.channel == 'occipital_9':
        #     channels = channel_selection('occipital_9')

        # create the paradigm
        # start_pnt = dataset.events[events[0]][1][0]#把第一个事件的起始时间（通常为0）赋值给 start_pnt，用于后续 paradigm 的时间窗口设置
        paradigm = create_mi_paradigm(
            channels=channels,
            events=['left_hand', 'right_hand'],
            interval=[(0, 7)], # 7 seconds
            srate=srate
        )


        # print("paradigm",paradigm)
        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[self.subject],
            return_concat=True,#返回拼接后的数据（通常是 trials 合并成一个大矩阵）
            n_jobs=None,#并行加载数据，加快处理速度 8
            verbose=False
        )
        # print("X nbytes:", X.nbytes)
        if dataset.dataset_code == "munichmi":
            special_data = dataset
        else:
            special_data = None

        return dict(
            X=X,                         # EEG特征数据，已带通滤波
            y=y,                         # 标签
            meta=meta,                   # 元信息
            srate=srate,                 # 采样率
            duration=self.duration,      # trial时长
            dataset_name=dataset.dataset_code,  # 数据集名称
            special_data=special_data,     # 额外传递的特殊数据（如 CCA 参考模板）
        )
