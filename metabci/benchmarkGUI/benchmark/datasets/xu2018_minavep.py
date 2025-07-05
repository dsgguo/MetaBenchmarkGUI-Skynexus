from benchopt import BaseDataset, safe_import_context
from benchopt.benchmark import Benchmark

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.metabci_utils import (
        create_avep_paradigm, p300_channel_selection, 
        p300_raw_hook, p300_trial_hook, 
        p300_epochs_hook, p300_data_hook,)

    from metabci.brainda.datasets import Xu2018MinaVep



# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Xu_aVEP_min_aVEP"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        # 'channel': ['occipital_8', 'all'],
        # 'duration': [i for i in np.arange(0.3, 0.7, 0.1)],  # P300常用窗口0.2~0.6s
        'duration': None,
        'subject': [i for i in range(1, 13)] 
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
        srate = 1000

        # Initialize the dataset
        dataset = Xu2018MinaVep()
        # events = sorted(list(dataset.events.keys()))#取出所有事件的键,把这些键转成列表,并对这些字符串进行排序，得到有序的事件列表

        # if self.channel == 'all':
        channels = dataset.channels
        # elif self.channel == 'occipital_8':
        #     channels = p300_channel_selection('occipital_8')

        # create the paradigm
        # start_pnt = dataset.events[events[0]][1][0]#把第一个事件的起始时间（通常为0）赋值给 start_pnt，用于后续 paradigm 的时间窗口设置
        paradigm = create_avep_paradigm(
            channels,
        )


        # paradigm.register_raw_hook(p300_raw_hook)
        # paradigm.register_trial_hook(p300_trial_hook)
        # paradigm.register_epochs_hook(p300_epochs_hook)
        # paradigm.register_data_hook(p300_data_hook)

        # print(self.subject)
        # print("paradigm",paradigm)
        X, y, meta = paradigm.get_data(
            dataset=dataset,
            subjects=[self.subject],
            return_concat=True,#返回拼接后的数据（通常是 trials 合并成一个大矩阵）
            n_jobs=8,#并行加载数据，加快处理速度
            verbose=False
        )
        
        # print(X, y, meta)
        # print(X.shape, y.shape, meta.shape)

        # freqs = [dataset.get_freq(event) for event in events]
        # phases = [dataset.get_phase(event) for event in events]

        # Yf = generate_cca_references(
        #     freqs,
        #     srate,
        #     self.duration,
        #     phases=phases,
        #     n_harmonics=5)
        if dataset.dataset_code == "Xu_aVEP_min_aVEP":
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
