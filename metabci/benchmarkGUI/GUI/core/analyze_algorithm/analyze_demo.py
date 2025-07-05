import numpy as np
from scipy.signal import sosfiltfilt
from metabci.brainda.datasets import Wang2016, Nakanishi2015
from metabci.brainda.paradigms import SSVEP
from data_fft import filter, fft
import matplotlib.pyplot as plt

# dataset = Wang2016()
dataset = Nakanishi2015()
events:list[str] = sorted(list(dataset.events.keys()))
freqs:list[int] = [dataset.get_freq(event) for event in events]
phases:list[int] = [dataset.get_phase(event) for event in events]
channels:list[str] = dataset._CHANNELS

delay = 0.14  # seconds
select_channels = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
srate = 256  # Hz
duration = 5  # seconds

paradigm = SSVEP(
    srate=srate,
    channels=channels,
    intervals=[
        (delay, delay + duration)
    ],  # more seconds for TDCA
    events=events,
)

'''
X:dict[str, np.ndarray] = {
    "10": np.arry(blocks, channels, datapoints),
    ...
    }
'''
# X, _, meta = paradigm.get_data(
#     dataset, subjects=[1], return_concat=False, n_jobs=1, verbose=False
# )


# Tenhz:np.ndarray = X["10"]
# filtered_data = filter(
#     Tenhz,
#     srate=srate,
#     filter_type="bandpass",
#     freq_range=(1, 30),
#     order=4,
# )

channel_idx = [idx for idx, _ in enumerate(channels)]
print(f"Channels: {channels}")
print(f"Channels idx: {channel_idx}")

# # ex1 多通道合并在一张图上
# freqs, power = fft(
#     filtered_data,
#     srate=srate,
#     plot=True,
#     channel_idx=[0,1,2,3],
#     plot_mode="single"
# )

# freqs, power = fft(
#     filtered_data,
#     srate=srate,
#     plot=True,
#     channel_idx=[0,1,2,3,4,5],
#     plot_mode="subplot"
# )

# freqs, power = fft(
#     filtered_data,
#     srate=srate,
#     plot=True,
#     channel_idx=[0,1,2,3],
#     avg_channels=True
# )

# # 示例4: 当有大量通道时，自动简化图例
# freqs, power = fft(
#     filtered_data,
#     srate=250,
#     plot=True,
#     channel_idx=list(range(20)),  # 选择20个通道
#     plot_mode="single",
#     max_items=8,  # 超过8个通道时简化图例显示
# )



