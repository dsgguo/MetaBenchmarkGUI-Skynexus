import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def filter(data:np.ndarray, srate:float, filter_type='bandpass', 
                  freq_range=None, order=4, notch_freq=50, notch_q=30):
    """
    对信号进行滤波处理
    
    参数:
    data: np.ndarray - 输入信号数据，形状为(trials, channels, samples)或(channels, samples)
    srate: float - 采样率(Hz)
    filter_type: str - 滤波类型，可选 'bandpass', 'lowpass', 'highpass', 'notch', 'butter'
    freq_range: tuple or float - 滤波频率范围，对于带通滤波为(low, high)，低通或高通滤波为截止频率
    order: int - 滤波器阶数
    notch_freq: float - 陷波滤波器中心频率(Hz)，通常用于去除工频干扰
    notch_q: float - 陷波滤波器品质因数，控制陷波宽度
    
    返回:
    np.ndarray - 滤波后的数据，与输入数据形状相同
    """
    # 保存原始数据的形状
    original_shape = data.shape
    need_reshape = len(original_shape) == 3
    
    # 如果数据是3D的(trials, channels, samples)，重塑为2D(channels, samples)方便处理
    if need_reshape:
        data = np.reshape(data, (original_shape[0] * original_shape[1], original_shape[2]))
    
    # 根据滤波类型应用不同的滤波器
    if filter_type == 'bandpass':
        if freq_range is None:
            freq_range = (8, 30)  # 默认频段
        nyquist = srate / 2
        low, high = freq_range
        low, high = low / nyquist, high / nyquist
        sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
    
    elif filter_type == 'lowpass':
        if freq_range is None:
            freq_range = 40  # 默认截止频率
        nyquist = srate / 2
        cutoff = freq_range / nyquist
        sos = signal.butter(order, cutoff, btype='lowpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
    
    elif filter_type == 'highpass':
        if freq_range is None:
            freq_range = 1  # 默认截止频率
        nyquist = srate / 2
        cutoff = freq_range / nyquist
        sos = signal.butter(order, cutoff, btype='highpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
    
    elif filter_type == 'notch':
        nyquist = srate / 2
        freq = notch_freq / nyquist
        b, a = signal.iirnotch(freq, notch_q)
        filtered_data = signal.filtfilt(b, a, data, axis=1)
    
    elif filter_type == 'butter':
        # 通用巴特沃斯滤波器
        if freq_range is None:
            freq_range = (8, 30)  # 默认频段
        nyquist = srate / 2
        if isinstance(freq_range, tuple):
            low, high = freq_range
            low, high = low / nyquist, high / nyquist
            sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
        else:
            cutoff = freq_range / nyquist
            sos = signal.butter(order, cutoff, btype='lowpass', output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
    
    else:
        raise ValueError(f"不支持的滤波类型: {filter_type}")
    
    # 将数据重塑回原始形状
    if need_reshape:
        filtered_data = np.reshape(filtered_data, original_shape)
    
    return filtered_data


def fft(data:np.ndarray, srate:float, plot:bool=False, channel_idx=None, trial_idx=None,
        plot_mode='single', avg_channels=False, max_items=10):
    """
    对信号执行FFT分析
    
    参数:
    data: np.ndarray - 输入信号数据，形状为(trials, channels, samples)或(channels, samples)
    srate: float - 采样率(Hz)
    plot: bool - 是否绘制频谱图
    channel_idx: int  - 需要绘制的通道索引，如果为None且plot=True，则绘制所有通道
    trial_idx: int - 如果数据是3D的，指定要处理的试次,如果不指定则叠加平均
    plot_mode: str - 绘图模式，可选 'single', 'subplot'
    avg_channels: bool - 是否对选定通道进行平均
    max_items: int - 如果通道数量超过此值，则在图例中只显示前后各一个通道
    返回:
    freqs: np.ndarray - 频率数组
    power: np.ndarray - 功率谱密度，形状为(n_channels, n_freqs)或(n_freqs,)
    """
    # 确定数据维度
    if len(data.shape) == 3:  # (trials, channels, samples)
        if trial_idx is None:
            # If no trial specified, average all trials
            data_to_process = np.mean(data, axis=0)
        else:
            data_to_process = data[trial_idx]
    else:  # (channels, samples)
        data_to_process = data

    n_channels, n_samples = data_to_process.shape

    # Process channel selection
    if channel_idx is None:
        channels = list(range(n_channels))
    elif isinstance(channel_idx, int):
        channels = [channel_idx]
    else:
        channels = channel_idx  # Assume it's a list

    # 计算FFT
    fft_result = np.fft.rfft(data_to_process, axis=1)
    # 计算功率谱密度
    power = np.abs(fft_result) ** 2 / n_samples
    # 频率数组
    freqs = np.fft.rfftfreq(n_samples, 1/srate)

    # Plot spectrum
    if plot:
        if avg_channels and len(channels) > 1:
            # Average selected channels
            avg_power = np.mean(power[channels], axis=0)

            plt.figure(figsize=(12, 6))
            plt.plot(freqs, avg_power)
            plt.title(f"Average Spectrum (Channels: {channels[0]}..{channels[-1]})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.xlim(0, min(srate / 2, 50))  # Display up to 50Hz or Nyquist frequency
            plt.grid(True)
            plt.show()

        elif plot_mode == "single":
            # Plot all selected channels on a single figure
            plt.figure(figsize=(12, 6))

            # Handle large number of channels in legend
            show_legend = True
            if len(channels) > max_items:
                plt.plot(freqs, power[channels[0]], label=f"Ch {channels[0]}")
                plt.plot(freqs, power[channels[-1]], label=f"Ch {channels[-1]}")

                for ch in channels[1:-1]:
                    plt.plot(freqs, power[ch], color="grey", alpha=0.5)

                plt.title(f"Channel Spectra (Showing {len(channels)} channels)")
                show_legend = False
            else:
                for ch in channels:
                    plt.plot(freqs, power[ch], label=f"Ch {ch}")
                plt.title("Channel Spectra")

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.xlim(0, min(srate / 2, 50))  # Display up to 50Hz or Nyquist frequency
            plt.grid(True)

            if show_legend:
                plt.legend()
            plt.show()

        elif plot_mode == "subplot":
            # Plot each channel in a separate subplot
            n_plots = len(channels)
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            fig.suptitle("Channel Spectra")

            # Flatten axes if it's a 2D array
            if n_plots > 1:
                axes = axes.flatten()
            else:
                axes = [axes]

            for i, ch in enumerate(channels):
                if i < len(axes):
                    axes[i].plot(freqs, power[ch])
                    axes[i].set_title(f"Channel {ch}")
                    axes[i].set_xlabel("Frequency (Hz)")
                    axes[i].set_ylabel("Power Spectral Density")
                    axes[i].set_xlim(0, min(srate / 2, 50))
                    axes[i].grid(True)

            # Hide unused subplots
            for i in range(len(channels), len(axes)):
                axes[i].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
            plt.show()

    # Return only the selected channels' power spectrum if specific channels were requested
    if channel_idx is not None:
        return freqs, power[channels]
    return freqs, power
