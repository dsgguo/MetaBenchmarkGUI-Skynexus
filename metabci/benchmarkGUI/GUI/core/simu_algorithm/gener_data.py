import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from mne.filter import resample
from metabci.brainda.datasets import Wang2016
from metabci.brainda.paradigms import SSVEP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.decomposition import FBTRCA
from metabci.brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices,
    match_loo_indices,
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SSVEP数据生成和模型训练')
    parser.add_argument('--sub_id', type=int, default=1, help='受试者ID，范围1-10')
    parser.add_argument('--output_dir', type=str, default='./ssvep_train_data', 
                      help='输出数据的基础目录路径')
    return parser.parse_args()

def select_trials_from_sequence(label_sequence, label_data_dict):
    """
    根据标签序列从label_data_dict中选择对应的试次

    参数:
    label_sequence: 标签序列，如[8, 23, 16, ...]
    label_data_dict: 包含每个标签对应试次数据的字典

    返回:
    selected_trials: 选择的试次数据列表
    """
    # 为每个标签维护一个已使用的试次索引集合
    used_trials = {label: set() for label in label_data_dict.keys()}

    # 存储选择的试次
    selected_trials = []

    # 记录无法满足的请求
    unsatisfied_requests = []

    print(f"开始选择试次，标签序列长度: {len(label_sequence)}")

    for i, label in enumerate(label_sequence):
        if label not in label_data_dict:
            print(f"警告: 标签 {label} 在 label_data_dict 中不存在")
            unsatisfied_requests.append((i, label))
            continue

        # 获取该标签的所有试次数据
        label_trials = label_data_dict[label]
        num_trials = label_trials.shape[0]

        # 找到一个未使用的试次索引
        available_indices = [
            j for j in range(num_trials) if j not in used_trials[label]
        ]

        if not available_indices:
            print(f"警告: 标签 {label} 的所有试次已用完，位置 {i}")
            unsatisfied_requests.append((i, label))
            # 如果所有试次都用完了，重置已使用集合
            used_trials[label] = set()
            available_indices = list(range(num_trials))

        # 选择未使用的试次
        trial_idx = available_indices[0]
        used_trials[label].add(trial_idx)

        # 添加选中的试次数据
        selected_trials.append(label_trials[trial_idx])

    print(f"完成选择，总共选择了 {len(selected_trials)} 个试次")
    if unsatisfied_requests:
        print(f"有 {len(unsatisfied_requests)} 个请求无法满足，已重新循环使用试次")

    # 将选择的试次堆叠成一个数组
    if selected_trials:
        selected_trials_array = np.stack(selected_trials)
        return selected_trials_array
    else:
        return np.array([])


def segmentX(f:float, phi_num:float, data:np.array, time:float=None):
    # 已知的信号参数
    srate = 250  # 采样率
    phi = phi_num * np.pi  # 初相位

    # 定义每个数据段的长度
    segment_length = int(time * srate)

    # 计算分割数
    num_segments = 10

    # 初始化一个用于存储所有分割结果的数组
    all_segments = np.empty((num_segments, data.shape[0], segment_length))

    periods = int(srate / f)

    # 遍历 data 的第一维度
    for j in range(data.shape[0]):
        # 生成信号
        signal = data[j]

        # 计算初相位对应的数据点位置
        start_point = 0

        # 截取信号
        for i in range(num_segments):
            if i == 0:
                # 截取平移后的信号
                segment = signal[start_point : start_point + segment_length]
                # 将截取的信号添加到结果数组中
                all_segments[i, j, :] = segment
                # 更新开始的索引
                start_point += segment_length
                continue
            # 计算第125个数据点的相位
            phi_n = ((segment_length * i) * 2 * np.pi * f / srate) + phi

            # 使用给定的公式计算出平移的距离
            # 移动到0相位的距离
            move_to_phi_0 = int((2 * np.pi - phi_n) * srate / (2 * np.pi * f))
            # 移动到初相位的距离
            move_to_phi_c = int((phi) * srate / (2 * np.pi * f))

            seg_point = start_point + move_to_phi_0 + move_to_phi_c

            # 截取平移后的信号
            segment = signal[seg_point : seg_point + segment_length]

            if len(segment) < segment_length:
                if segment_length - len(segment) < periods:
                    seg_point = start_point + move_to_phi_0 + move_to_phi_c - periods
                    segment = signal[seg_point : seg_point + segment_length]
                else:
                    seg_point = (
                        start_point + move_to_phi_0 + move_to_phi_c - periods * 2
                    )
                    segment = signal[seg_point : seg_point + segment_length]
            # 将截取的信号添加到结果数组中
            all_segments[i, j, :] = segment

            # 更新开始的索引
            start_point += segment_length

    return all_segments

def train_model(X, y, srate=250):
    y = np.reshape(y, (-1))
    X = resample(X, up=250, down=srate)

    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]
    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)

    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)

    model = FBTRCA(
        filterbank, n_components=1, filterweights=np.array(filterweights)
    )
    model = model.fit(X, y)
    return model


dataset = Wang2016()

srate = 250  # Hz
delay = 0.14  # seconds
duration = 5  # seconds
channels = ["PZ", "PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]

Command = {
        "0":["A",   8,    0.00],
        "1":["I",   8.2,    0.50],
        "2":["Q",   8.4,    1.00],
        "3":["Y",   8.6,    1.50],
        "4":["B",   9,    0.50],
        "5":["J",   9.2,    1.00],
        "6":["R",   9.4,    1.50],
        "7":["Z",   9.6,    0.00],
        "8":["C",   10,   1.00],
        "9":["K",   10.2,   1.50],
        "10":["S",  10.4,   0.00],
        "11":[" ",  12.8,   0.00,],
        "12":["D",  11,   1.50],
        "13":["L",  11.2,   0.00],
        "14":["T",  11.4,   0.50],
        "15":["E",  12,   0.00],
        "16":["M",  12.2,   0.50],
        "17":["U",  12.4,   1.00],
        "18":["F",  13,   0.50],
        "19":["N",  13.2,   1.00],
        "20":["V",  13.4,   1.50],
        "21":[",",  13.8,   0.50,],
        "22":["G",  14,   1.00],
        "23":["O",  14.2,   1.50],
        "24":["W",  14.4,   0.00],
        "25":[".",  14.8,   1.00],
        "26":["H",  15,   1.50],
        "27":["P",  15.2,   0.00],
        "28":["X",  15.4,   0.50],
        "29":["Back", 15.8,   1.50]
    }

events = list(dataset.events.keys())
selected_keys = [str(Command[freq][1]) for freq in Command.keys()]
events = [event for event in events if event in selected_keys]

freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]
freq_to_command_key = {}
for key, value in Command.items():
    freq = value[1]  # 取第二个元素，即频率
    freq_to_command_key[str(freq)] = int(key)

paradigm = SSVEP(
    srate=srate,
    channels=channels,
    intervals=[
        (delay, delay + duration )
    ],  # more seconds for TDCA
    events=events
)

if __name__ == "__main__":
    args = parse_args()
    sub_id = args.sub_id
    X, y, meta = paradigm.get_data(
        dataset, subjects=[sub_id], return_concat=True, n_jobs=1, verbose=False
    )

    y_new = []
    for event in events:
        event_freq = str(dataset.get_freq(event))
        if event_freq in freq_to_command_key:
            y_new.append(freq_to_command_key[event_freq])
    if len(y) == X.shape[0]:
        # 如果原始y长度正确，我们需要扩展y_new以匹配
        # 假设每个频率在X中有相同数量的样本
        samples_per_freq = len(y) // len(events)
        y = np.array([])
        for cmd_key in y_new:
            y = np.append(y, [cmd_key] * samples_per_freq)

    set_random_seeds(64)
    loo_indices = generate_loo_indices(meta)
    filterX, filterY = np.copy(X[..., : int(srate * duration)]), np.copy(y)
    filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
    n_loo = len(loo_indices[sub_id][events[0]])
    loo_accs = []
    output_base_dir = Path(args.output_dir)
    # save_dir = Path(f"./ssvep_train_data/sub{sub_id}/")
    save_dir = output_base_dir / f"sub{sub_id}/"
    save_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_loo):  
        train_ind, validate_ind, test_ind = match_loo_indices(
            k, meta, loo_indices)
        test_ind = np.concatenate([test_ind, validate_ind])
        trainX, trainY = filterX[train_ind], filterY[train_ind]
        MetaX, MetaY = filterX[test_ind], filterY[test_ind]

        test_data = []
        for i in range(MetaX.shape[0]):
            # 取出对应频率的数据
            data = MetaX[i]
            # 取出对应的频率
            if i < 30:
                frequency = freqs[i]
                phase_num = phases[i]
            else:
                frequency = freqs[i - 30]
                phase_num = phases[i - 30]
            # 将数据、频率和相位作为参数传递给你的函数
            a = segmentX(frequency, phase_num, data, time=1.5)
            # 将 a 添加到列表中
            test_data.append(a)  
        testX = np.concatenate(test_data, axis=0)
        testY = np.repeat(MetaY, 10)
        # 计算每个频率有多少个试次（应该是10个）
        trials_per_freq = 10

        # 创建一个字典，用于存储按标签组织的数据
        label_data_dict = {}

        # 获取testY中的唯一标签
        unique_labels = np.unique(testY)

        # 遍历每个唯一标签
        for label in unique_labels:
            # 找到当前标签在testY中的索引
            label_indices = np.where(testY == label)[0]

            if len(label_indices) == 0:
                print(f"标签 {label} 没有对应的数据")
                continue

            # 收集对应索引的数据
            label_data = testX[label_indices]

            # 检查收集到的数据数量
            expected_trials = (
                trials_per_freq * 2
            )  # 每个频率应该有20个试次（假设每个标签对应一个频率，有两个测试块）
            if len(label_indices) != expected_trials:
                print(
                    f"警告: 标签 {label} 有 {len(label_indices)} 个试次，期望 {expected_trials} 个"
                )

            # 重塑数组为[20,9,375]形状（如果不是20个试次，则使用实际数量）
            label_data_shaped = label_data.reshape(-1, 9, 375)

            # 添加到字典
            label_data_dict[int(label)] = label_data_shaped

        # label_sequence = [27, 15, 6, 10, 15, 20, 15, 6, 0, 19, 8, 15, 11, 8, 23, 19, 2, 17, 15, 6, 10, 11, 0, 12, 20, 15, 6, 10, 1, 14, 3, 25]

        label_sequence = [27, 15, 6, 10, 15, 20, 15, 6, 0, 19, 8, 15, 11, 8, 23, 19, 2, 17, 15, 6, 10, 11, 0,
                          12, 20, 15, 6, 10, 1, 14, 3, 25, 8, 15, 13, 15, 4, 6, 0, 14, 1, 23, 
                          19, 11, 1, 19, 10, 27, 1, 6, 15, 10, 11, 26, 0, 27, 27, 1, 19, 15, 10, 10, 25]

        selected_data = select_trials_from_sequence(label_sequence, label_data_dict)
        trainX = trainX[:, :,:int(srate * 1)]
        fold_dir = save_dir / f"fold_{k+1}"
        fold_dir.mkdir(exist_ok=True)
        np.save(fold_dir / "trainX.npy", trainX)
        np.save(fold_dir / "trainY.npy", trainY)
        np.save(fold_dir / "testX.npy", selected_data)
        model = train_model(trainX, trainY, srate=srate)
        p_labels = model.predict(selected_data[:,:,:250])
        p_labels2 = model.predict(testX[:, :, : int(srate * 1)])
        accs1 = balanced_accuracy_score(label_sequence, p_labels)
        accs2 = balanced_accuracy_score(testY, p_labels2)
        # 保存一些元数据信息（如形状、类别等）
        with open(fold_dir / "metadata.txt", "w") as f:
            f.write(f"ACC of Label Sequence:{accs1}\n")
            f.write(f"ACC of Test Label:{accs2}\n")
            f.write(f"trainX shape: {trainX.shape}\n")
            f.write(f"trainY shape: {trainY.shape}\n")
            f.write(f"testX shape: {selected_data.shape}\n")
            f.write(f"test labels: {label_sequence}\n")

        print(f"保存折 {k+1} 数据到: {fold_dir}")
    # print("LOO Acc:{:.2f}".format(np.mean(loo_accs)))
