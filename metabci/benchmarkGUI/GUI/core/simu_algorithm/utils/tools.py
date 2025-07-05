import numpy as np
from mne.filter import resample
from pathlib import Path

def save_txt(records:dict, file_name:str, dir_name:str=None)-> None:
    if dir_name:
        save_dir = Path(f"./records/{dir_name}/")
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / file_name
        save_str = ""  # 初始化空字符串
        with open(file_path, "a") as f:
            for key, value in records.items():
                save_str += f"{key}: {value} | "
            save_str = save_str.rstrip(" | ")
            f.write(save_str + "\n")
    else:       
        save_str = ""  # 初始化空字符串
        with open(file_name, "a") as f:
            for key, value in records.items():
                save_str += f"{key}: {value} | "
            save_str = save_str.rstrip(" | ")
            f.write(save_str + "\n")


def data_preprocess(X, srate=1000):
    X = resample(X, up=250, down=srate)
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    return X


def label_encoder(y, labels):
    new_y = y.copy()
    for i, label in enumerate(labels):
        ix = y == label
        new_y[ix] = i
    return new_y
