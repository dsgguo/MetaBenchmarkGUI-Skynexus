# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan - 基于队列输出的版本

"""
import time
import numpy as np
from sklearn.base import clone
import socket
import tomllib
from pathlib import Path
import threading
import queue

import mne
from mne.io import read_raw_cnt
from mne.filter import resample

from metabci.brainda.algorithms.decomposition import (
    FBTRCA,
    generate_filterbank,
    generate_cca_references,
)
from metabci.brainda.utils import upper_ch_names
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from metabci.brainda.algorithms.dynamic_stopping import ce
# from utils.decomposition import ce

from utils.sender import EEGSender
from utils.tools import save_txt, data_preprocess
from utils.worker import BaseWorker

# 全局输出队列
_output_queue = None

def set_output_queue(output_queue):
    """设置全局输出队列"""
    global _output_queue
    _output_queue = output_queue

def queue_print(*args, **kwargs):
    """自定义print函数，将输出发送到队列"""
    global _output_queue
    
    # 构造输出字符串
    output = ' '.join(str(arg) for arg in args)
    
    # 如果有队列，发送到队列
    if _output_queue is not None:
        try:
            _output_queue.put(output)
        except Exception as e:
            # 如果队列出错，回退到标准输出
            print(f"队列输出失败: {e}")
            print(output)
    else:
        # 没有队列时使用标准输出
        print(output)

# 替换utils模块中的print函数
def patch_utils_print():
    """为utils模块打补丁，使用queue_print替换print"""
    try:
        import utils.sender
        import utils.worker
        
        # 替换sender模块中的print
        utils.sender.print = queue_print
        
        # 替换worker模块中的print
        utils.worker.print = queue_print
        
    except ImportError as e:
        queue_print(f"警告：无法导入utils模块进行打补丁: {e}")


def train_model(X, y, srate=1000):

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

def model_predict(X, srate=1000, model=None):
    X = resample(X, up=250, down=srate)
    X = np.reshape(X, (-1, X.shape[-2], X.shape[-1]))
    X = X - np.mean(X, axis=-1, keepdims=True)
    X = X / np.std(X, axis=(-1, -2), keepdims=True)
    p_labels = model.predict(X)
    return p_labels

def offline_validation(X, y, srate=1000):
    y = np.reshape(y, (-1))
    spliter = EnhancedLeaveOneGroupOut(return_validate=False)

    kfold_accs = []
    for train_ind, test_ind in spliter.split(X, y=y):
        X_train, y_train = np.copy(X[train_ind]), np.copy(y[train_ind])
        X_test, y_test = np.copy(X[test_ind]), np.copy(y[test_ind])
        model = train_model(X_train, y_train, srate=srate)
        p_labels = model_predict(X_test, srate=srate, model=model)
        kfold_accs.append(np.mean(p_labels == y_test))
    return np.mean(kfold_accs)


def train_ds(X, y, srate, start: int, end: int):
    wp = [[6, 88], [14, 88], [22, 88], [30, 88], [38, 88]]
    ws = [[4, 90], [12, 90], [20, 90], [28, 90], [36, 90]]

    filterweights = np.arange(1, 6) ** (-1.25) + 0.25
    filterbank = generate_filterbank(wp, ws, 250)
    ce_model = ce.CE(
        clone(
            FBTRCA(
                filterbank=filterbank,
                n_components=1,
                filterweights=np.array(filterweights),
            )
        ),
        30,
    )
    y = np.reshape(y, (-1))

    for duration in np.arange(start, end, 1):
        duration = duration / 10
        queue_print(f"Currunt_Train_Duration: {duration}")

        time_point = int(duration * srate)
        X_duration = X[:, :, :time_point]
        data_pre = data_preprocess(X_duration, srate)
        ce_model.fit(data_pre, y, duration)
    queue_print("CE model trained")
    return ce_model


class FeedbackWorker(BaseWorker):
    def __init__(
        self,
        file_path: str,
        pick_chs,
        stim_interval,
        stim_labels,
        srate,
        timeout,
        worker_name,
        word_dict,
        server_host="127.0.0.1",
        server_port=8888,
        data_sent_event=None,
        data_processed_event=None,
    ):
        # 调用父类BaseWorker的__init__方法
        super().__init__(
            worker_id=0,  # 默认worker_id为0
            server_host=server_host,
            server_port=server_port,
            timeout=timeout,
            data_sent_event=data_sent_event,
            data_processed_event=data_processed_event,
        )

        self.file_path = file_path
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.worker_name = worker_name

        # 创建UDP套接字用于发送结果
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 初始化当前预测次数
        self.word_dict = word_dict
        self.t = 0
        self.current_word = " "
        self.pool = np.array([])
        self.bool_result = False
        self.status = False

          
        self.last_trial_time = time.time()

        queue_print(f"FeedbackWorker initialized. Listening on {server_host}:{server_port}")

    def pre(self):
        X_file_path = self.file_path / "trainX.npy"
        y_file_path = self.file_path / "trainY.npy"
        X = np.load(X_file_path, allow_pickle=True)
        y = np.load(y_file_path, allow_pickle=True)

        # 离线验证
        acc = offline_validation(X, y, srate=self.srate)
        queue_print(f"Current Model accuracy: {acc:.2f}")
        self.estimator = train_model(X, y, srate=self.srate)
        self.ce_model = train_ds(X, y, self.srate, 5, 11)
        queue_print("Initialization complete, ready to receive data...")

    def consume(self, data):
        np.random.seed(42)

        if len(self.pool) == 0:
            self.pool = data
        else:
            self.pool = np.concatenate((self.pool, data), axis=1)

        # 改变变量名，避免与time模块冲突
        time_threshold = 0.5
        # 当累积足够的数据时进行处理
        if (
            self.pool.shape[1] >= time_threshold * self.srate
            and self.bool_result == False
        ):
            data_pre = data_preprocess(self.pool, srate=self.srate)
            duration = self.pool.shape[1] / self.srate
            bool_result, features = self.ce_model.transform(data_pre, duration, t_max = 1)
            # features = self.estimator.transform(data_pre)
            p_labels = self.estimator.classes_[np.argmax(features, axis=-1)]  # 分类字符
            p_labels = [p_labels]
            self.p_labels = [int(label) for label in p_labels]
            self.bool_result = bool_result
            if bool_result:
                self.status = True

        if self.bool_result:
            if self.status:
                match self.word_dict[self.p_labels[0]]:
                    case "Back":
                        if self.t > 0:
                            self.current_word = self.current_word[:-1]
                            self.t -= 1
                    case " " | ".":
                        self.current_word = " "
                        self.t = 0
                    case char:
                        self.current_word += char  # 添加新字符
                        self.t += 1
                duration = self.pool.shape[1] / self.srate
                records = {
                    "Time": duration,
                    "Current word": self.current_word,
                    "Predict label": self.p_labels}
                queue_print(f"duration: {duration:.2f}s, current word:{self.current_word} label: {self.p_labels[0]}")
                path = Path(self.file_path)
                dir_name = f"{path.parts[-2]}\\{path.parts[-1]}"
                save_txt(records, file_name="Ds_records.txt", dir_name=dir_name)
                self.status = False

        if self.pool.shape[1] == int(1.5 * self.srate):
            self.bool_result = False
            self.pool = np.array([])
            self.p_labels = [11]


def simu_run(fold_path, stop_event=None, max_duration=300, output_queue=None):
    # 设置全局输出队列
    set_output_queue(output_queue)
    
    # 为utils模块打补丁
    patch_utils_print()
    
    current_dir = Path(__file__).parent
    config_path = current_dir / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # eeg config
    eeg_conf = config["eeg"]
    srate = eeg_conf["srate"]
    stim_interval = eeg_conf["stim_interval"]
    stim_labels = eeg_conf["stim_labels"]
    pick_chs = eeg_conf["pick_chs"]

    # worker config
    feedback_worker_name = "feedback_worker"

    # synchro event
    data_sent_event = threading.Event()
    data_processed_event = threading.Event()

    file_path = Path(fold_path)
    word_dict = config["worker"]["word_dict"]
    word_dict = {int(k): v for k, v in word_dict.items() if k.isdigit()}
    worker = FeedbackWorker(
        file_path=file_path,
        pick_chs=pick_chs,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        word_dict=word_dict,
        srate=srate,
        timeout=5e-2,
        worker_name=feedback_worker_name,
        data_sent_event=data_sent_event,
        data_processed_event=data_processed_event,
    )

    # 准备sender用的数据
    test_files = file_path / "testX.npy"
    queue_print(f"Loading test data from: {test_files}")
    X_sender = np.load(test_files)

    sender = EEGSender(
        X=X_sender,
        srate=srate,
        host="127.0.0.1",  # 连接到本地接收器
        data_sent_event=data_sent_event,
        data_processed_event=data_processed_event,
        patch_size=25,
    )
    # 启动线程
    worker_thread = None
    sender_thread = None
    try:
        # 先启动worker线程
        worker_thread = threading.Thread(target=worker.run)
        worker_thread.daemon = True
        worker_thread.start()
        queue_print("Worker thread started")

        # 然后启动sender线程
        sender_thread = threading.Thread(target=sender.run)
        sender_thread.daemon = True
        sender_thread.start()
        queue_print("Sender thread started")

        # 监控线程 - 添加停止条件和超时
        start_time = time.time()
        while (worker_thread.is_alive() and sender_thread.is_alive()):
            # 检查停止事件
            if stop_event and stop_event.is_set():
                queue_print("接收到停止信号，正在停止...")
                break
                
            # 检查超时
            if time.time() - start_time > max_duration:
                queue_print(f"达到最大运行时间 {max_duration}s，正在停止...")
                break
                
            time.sleep(1)

    except KeyboardInterrupt:
        queue_print("Stopping all threads...")
    except Exception as e:
        queue_print(f"运行过程中出现错误: {e}")
    finally:
        # 安全地停止线程和清理资源
        queue_print("正在清理资源...")
        
        try:
            # 停止worker和sender
            worker.stop()
            sender.stop()
        except Exception as e:
            queue_print(f"停止服务时出错: {e}")
        
        # 等待线程结束，设置超时避免无限等待
        if worker_thread and worker_thread.is_alive():
            worker_thread.join(timeout=3.0)
            if worker_thread.is_alive():
                queue_print("警告: Worker线程未能正常结束")
                
        if sender_thread and sender_thread.is_alive():
            sender_thread.join(timeout=3.0)
            if sender_thread.is_alive():
                queue_print("警告: Sender线程未能正常结束")
        
        # 关闭资源
        try:
            worker.close()
        except Exception as e:
            queue_print(f"关闭worker时出错: {e}")
            
        try:
            sender.close()
        except Exception as e:
            queue_print(f"关闭sender时出错: {e}")
            
        queue_print("All resources closed")
