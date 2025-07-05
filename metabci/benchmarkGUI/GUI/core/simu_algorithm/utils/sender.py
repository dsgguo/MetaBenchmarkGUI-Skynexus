import socket
import threading
import time
import numpy as np

class EEGSender:
    def __init__(
        self,
        X,
        srate=1000,
        host="127.0.0.1",
        port=8888,
        data_sent_event=None,
        data_processed_event=None,
        patch_size=100,
    ):
        self.X = X  # 数据集，形状为(trials, channels, samples)
        self.srate = srate
        self.patch_size = patch_size  # 每个数据块的采样点数
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.stop_event = threading.Event()
        # 添加同步事件
        self.data_sent_event = data_sent_event
        self.data_processed_event = data_processed_event
        # 添加计数器
        self.trials_sent = 0
        self.chunks_sent = 0
        self.retry_count = 0
        self.max_retries = 5  # 最大重试次数

    def connect(self):
        """尝试连接到接收器"""
        try:
            self.socket.connect((self.host, self.port))
            self.connected = True
            # print(f"Sender connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            # print(f"Sender connection error: {e}")
            time.sleep(1)  # 等待一段时间再重试
            return False

    def send_data(self, data, trial_idx=None, chunk_idx=None):
        """发送数据包"""
        if not self.connected:
            if not self.connect():
                return False

        try:
            # 将数据转换为字节
            data_bytes = data.astype(np.float64).tobytes()

            # 发送数据大小
            size_bytes = len(data_bytes).to_bytes(8, byteorder="big")
            self.socket.sendall(size_bytes)

            # 发送实际数据
            self.socket.sendall(data_bytes)
            
            # 成功发送后增加计数
            self.chunks_sent += 1
            # if trial_idx is not None and chunk_idx is not None:
                # print(f"Sent chunk {chunk_idx} of trial {trial_idx+1}/{self.X.shape[0]}")
                
            self.retry_count = 0  # 重置重试计数
            return True
        except Exception as e:
            print(f"Sending error: {e}")
            self.connected = False
            self.retry_count += 1
            return False

    def run(self):
        """运行发送器"""
        # print("Sender waiting 5 seconds before starting...")
        time.sleep(5)  # 启动前等待5秒

        # 初始化 - 等待接收端就绪
        if self.data_processed_event:
            # print("等待接收端初始化完成...")
            self.data_processed_event.wait()
            self.data_processed_event.clear()
            # print("接收端准备就绪，开始发送数据")

        num_trials = self.X.shape[0]
        chunk_size = self.patch_size  # 每个数据块的采样点数
        # print(f"准备发送 {num_trials} 个trial...")

        trial_index = 0
        while not self.stop_event.is_set() and trial_index < num_trials:
            # 获取当前试次的数据
            current_trial = self.X[trial_index]
            # print(f"开始发送 trial {trial_index + 1}/{num_trials}")

            # 一块块发送数据
            num_samples = current_trial.shape[1]
            chunk_count = (num_samples + chunk_size - 1) // chunk_size  # 总块数
            
            for chunk_idx, start_idx in enumerate(range(0, num_samples, chunk_size)):
                if self.stop_event.is_set():
                    break

                # 在发送下一块数据前，确保上一块已处理完成
                if self.data_processed_event and start_idx > 0:
                    # print(f"等待数据处理完成: trial {trial_index+1}, chunk {chunk_idx}/{chunk_count}...")
                    self.data_processed_event.wait()
                    self.data_processed_event.clear()
                    # print(f"数据处理完成，继续发送: trial {trial_index+1}, chunk {chunk_idx+1}/{chunk_count}")

                end_idx = min(start_idx + chunk_size, num_samples)
                chunk = current_trial[:, start_idx:end_idx]

                success = self.send_data(chunk, trial_index, chunk_idx)
                self.retry_count = 0
                
                while not success and self.retry_count < self.max_retries:
                    # print(f"重试发送数据: trial {trial_index+1}, chunk {chunk_idx+1}/{chunk_count}, 第{self.retry_count+1}次重试...")
                    time.sleep(1)
                    self.connect()
                    success = self.send_data(chunk, trial_index, chunk_idx)
                    self.retry_count += 1
                
                if not success:
                    # print(f"警告：发送失败达到最大重试次数，跳过此数据块: trial {trial_index+1}, chunk {chunk_idx+1}/{chunk_count}")
                    continue  # 重试这个块

                # 通知接收端：数据已发送
                if self.data_sent_event:
                    # print(f"通知接收端：trial {trial_index+1}, chunk {chunk_idx+1}/{chunk_count} 数据已发送")
                    self.data_sent_event.set()
                else:
                    # 如果没有同步事件，按原来的固定间隔发送
                    time.sleep(0.1)

            # 完成一个试次后等待1.5秒
            # print(f"完成发送 trial {trial_index + 1}/{num_trials}, 等待1.5秒...")
            time.sleep(1.5)
            trial_index += 1
            self.trials_sent += 1
            
        # print(f"发送器完成所有trial: 共发送了 {self.trials_sent}/{num_trials} 个trial, {self.chunks_sent} 个数据块")

    def stop(self):
        """停止发送器"""
        self.stop_event.set()

    def close(self):
        """关闭资源"""
        if self.connected:
            self.socket.close()
            self.connected = False
