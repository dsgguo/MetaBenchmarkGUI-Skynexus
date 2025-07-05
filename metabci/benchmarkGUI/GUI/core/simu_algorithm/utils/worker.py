from abc import ABC, abstractmethod
import socket
import numpy as np  
import threading
import time

class BaseWorker(ABC):
    def __init__(
        self,
        worker_id: int,
        server_host: str,
        server_port: int,
        timeout: float,  
        data_sent_event: threading.Event = None,
        data_processed_event: threading.Event = None
    ):
        self.worker_id = worker_id
        self.timeout = timeout  # 超时时间
        # 创建UDP套接字用于发送结果
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 创建TCP服务器用于接收数据
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((server_host, server_port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)  # 设置为非阻塞模式
        self.client_socket = None

        # 添加同步事件
        self.stop_event = threading.Event()
        self.data_sent_event = data_sent_event
        self.data_processed_event = data_processed_event
        
        # 添加数据接收计数
        self.chunks_received = 0
        self.failed_receive_count = 0
        self.last_receive_time = time.time()
        self.max_receive_wait = 5.0  # 最大等待时间(秒)

    @abstractmethod
    def pre(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def consume(self, data):
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self):
        """处理接收数据的主循环"""
        self.pre()  # 初始化
        
        # 通知发送器接收端已准备完毕
        if self.data_processed_event:
            print("初始化完成，通知发送端")
            self.data_processed_event.set()
            
        while not self.stop_event.is_set():
            # 等待数据发送
            if self.data_sent_event:
                # print("等待数据发送...")
                self.data_sent_event.wait(timeout=self.max_receive_wait)
                
                # 检查是否因超时而继续
                if not self.data_sent_event.is_set():
                    self.failed_receive_count += 1
                    print(f"警告：等待数据超时 ({self.failed_receive_count} 次)")
                    
                    # 如果连续多次超时，可能需要重新建立连接
                    if self.failed_receive_count >= 3:
                        print("尝试重新建立客户端连接...")
                        if self.client_socket:
                            self.client_socket.close()
                            self.client_socket = None
                        self.failed_receive_count = 0
                    continue
                    
                self.data_sent_event.clear()
                # print("数据发送完成，开始处理")
                
            # 尝试接受新的客户端连接
            if not self.client_socket:
                self._accept_client()
                if not self.client_socket and self.data_sent_event:
                    # 如果等待数据但没有连接，通知发送者继续尝试
                    print("警告：没有建立连接，通知发送端继续尝试")
                    self.data_processed_event.set()
                    time.sleep(0.5)  # 短暂等待后再尝试
                    continue
            
            # 尝试接收数据
            data = self._receive_data()
            current_time = time.time()
            time_diff = current_time - self.last_receive_time
            
            if data is not None:
                self.chunks_received += 1
                self.failed_receive_count = 0  # 重置失败计数
                self.last_receive_time = current_time
                
                # 处理数据
                self.consume(data)
                
                # 通知发送端数据处理完成
                if self.data_processed_event:
                    self.data_processed_event.set()
            elif self.data_sent_event and time_diff > self.max_receive_wait:
                # 如果等待了发送完成事件，但长时间没收到数据，也通知处理完毕
                print(f"警告：数据块接收失败或超时 ({time_diff:.2f}秒)，通知发送端继续")
                self.failed_receive_count += 1
                self.data_processed_event.set()
                        
            # 如果没有使用同步事件，则按原来的间隔处理
            if not self.data_sent_event:
                time.sleep(self.timeout)
                
    def _accept_client(self):
        """尝试接受新的客户端连接"""
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"已连接到客户端: {addr}")
            self.client_socket.setblocking(False)
        except BlockingIOError:
            pass

    def _receive_data(self):
        """从客户端接收数据"""
        if not self.client_socket:
            return None

        try:
            # 先接收数据大小
            size_data = self.client_socket.recv(8)
            if not size_data:
                print("客户端断开连接")
                self.client_socket.close()
                self.client_socket = None
                return None

            data_size = int.from_bytes(size_data, byteorder='big')

            # 接收实际数据
            received_data = b''
            start_time = time.time()
            while len(received_data) < data_size:
                chunk = self.client_socket.recv(min(data_size - len(received_data), 4096))
                if not chunk:
                    if time.time() - start_time > self.max_receive_wait:
                        print(f"接收数据超时，已接收: {len(received_data)}/{data_size} 字节")
                        break
                    continue
                received_data += chunk
                
            if len(received_data) == data_size:
                # 解析数据
                data = np.frombuffer(received_data, dtype=np.float64)
                # 重塑数据形状
                num_channels = 9
                # 根据发送端的实际数据格式重塑
                samples_per_chunk = len(data) // num_channels
                data = data.reshape((num_channels, samples_per_chunk))
                return data
            else:
                print(f"数据接收不完整: {len(received_data)}/{data_size} 字节")

        except BlockingIOError:
            pass
        except Exception as e:
            print(f"接收数据错误: {e}")

        return None

    def close(self):
        """关闭资源"""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        if self.s:
            self.s.close()
        print(f"Worker已关闭，共接收 {self.chunks_received} 个数据块")

    def stop(self):
        """停止工作线程"""
        self.stop_event.set()
        print("Worker停止信号已发出")
