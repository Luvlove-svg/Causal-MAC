# communication/protocol.py
import os
import subprocess
import threading
import queue
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .message_utils import MessageEncoder

class MCPProtocol:
    """消息控制协议（MCP）实现，支持Stdio模式"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MCP协议
        
        参数:
            config: 协议配置
        """
        self.config = config
        self.mode = config.get("mode", "stdio")
        self.encoder = MessageEncoder(config.get("encoding", {}))
        self.processes = {}  # 工具进程字典
        self.message_queues = {}  # 消息队列
        self.threads = {}  # 监听线程
        self.running = False
        
        # 注册工具
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict]:
        """注册可用工具"""
        tools = {
            "path_planner": {
                "command": ["python", "tools/path_planner.py"],
                "description": "路径规划工具",
                "input_type": "numpy_array",
                "output_type": "numpy_array"
            },
            "object_detector": {
                "command": ["python", "tools/object_detector.py"],
                "description": "目标检测工具",
                "input_type": "numpy_array",
                "output_type": "numpy_array"
            },
            "decision_support": {
                "command": ["python", "tools/decision_support.py"],
                "description": "决策支持工具",
                "input_type": "numpy_array",
                "output_type": "numpy_array"
            },
            "communication_relay": {
                "command": ["python", "tools/communication_relay.py"],
                "description": "通信中继工具",
                "input_type": "json",
                "output_type": "json"
            }
        }
        return tools
    
    def start(self):
        """启动MCP服务"""
        if self.running:
            return
        
        self.running = True
        
        # 启动所有工具进程
        for tool_name, tool_info in self.tools.items():
            self._start_tool_process(tool_name)
    
    def stop(self):
        """停止MCP服务"""
        self.running = False
        
        # 停止所有工具进程
        for tool_name in list(self.processes.keys()):
            self._stop_tool_process(tool_name)
    
    def _start_tool_process(self, tool_name: str):
        """启动单个工具进程"""
        if tool_name in self.processes:
            return
        
        tool_info = self.tools[tool_name]
        
        # 创建进程
        process = subprocess.Popen(
            tool_info["command"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,  # 使用二进制模式
            bufsize=0    # 无缓冲
        )
        
        # 创建消息队列
        self.message_queues[tool_name] = queue.Queue()
        
        # 创建监听线程
        thread = threading.Thread(
            target=self._listen_to_tool,
            args=(tool_name, process),
            daemon=True
        )
        thread.start()
        
        self.processes[tool_name] = process
        self.threads[tool_name] = thread
    
    def _stop_tool_process(self, tool_name: str):
        """停止单个工具进程"""
        if tool_name not in self.processes:
            return
        
        process = self.processes[tool_name]
        
        # 发送退出命令
        try:
            if self.tools[tool_name]["input_type"] == "json":
                exit_cmd = json.dumps({"command": "exit"}).encode('utf-8')
            else:
                exit_cmd = b"EXIT"
            
            process.stdin.write(exit_cmd + b"\n")
            process.stdin.flush()
        except:
            pass
        
        # 终止进程
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
        
        # 移除记录
        del self.processes[tool_name]
        del self.message_queues[tool_name]
        del self.threads[tool_name]
    
    def _listen_to_tool(self, tool_name: str, process: subprocess.Popen):
        """监听工具输出"""
        while self.running:
            try:
                # 读取一行输出
                raw_output = process.stdout.readline()
                if not raw_output:
                    break
                
                # 解码消息
                try:
                    if self.tools[tool_name]["output_type"] == "json":
                        output = json.loads(raw_output.decode('utf-8'))
                    else:
                        output = self.encoder.decode(raw_output)
                except:
                    output = {"error": "Failed to decode message"}
                
                # 添加到消息队列
                self.message_queues[tool_name].put(output)
            except Exception as e:
                print(f"监听工具 {tool_name} 时出错: {str(e)}")
                break
    
    def send_message(self, tool_name: str, message: Any, metadata: Dict = None) -> bool:
        """向工具发送消息"""
        if tool_name not in self.processes:
            return False
        
        process = self.processes[tool_name]
        
        try:
            # 编码消息
            if self.tools[tool_name]["input_type"] == "json":
                if not isinstance(message, dict):
                    message = {"data": message}
                if metadata:
                    message.update(metadata)
                encoded = json.dumps(message).encode('utf-8')
            else:
                if isinstance(message, np.ndarray):
                    encoded = self.encoder.encode(message, metadata)
                else:
                    # 尝试转换为numpy数组
                    try:
                        array = np.array(message)
                        encoded = self.encoder.encode(array, metadata)
                    except:
                        encoded = json.dumps({"data": message, "metadata": metadata}).encode('utf-8')
            
            # 发送消息
            process.stdin.write(encoded + b"\n")
            process.stdin.flush()
            return True
        except Exception as e:
            print(f"向工具 {tool_name} 发送消息失败: {str(e)}")
            return False
    
    def receive_message(self, tool_name: str, timeout: float = 0.1) -> Any:
        """从工具接收消息"""
        if tool_name not in self.message_queues:
            return None
        
        try:
            return self.message_queues[tool_name].get(timeout=timeout)
        except queue.Empty:
            return None
    
    def call_tool(self, tool_name: str, input_data: Any, metadata: Dict = None, timeout: float = 1.0) -> Any:
        """调用工具并获取响应"""
        if not self.send_message(tool_name, input_data, metadata):
            return None
        
        # 等待响应
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.receive_message(tool_name)
            if response is not None:
                return response
            time.sleep(0.01)
        
        return None
    
    def broadcast(self, message: Any, metadata: Dict = None, exclude: List[str] = None):
        """广播消息到所有工具"""
        exclude = exclude or []
        for tool_name in self.tools:
            if tool_name not in exclude:
                self.send_message(tool_name, message, metadata)
    
    def tool_status(self) -> Dict[str, str]:
        """获取工具状态"""
        status = {}
        for tool_name, process in self.processes.items():
            status[tool_name] = "running" if process.poll() is None else "stopped"
        return status