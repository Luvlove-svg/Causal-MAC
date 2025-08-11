# communication/message_utils.py
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import base64
import zlib

class MessageEncoder:
    """消息编码工具，支持高效序列化和压缩"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compression_level = config.get("compression_level", 6)
        self.use_binary = config.get("use_binary", True)
    
    def encode(self, message: np.ndarray, metadata: Dict[str, Any] = None) -> bytes:
        """编码消息为字节流"""
        # 创建消息字典
        message_dict = {
            'data': message.tolist(),
            'dtype': str(message.dtype),
            'shape': message.shape,
            'metadata': metadata or {}
        }
        
        # 序列化为JSON
        json_str = json.dumps(message_dict)
        
        if self.use_binary:
            # 转换为二进制
            binary_data = np.ascontiguousarray(message).tobytes()
            
            # 压缩
            compressed_data = zlib.compress(binary_data, level=self.compression_level)
            
            # 添加头部信息
            header = json.dumps({
                'dtype': str(message.dtype),
                'shape': message.shape,
                'metadata': metadata or {}
            }).encode('utf-8')
            
            # 组合为完整消息: [header_length:4字节][header][compressed_data]
            header_len = len(header)
            full_message = (
                header_len.to_bytes(4, 'big') + 
                header + 
                compressed_data
            )
            return full_message
        else:
            # 文本模式
            return json_str.encode('utf-8')
    
    def decode(self, encoded: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """从字节流解码消息"""
        if self.use_binary:
            # 解析头部长度
            header_len = int.from_bytes(encoded[:4], 'big')
            header = encoded[4:4+header_len]
            compressed_data = encoded[4+header_len:]
            
            # 解压缩
            binary_data = zlib.decompress(compressed_data)
            
            # 解析头部
            header_dict = json.loads(header.decode('utf-8'))
            
            # 重建数组
            array = np.frombuffer(binary_data, dtype=np.dtype(header_dict['dtype']))
            array = array.reshape(header_dict['shape'])
            
            return array, header_dict['metadata']
        else:
            # 文本模式
            message_dict = json.loads(encoded.decode('utf-8'))
            array = np.array(message_dict['data'], dtype=message_dict['dtype'])
            return array, message_dict['metadata']
    
    def encode_batch(self, messages: List[np.ndarray], metadata: List[Dict] = None) -> bytes:
        """批量编码消息"""
        if metadata is None:
            metadata = [{}] * len(messages)
        
        if self.use_binary:
            # 组合所有消息
            all_data = b''
            for msg, meta in zip(messages, metadata):
                all_data += self.encode(msg, meta)
            return all_data
        else:
            # 文本模式批量编码
            batch = [
                {'data': msg.tolist(), 'dtype': str(msg.dtype), 'shape': msg.shape, 'metadata': meta}
                for msg, meta in zip(messages, metadata)
            ]
            return json.dumps(batch).encode('utf-8')
    
    def decode_batch(self, encoded: bytes) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """批量解码消息"""
        if self.use_binary:
            # 二进制模式需要特殊处理
            results = []
            offset = 0
            while offset < len(encoded):
                # 读取头部长度
                header_len = int.from_bytes(encoded[offset:offset+4], 'big')
                offset += 4
                
                # 读取头部
                header = encoded[offset:offset+header_len]
                offset += header_len
                
                # 读取数据长度（从头部获取）
                header_dict = json.loads(header.decode('utf-8'))
                data_size = np.prod(header_dict['shape']) * np.dtype(header_dict['dtype']).itemsize
                
                # 读取压缩数据
                compressed_data = encoded[offset:offset+data_size]
                offset += data_size
                
                # 解码单个消息
                msg, meta = self.decode(
                    header_len.to_bytes(4, 'big') + header + compressed_data
                )
                results.append((msg, meta))
            
            return results
        else:
            # 文本模式批量解码
            batch = json.loads(encoded.decode('utf-8'))
            results = []
            for item in batch:
                array = np.array(item['data'], dtype=item['dtype'])
                results.append((array, item['metadata']))
            return results