#!/usr/bin/env python3
"""
检查Docker容器中可用的GPU设备
"""

import torch
import os

def check_gpu_devices():
    print("=== GPU设备检查 ===")
    
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 检查GPU设备数量
        device_count = torch.cuda.device_count()
        print(f"GPU设备数量: {device_count}")
        
        # 列出所有可用的GPU设备
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {device_name}")
            print(f"  - 总内存: {device_props.total_memory / 1024**3:.2f} GB")
            print(f"  - 多处理器数量: {device_props.multi_processor_count}")
            
            # 检查当前内存使用情况
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  - 已分配内存: {allocated:.2f} GB")
            print(f"  - 已保留内存: {reserved:.2f} GB")
            print()
        
        # 检查当前默认设备
        current_device = torch.cuda.current_device()
        print(f"当前默认设备: cuda:{current_device}")
        
        # 测试指定设备是否可用
        test_devices = ['cuda:0', 'cuda:1', 'cuda:2']
        for device in test_devices:
            try:
                if int(device.split(':')[1]) < device_count:
                    test_tensor = torch.tensor([1.0]).to(device)
                    print(f"设备 {device}: 可用 ✓")
                else:
                    print(f"设备 {device}: 不存在 ✗")
            except Exception as e:
                print(f"设备 {device}: 错误 - {e}")
    else:
        print("CUDA不可用，只能使用CPU")
    
    # 检查环境变量
    print("\n=== 环境变量检查 ===")
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES')
    print(f"NVIDIA_VISIBLE_DEVICES: {nvidia_visible_devices}")

if __name__ == "__main__":
    check_gpu_devices()