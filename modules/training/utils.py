import torch
import subprocess
import psutil


def print_available_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info = subprocess.check_output(f'nvidia-smi -i {i} --query-gpu=memory.free --format=csv,nounits,noheader', shell=True)
            memory_info = str(memory_info).split("\\")[0][2:]
            print(f"GPU {i} memory available: {memory_info} MiB")
    else:
        print("No GPUs available")


def print_available_ram():
    memory_info = psutil.virtual_memory()
    
    print(f"Available RAM: {memory_info.available / (1024.0 ** 3)} GB")  # convert bytes to GB
