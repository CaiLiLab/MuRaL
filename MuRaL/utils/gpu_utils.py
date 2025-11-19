import sys
import torch
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def get_available_gpu(ntrials):
    nvmlInit()
    gpu_device_count = nvmlDeviceGetCount()
    cuda_id = '0'
    for i in range(gpu_device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_gb = memory_info.free / (2 ** 30)
        print(f'Free GPU memory for cuda:{i}: {free_memory_gb} GB')

        if memory_info.free > (ntrials) * (2.5 * 2 ** 30):
            cuda_id = str(i)
            break

        print('CUDA available:', torch.cuda.is_available())
        print('Using GPU device:', 'cuda:' + cuda_id)
    return cuda_id

def check_cuda_id(cuda_id):
    nvmlInit()
    gpu_device_count = nvmlDeviceGetCount()
    if gpu_device_count <= int(cuda_id):
        print(f'Error: GPU Device Count = {gpu_device_count}, but cuda_id = {cuda_id}. Please set cuda_id not more than {gpu_device_count - 1}.', file=sys.stderr)
        sys.exit(1)
