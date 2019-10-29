import os
import numpy as np
import torch


class Device:
    def __init__(self, free_only=True, min_memory=2000, max_gpu=4, verbose=True):
        self.free_only = free_only
        self.min_memory = min_memory
        self.max_gpu = max_gpu
        self.device_list = None

        if torch.cuda.is_available():
            self.gpu_list = self.create_gpu_list()
        self.__use_gpu = torch.cuda.is_available() and len(self.gpu_list) > 0

        self.__data_loc = 'cuda:{}'.format(self.gpu_list[0]) if self.__use_gpu else 'cpu'

        if verbose:
            print('[Device]: Computing on ' +
                  ('{} GPU(s)'.format(len(self.gpu_list)) if self.__use_gpu else 'CPU'))

    def create_gpu_list(self):
        if self.free_only:
            gpu_list = self.gpu_memory()
            device_list = np.where(np.array([x[0] for x in gpu_list]) < self.min_memory)[0].tolist()  # less than 50 MB
        else:
            device_list = list(range(torch.cuda.device_count()))
        # Prevent taking all GPUs in a shared server
        device_list = device_list[:self.max_gpu]
        self.device_list = device_list
        return device_list

    def use_gpu(self):
        return self.__use_gpu

    def data_loc(self):
        return self.__data_loc

    def gpu_memory(self):
        raw_list = os.popen('nvidia-smi | grep -o \"[0-9]*MiB / [0-9]*MiB \"').read().strip().split('\n')
        gpu_list = [(int(x[0].strip().strip('MiB')), int(x[1].strip().strip('MiB'))) for x in [x.split(' / ') for x in
                                                                                    raw_list]]
        return gpu_list

    def string_gpu_memory(self):
        try:
            gpu_list = [e for i, e in enumerate(self.gpu_memory()) if i in self.device_list]
            return '; '.join(['/'.join([str(y) for y in x]) for x in gpu_list])
        except:
            return 'GPU Info Error'

