# CUDA Device Information Parser

![](https://img.shields.io/badge/Made%20for-PyTorch-orange)

A Python module helping determing GPU device information, for PyTorch users who share machine with others.

## Prerequisites

* PyTorch
* NumPy
* The NVIDIA System Management Interface

## Usage

The object will keep the list of devices that meet the requirements specified at initialization. 

* Use `device_list` to get the numbers of devices selected
* Use `data_loc()` to get the location the data should be stored, especially if using DataParallel
* Use `string_gpu_memory()`  to print the current status of GPUs selected. 

## TODO

* Use `nvidia-smi` interface instead of parsing the output