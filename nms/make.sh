#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

cd src/cuda
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_61
cd ../../
python build.py
