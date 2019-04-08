cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.o nms_cuda_kernel.cu -x cu -Xcompiler /MD
cd ..\\

py -3 build.py