#include <THC/THC.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include "nms_cuda_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
//extern THCState *state;
THCState *state = at::globalContext().getTHCState();

extern "C"{
int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
		     THCudaIntTensor *num_out, float nms_overlap_thresh) {

    nms_cuda_compute(THCudaIntTensor_data(state, keep_out),
                     THCudaIntTensor_data(state, num_out),
                     THCudaTensor_data(state, boxes_host),
                     THCudaTensor_size(state, boxes_host, 0),
                     THCudaTensor_size(state, boxes_host, 1),
                     nms_overlap_thresh);

	return 1;
}
}
