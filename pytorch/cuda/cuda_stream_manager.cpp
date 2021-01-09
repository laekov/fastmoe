#include <cuda_runtime.h>
#include <cassert>
#include <thread>

#include "cuda_stream_manager.h"

void CudaStreamManager::sync(int i) {
	if (i > -1) {
		cudaStreamSynchronize(streams[i]);
		return;
	}
	for (size_t i = 0; i < this->num_expert; ++i) {
		cudaStreamSynchronize(streams[i]);
	}
}

void CudaStreamManager::setup(const size_t num_expert, const int device) {
#ifdef MOE_DEBUG
	printf("setup at device %d\n", device);
#endif
	this->num_expert = num_expert;
	if (device == -1) {
        checkCudaErrors(cudaGetDevice(&this->device));
	} else {
		this->device = device;
	}
	checkCudaErrors(cudaSetDevice(this->device));
	streams = new cudaStream_t[num_expert];
	handles = new cublasHandle_t[num_expert];
	for (size_t i=0; i<num_expert; ++i) {
		checkCudaErrors(cudaStreamCreate(streams+i));
		checkCudaErrors(cublasCreate(handles + i));
		cublasSetStream(handles[i], streams[i]);
	}
}

