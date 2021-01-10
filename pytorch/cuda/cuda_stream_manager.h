#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

class CudaStreamManager {
public:
    int device;
    cublasHandle_t* handles;
    cudaStream_t* streams;

public:
    CudaStreamManager(int device_): device(device_) {
		this->setup(device);
    }

	void setup(int);
	void sync(int=0);
	void destroy();

	cudaStream_t stream(size_t=0);
	cublasHandle_t handle(size_t=0);

    ~CudaStreamManager() {
		this->destroy();
    }
}; 

CudaStreamManager* getCudaStreamManager(const int device);

#endif  // CUDA_STREAM_MANAGER 
