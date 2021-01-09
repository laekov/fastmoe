#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 

#include <cstdio>


class CudaStreamManager {
public:
    size_t num_expert;
    int device;
    cublasHandle_t* handles;
    cudaStream_t* streams;

public:
    CudaStreamManager() : num_expert(0), device(0), streams(NULL) {
        int current_device;
        checkCudaErrors(cudaGetDevice(&current_device));
#ifdef MOE_DEBUG
        printf("constructor at device %d\n", current_device);
#endif
    }

    void setup(const size_t num_expert, const int device) {
#ifdef MOE_DEBUG
        printf("setup at device %d\n", device);
#endif
        this->num_expert = num_expert;
        this->device = device;
        checkCudaErrors(cudaSetDevice(device));        
        streams = new cudaStream_t[num_expert];
        handles = new cublasHandle_t[num_expert];
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamCreate(streams+i));
			checkCudaErrors(cublasCreate(handles + i));
			cublasSetStream(handles[i], streams[i]);
		}
    }

    ~CudaStreamManager() {
#ifdef MOE_DEBUG
        printf("destructor at device %d\n", device);
#endif
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamDestroy(*(streams+i)));
			checkCudaErrors(cublasDestroy(handles[i]));
		}
        delete[] streams;
    }

	void sync(int=-1);
}; 

// CudaStreamManager* getCudaStreamManager(const size_t num_expert, const int device);

#endif  // CUDA_STREAM_MANAGER 
