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
    CudaStreamManager() : num_expert(0), streams(NULL) {
        int current_device;
        checkCudaErrors(cudaGetDevice(&current_device));
#ifdef MOE_DEBUG
        printf("constructor at device %d\n", current_device);
#endif
    }

    void setup(const size_t num_expert, const int device=-1);

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

#define ENSURE_SMGR(__smgr__, __num_expert__) { \
	if (__smgr__.num_expert == 0) { \
		__smgr__.setup(__num_expert__); \
	} \
}

// CudaStreamManager* getCudaStreamManager(const size_t num_expert, const int device);

#endif  // CUDA_STREAM_MANAGER 
