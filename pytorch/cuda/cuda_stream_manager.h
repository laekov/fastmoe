#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 


#define MAX_STREAMS 16


struct CudaStreamManager {
    const size_t num_expert;
    cublasHandle_t* handles;
    cudaStream_t* streams;

    CudaStreamManager(const size_t num_expert_) : num_expert(num_expert_) {
        streams = new cudaStream_t[MAX_STREAMS];
		handles = new cublasHandle_t[MAX_STREAMS];
        for (size_t i=0; i<MAX_STREAMS; ++i) {
			checkCudaErrors(cublasCreate(handles + i));
			checkCudaErrors(cudaStreamCreate(streams + i));
			checkCudaErrors(cublasSetStream(handles[i], streams[i]));
		}
    }

    ~CudaStreamManager() {
        for (size_t i=0; i<MAX_STREAMS; ++i) {
            checkCudaErrors(cudaStreamDestroy(streams[i]));
			checkCudaErrors(cublasDestroy(handles[i]));
		}
    }

	inline cudaStream_t& getStream(int idx) {
		return streams[idx % MAX_STREAMS];
	}
	inline cublasHandle_t& getHandle(int idx) {
		return handles[idx % MAX_STREAMS];
	}

	void sync(int=-1);
}; 

CudaStreamManager* getCudaStreamManager(const size_t num_expert);

#endif  // CUDA_STREAM_MANAGER 
