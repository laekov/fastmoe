#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 

#include <cstdio>


class CudaStreamManager {
public:
    CudaStreamManager(const size_t num_expert_, const int device_) : num_expert(num_expert_), device(device_) {
        checkCudaErrors(cudaSetDevice(device));
        printf("set device %d\n", device);
        streams = new cudaStream_t[num_expert];
        checkCudaErrors(cublasCreate(&handle));
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamCreate(streams+i));
        }
    }
    ~CudaStreamManager() {
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamDestroy(*(streams+i)));
        }
        checkCudaErrors(cublasDestroy(handle));
    }
    const size_t num_expert;
    const int device;
    cublasHandle_t handle;
    cudaStream_t* streams;
}; 

CudaStreamManager* getCudaStreamManager(const size_t num_expert, const int device);

#endif  // CUDA_STREAM_MANAGER 
