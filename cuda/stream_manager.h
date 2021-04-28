#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include "utils/helper_cuda.h"

#ifdef FMOE_USE_NCCL
#include <nccl.h>

#define NCCL_SAFE_CALL(__fn__) { \
    auto __res__ = __fn__; \
    if (__res__ != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d value %d\n", __FILE__, __LINE__, __res__); \
        exit(-1); \
    } \
}

#endif

class CudaStreamManager {
public:
    int device;
    cublasHandle_t* handles;
    cudaStream_t* streams;
#ifdef FMOE_USE_NCCL
    char ncclgood;
    ncclComm_t ncclcomm;
#endif

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
