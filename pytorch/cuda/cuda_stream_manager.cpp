#include <cuda_runtime.h>

#include "cuda_stream_manager.h"

CudaStreamManager* smgr = NULL;

CudaStreamManager* getCudaStreamManager(const size_t num_expert) { 
    if (!smgr) {
        smgr = new CudaStreamManager(num_expert);        
    }
    return smgr;
}

void CudaStreamManager::sync() {
	for (size_t i=0; i<MAX_STREAMS; ++i) {
		cudaStreamSynchronize(streams[i]);
	}
}
