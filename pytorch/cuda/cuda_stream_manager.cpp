/* TODO: make it ke xue
#include <cuda_runtime.h>
#include <cassert>
#include <thread>

#include "cuda_stream_manager.h"

thread_local CudaStreamManager smgr;


CudaStreamManager* getCudaStreamManager(const size_t num_expert, const int device) { 
    if (!smgr) {
        smgr = new CudaStreamManager(num_expert, device);        
    }
<<<<<<< HEAD
    return smgr;
}

void CudaStreamManager::sync(int i) {
	if (i > -1) {
		cudaStreamSynchronize(streams[i]);
		return;
	}
	for (size_t i=0; i<MAX_STREAMS; ++i) {
		cudaStreamSynchronize(streams[i]);
	}
}
}
*/
