#include <cassert>
#include <thread>

#include "cuda_stream_manager.h"

thread_local CudaStreamManager* smgr = NULL;

CudaStreamManager* getCudaStreamManager(const size_t num_expert, const int device) { 
    if (!smgr) {
        smgr = new CudaStreamManager(num_expert, device);        
    }
    assert(smgr->num_expert == num_expert);
    assert(smgr->device == device);
    return smgr;
}
