#include <cassert>

#include "cuda_stream_manager.h"

CudaStreamManager* smgr = NULL;

CudaStreamManager* getCudaStreamManager(const size_t num_expert) { 
    if (!smgr) {
        smgr = new CudaStreamManager(num_expert);        
    }
    assert(smgr->num_expert == num_expert);
    return smgr;
}
