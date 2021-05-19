#include "stream_manager.h"
#include "utils/helper_cuda.h"
#include "utils/fmoe_utils.h"

__global__
void assign_pos_kernel(int* cum_count, const long* gate, long* pos,
        size_t numel, size_t topk) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numel) {
        long gate_idx = gate[idx];
        if (gate_idx > -1) {
            int p = atomicSub(cum_count + gate_idx, 1);
            pos[p - 1] = (long)idx;
        }
    }
}

void fmoe_cuda_assign_pos_impl(
        int* cum_count, const long* gate, long* pos,
        const size_t batch_size, const size_t topk,
        CudaStreamManager* smgr) {
    size_t numel = batch_size * topk;
    assign_pos_kernel
        <<<CEIL(numel, 256), 256, 0, smgr->stream(0)>>>
        (cum_count, gate, pos, numel, topk);
    smgr->sync(1);
}
