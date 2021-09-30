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

#define PERTHREAD_EXPERTS 256

#ifdef FMOE_USE_HIP
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

__global__
void expert_count_kernel(const long* gate_idx, int* expert_count,
        const size_t batch_size, const size_t n_expert) {
    int res_tmp[PERTHREAD_EXPERTS] = {0};
    long expert_min = blockIdx.x * PERTHREAD_EXPERTS;
    long expert_max = expert_min + PERTHREAD_EXPERTS;
    if (expert_max > n_expert) {
        expert_max = n_expert;
    }
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
        long idx = gate_idx[i];
        if (idx == -1) {
            continue;
        }
        if (idx < expert_min || idx >= expert_max) {
            continue;
        }
        res_tmp[idx - expert_min] += 1;
    }
    for (int i = expert_min; i < expert_max; ++i) {
        int x = res_tmp[i - expert_min];
#pragma unroll
        for (int j = 1; j < WARP_SIZE; j <<= 1) {
#ifdef FMOE_USE_HIP
            x = x + __shfl_down(x, j);
#else
            x = x + __shfl_down_sync(-1u, x, j);
#endif
        }
        if (threadIdx.x % WARP_SIZE == 0) {
            atomicAdd(expert_count + i, x);
        }
    }
}

void fmoe_cuda_expert_count_impl(
        const long* gate_idx, int* expert_count,
        const size_t batch_size, const size_t n_expert,
        CudaStreamManager* smgr) {
    expert_count_kernel
        <<<CEIL(n_expert, PERTHREAD_EXPERTS), 256, 0, smgr->stream(0)>>>
        (gate_idx, expert_count, batch_size, n_expert);
    smgr->sync(1);
}
