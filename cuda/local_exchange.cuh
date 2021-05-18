#include "stream_manager.h"
#include "utils/helper_cuda.h"
#include "utils/fmoe_utils.h"

template <typename scalar_t>
__global__
void batch_scatter_kernel(size_t wid, const long* pos, 
        const scalar_t* inbuf, scalar_t* oubuf) { 
    inbuf += wid * pos[blockIdx.x];
    oubuf += wid * blockIdx.x;
    for (int i = threadIdx.x; i < wid; i += blockDim.x) {
        oubuf[i] = inbuf[i];
    }
}

template <typename scalar_t>
void fmoe_cuda_local_scatter_impl(
        const scalar_t* input,
        const long* d_pos,
        scalar_t* input_buf,
        const long batch_size,
        const long in_feat, 
        CudaStreamManager* smgr) {
    batch_scatter_kernel<scalar_t>
        <<<batch_size, 256, 0, smgr->stream(0)>>>(in_feat, d_pos, input,
                input_buf); 
    smgr->sync(1);
}

template <typename scalar_t>
__global__
void batch_gather_kernel(size_t wid, const long* pos, 
        const scalar_t* inbuf, scalar_t* oubuf) { 
    inbuf += wid * blockIdx.x;
    oubuf += wid * pos[blockIdx.x];
    for (int i = threadIdx.x; i < wid; i += blockDim.x) {
        oubuf[i] = inbuf[i];
    }
}

template <typename scalar_t>
void fmoe_cuda_local_gather_impl(
        const scalar_t* output_buf,
        const long* d_pos,
        scalar_t* output,
        const size_t batch_size,
        const size_t out_feat,
        CudaStreamManager* smgr) {
    batch_gather_kernel<scalar_t>
        <<<batch_size, 256, 0, smgr->stream(0)>>>(out_feat, d_pos, output_buf,
                output); 
    smgr->sync(1);
}

__global__
void assign_pos_kernel(int* cum_count, const long* gate, long* pos,
        size_t numel, size_t topk) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < numel) {
        long gate_idx = gate[idx];
        if (gate_idx > -1) {
            int p = atomicSub(cum_count + gate_idx, 1);
            pos[p] = (long)idx;
        }
    }
}

void fmoe_cuda_assign_pos_impl(
        int* cum_count, const long* gate, long* pos,
        const size_t batch_size, const size_t topk,
        CudaStreamManager* smgr) {
    size_t numel = batch_size * topk;
    assign_pos_kernel
        <<<CEIL(numel, 256), 256, 0, smgr->stream(0)>>>(cum_count, gate, pos,
                numel, topk);
    smgr->sync(1);
}
