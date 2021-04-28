#include "stream_manager.h"
#include "utils/helper_cuda.h"

template <typename scalar_t>
__global__
void generate_ptr_offset_kernel(size_t n, const scalar_t* base, size_t stride,
        const long* offset, const scalar_t** ptrs) { 
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        ptrs[idx] = base + stride * offset[idx];
    }
}

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

void fmoe_cuda_expert_count_impl(
        const int* d_gate,
        int* expert_count,
        int* d_pos,
        const size_t num_expert,
        const size_t batch_size) {
    int *gate = new int[batch_size];
    int *expert_ptr = new int[num_expert];
    memset(expert_count, 0, sizeof(int) * num_expert);

    checkCudaErrors(cudaMemcpy(gate, d_gate, sizeof(int) * batch_size,
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size; ++i) {
        ++expert_count[gate[i]];
    }
    expert_ptr[0] = 0;
    for (int i = 1; i < num_expert; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + expert_count[i - 1];
    }

    int *pos = new int[batch_size];

    for (int i = 0; i < batch_size; ++i) {
        pos[i] = expert_ptr[gate[i]]++;
    }
    for (int i = num_expert - 1; i > 0; --i) {
        expert_ptr[i] = expert_ptr[i - 1];
    }
    expert_ptr[0] = 0;
    checkCudaErrors(cudaMemcpy(d_pos, pos, sizeof(int) * batch_size,
                cudaMemcpyHostToDevice));
    delete [] gate;
    delete [] expert_ptr;
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
