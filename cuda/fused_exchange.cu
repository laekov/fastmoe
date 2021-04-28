#include "moe_cuda_kernel.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_stream_manager.h"
#include "cublas_wrapper.h"

#ifdef FMOE_USE_NCCL
#include <nccl.h>

template<typename scalar_t>
void moe_cuda_global_fused_forward_impl(
        const scalar_t* input_buf,
        const scalar_t* weight,
        scalar_t* global_input_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,
        const long* local_expert_count, 
        const long* global_expert_count, 
        long in_feat, long out_feat, 
        long num_expert, long world_size,
        CudaStreamManager* smgr) {

    int ptr = 0;
    int send_ptr = 0;
    int recv_ptr = 0;

    int *expert_ptr = new int[num_expert * world_size];
    expert_ptr[0] = 0;
    for (int i = 1; i < num_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    scalar_t alpha = 1, beta = 0; 

    for (int i = 0; i < num_expert; ++i) {
        int expert_count = 0;
        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = i + j * num_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        input_buf + expert_ptr[idx] * in_feat, 
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(i)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        global_input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(i)));
                recv_ptr += global_expert_count[idx];
                expert_count += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());

        checkCudaErrors(cublasXgemm(
                smgr->handle(i),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                out_feat, expert_count, in_feat,
                &alpha,
                weight + i * in_feat * out_feat, in_feat,
                global_input_buf + ptr * in_feat, in_feat,
                &beta,
                global_output_buf + out_feat * ptr, out_feat
                ));

        ptr += expert_count;

        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = i + j * num_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        global_output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(i)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        output_buf + expert_ptr[idx] * out_feat, 
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(i)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(num_expert);
}

std::vector<torch::Tensor> moe_cuda_global_fused_forward(
        torch::Tensor input_buf,
        torch::Tensor weight,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long global_batch_size, long local_batch_size, long n_workers) {
    const auto num_expert = local_expert_count.size(0) / n_workers;
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);

    auto smgr = getCudaStreamManager(input_buf.device().index());

    auto global_input_buf = input_buf.new_empty({global_batch_size, in_feat});
    auto global_output_buf = input_buf.new_empty({global_batch_size, out_feat});
    auto output_buf = input_buf.new_empty({local_batch_size, out_feat});
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "moe_cuda_global_fused_forward", ([&] {
        moe_cuda_global_fused_forward_impl(
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            global_input_buf.data_ptr<scalar_t>(),
            global_output_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            in_feat, out_feat, num_expert, n_workers,
            smgr);
    }));
    return {output_buf, global_input_buf};
}

#endif

