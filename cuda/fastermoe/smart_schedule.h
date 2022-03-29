#ifndef SMART_SCHEDULE_H
#define SMART_SCHEDULE_H

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "../stream_manager.h"


template<typename scalar_t>
void _exchange_with(
        const scalar_t* sendbuf, size_t sendcount, int t_send,
        scalar_t* recvbuf, size_t recvcount, int t_recv,
        long d_model,
        cudaStream_t stream, ncclComm_t comm) {
    if (sendcount) {
        ncclSend(sendbuf, sendcount * d_model * sizeof(scalar_t),
                ncclChar, t_send , comm, stream);
    }
    if (recvcount) {
        ncclRecv(recvbuf, recvcount * d_model * sizeof(scalar_t),
                ncclChar, t_recv, comm, stream);
    }
}


#define GEN_BASE(_step) \
    long to_base = (group_rank + _step) % n_groups * pipeline_gran; \
    long from_base = (group_rank + n_groups - _step) % n_groups * pipeline_gran;
#define GEN_IDX \
    int idx_send = ei + rank_send * num_expert; \
    int idx_recv = ei + rank_recv * num_expert; \
    int gidx_send = ei * world_size + rank_send; \
    int gidx_recv = ei * world_size + rank_recv; \
    int idx_self = ei +      rank * num_expert;

void _compute_ptrs(long num_expert, long rank, long world_size, 
        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,
        int *local_ptr,
        int *global_ptr,
        int *local_global_ptr) {
    local_ptr[0] = global_ptr[0] = local_global_ptr[0] = 0;
    
    for (int i = 0; i < num_expert * world_size; ++i) {
        local_ptr[i + 1] = local_ptr[i] + local_expert_count[i];

        local_global_ptr[i + 1] = local_global_ptr[i];
        // if model fetched, add local tokens
        if (stored_models[i]){
            local_global_ptr[i + 1] += local_expert_count[i];
        }

        auto expert_idx = i % num_expert;
        auto worker_idx = i / num_expert;
        auto gp_idx = expert_idx * world_size + worker_idx;
        // if local model wasn't fetched, receive global tokens
        if (stored_models[rank * num_expert + expert_idx]) {
            global_ptr[gp_idx + 1] = 0;
        } else {
            global_ptr[gp_idx + 1] = global_expert_count[i];
        }
    }
    global_ptr[0] = 0;
    for (int i = 0; i < num_expert * world_size; ++i) {
        global_ptr[i + 1] += global_ptr[i];
    }
}

template<typename scalar_t>
void _compute_fn(py::function fn, c10::Device device,
        scalar_t* inp_buf, scalar_t* out_buf,
        int ei, long step, long offset, long micro_batch_size, long d_model) {
    auto options = torch::TensorOptions()
        .dtype(c10::CppTypeToScalarType<scalar_t>::value)
        .device(device)
        .requires_grad(true);
    auto inp = torch::from_blob(inp_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    auto oup = torch::from_blob(out_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    fn(inp, oup, step);
}


template<typename scalar_t>
void fmoe_cuda_fused_forward_impl(
        py::function forward_fn,
        c10::Device device,

        const scalar_t* input_buf,
        scalar_t* global_input_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,

        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,

        long d_model,
        long num_expert, long rank, long world_size,
        long pipeline_gran, CudaStreamManager* smgr) {

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker
    _compute_ptrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);

    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    for (long step = 0; step < n_groups; ++step) {
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                _exchange_with(input_buf + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_input_buf + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(1), input_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size + 
                (from_base + pipeline_gran)] - offset;
            
            _compute_fn(forward_fn, device,
                    global_input_buf, global_output_buf,
                    ei, step, offset, micro_batch_size, d_model);
        }
        auto stream = c10::cuda::getCurrentCUDAStream().stream();
        cudaEventRecord(output_ready[step], stream);
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(0), output_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                _exchange_with(global_output_buf + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        output_buf + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    /* TODO: Shadowing support
    int offset = global_ptr[world_size * num_expert];
    for (int j = 0; j < world_size; j++) {
        
        for (int i = 0; i < num_expert; i++) {
            int idx = j * num_expert + i;
            if (!stored_models[idx])
                continue;
            weight1 = params[j][0][0].data_ptr<scalar_t>();
            weight2 = params[j][0][last].data_ptr<scalar_t>();

            auto stream = 2 + (idx % (SMGR_N_STREAMS- 2));

            _compute_mlp_forward(
                input_buf + local_ptr[idx] * d_model, weight1, weight2,
                middle_buf + (offset + local_global_ptr[idx]) * d_hidden, output_buf + local_ptr[idx] * d_model,
                i,
                0, local_expert_count[idx],
                d_model, d_hidden,
                smgr->stream(stream), smgr->handle(stream));

        }
    }*/


    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}


template<typename scalar_t>
void fmoe_cuda_fused_backward_impl(
        py::function backward_fn,
        c10::Device device,

        scalar_t* grad_out,
        scalar_t* global_grad_out,
        scalar_t* global_grad_in,
        scalar_t* grad_in,

        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,
        long d_model,
        long num_expert, long rank, long world_size,
        long pipeline_gran, CudaStreamManager* smgr) {

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker

    _compute_ptrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);
   
    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    for (long step = 0; step < n_groups; ++step) {
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                _exchange_with(grad_out + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_grad_out + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(1), input_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size + 
                (from_base + pipeline_gran)] - offset;

            _compute_fn(backward_fn, device,
                    global_grad_out, global_grad_in,
                    ei, step, offset, micro_batch_size, d_model);
        }
        // TODO: get pytorch's compute stream
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(0), output_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                _exchange_with(global_grad_in + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        grad_in + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    checkCudaErrors(cudaGetLastError());

    /* TODO: Shadowing support
    int offset = global_ptr[world_size * num_expert];
    for (int j = 0; j < world_size; j++) {
        
        for (int i = 0; i < num_expert; i++) {
            int idx = j * num_expert + i;
            if (!stored_models[idx])
                continue;
            
            weight1 = params[j][0][0].data_ptr<scalar_t>();
            weight2 = params[j][0][last].data_ptr<scalar_t>();    
            grad_weight1 = params[j][0][0].mutable_grad().data_ptr<scalar_t>();
            grad_weight2 = params[j][0][last].mutable_grad().data_ptr<scalar_t>();
            
            auto stream = 2 + (idx % (SMGR_N_STREAMS- 2));

            _compute_mlp_backward(
                original_input_buf + local_ptr[idx] * d_model, weight1, weight2,
                middle_buf + (offset + local_global_ptr[idx]) * d_hidden, output_buf, grad_out + local_ptr[idx] * d_model,
                grad_middle + (offset + local_global_ptr[idx]) * d_hidden, grad_weight1, grad_weight2, grad_in + local_ptr[idx] * d_model,
                i,
                0, local_expert_count[idx],
                d_model, d_hidden, 0, // we never consider it to be the first since it's already initialized to zero and we are lazy
                smgr->stream(stream), smgr->handle(stream));

        }
    }
    */


    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}

#endif  // SMART_SCHEDULE_H
