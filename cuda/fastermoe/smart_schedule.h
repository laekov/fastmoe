#ifndef SMART_SCHEDULE_H
#define SMART_SCHEDULE_H

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "../stream_manager.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 110010)
#define FMOE_SWE(__s__,__e__) cudaStreamWaitEvent(__s__,__e__,0)
#else
#define FMOE_SWE(__s__,__e__) cudaStreamWaitEvent(__s__,__e__)
#endif

template<typename scalar_t>
void exchangeWith(
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


void computePtrs(long num_expert, long rank, long world_size,
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
void computeFn(py::function fn, c10::Device device,
        scalar_t* inp_buf, scalar_t* out_buf,
        long expert_idx, long store_idx, long offset, long micro_batch_size, long d_model,
        CudaStreamManager* smgr) {
    if(micro_batch_size == 0) {
        return;
    }
    auto options = torch::TensorOptions()
        .dtype(c10::CppTypeToScalarType<scalar_t>::value)
        .device(device)
        .requires_grad(true);
    auto inp = torch::from_blob(inp_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    auto oup = torch::from_blob(out_buf + offset * d_model,
            {micro_batch_size, d_model}, options);
    smgr->use_default = true;
    fn(inp, oup, expert_idx, store_idx);
    smgr->use_default = false;
}


template<typename scalar_t>
void fmoe_cuda_fused_forward_impl(
        py::function forward_fn,
        py::function stash_fn,
        py::function pop_fn,
        c10::Device device,
        std::vector<torch::Tensor> params,

        scalar_t* input_buf,
        scalar_t* global_input_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,

        const long* local_expert_count,
        const long* global_expert_count,
        const bool* stored_models,

        long d_model,
        long num_expert, long rank, long world_size, long expert_size,
        long pipeline_gran, CudaStreamManager* smgr) {
    smgr->syncTorch();

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker
    computePtrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);

    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_torch_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
        cudaEventCreate(output_torch_ready + i);
    }

    // S_0 ... S_n
    for (long step = 0; step < n_groups; ++step) {
        for (long ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                exchangeWith(input_buf + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_input_buf + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(num_expert), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(num_expert));
    }

    // Broadcast shadowed experts
    cudaEvent_t evt_get, *evt_shadow;
    if (params.size() > 0) {
        evt_shadow = new cudaEvent_t[params.size()];
    }
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            if (i / num_expert == rank) {
                cudaEventCreate(&evt_get);
                cudaEventRecord(evt_get, smgr->stream(0));
                FMOE_SWE(smgr->stream(num_expert), evt_get);
                cudaEventDestroy(evt_get);
            }
            NCCL_SAFE_CALL(ncclBcast((void*)params[si].data_ptr<scalar_t>(),
                        expert_size * sizeof(scalar_t), ncclChar,
                        i / num_expert, smgr->ncclcomm, smgr->stream(num_expert)));
            cudaEventCreate(evt_shadow + si);
            cudaEventRecord(evt_shadow[si], smgr->stream(num_expert));
            ++si;
        }
    }

    // C_0 ... C_n
    for (long step = 0; step < n_groups; ++step) {
        FMOE_SWE(smgr->stream(0), input_ready[step]);
        FMOE_SWE(smgr->torchStream(), input_ready[step]);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size +
                (from_base + pipeline_gran)] - offset;
            computeFn(forward_fn, device,
                    global_input_buf, global_output_buf,
                    (long) ei, step * num_expert + ei, offset, micro_batch_size, d_model, smgr);
        }
        cudaEventRecord(output_ready[step], smgr->stream(0));
        cudaEventRecord(output_torch_ready[step], smgr->torchStream());
    }

    // Compute over shadowed experts
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            FMOE_SWE(smgr->stream(0), evt_shadow[si]);
            FMOE_SWE(smgr->torchStream(), evt_shadow[si]);
            stash_fn(params[si], si, 0); // always put shadowed expert at first, so expert_idx = 0
            long offset = local_ptr[i];
            long micro_batch_size = local_expert_count[i];
            computeFn(forward_fn, device,
                    input_buf, output_buf,
                    0, n_groups * num_expert + si, offset, micro_batch_size, d_model, smgr);
            ++si;
        }
    }
    pop_fn(0);

    // R_0 ... R_n
    for (long step = 0; step < n_groups; ++step) {
        FMOE_SWE(smgr->stream(num_expert), output_ready[step]);
        FMOE_SWE(smgr->stream(num_expert), output_torch_ready[step]);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                exchangeWith(global_output_buf + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        output_buf + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(num_expert), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }
    smgr->sync(num_expert + 1);

    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
        cudaEventDestroy(output_torch_ready[i]);
    }
    for (unsigned i = 0; i < params.size(); ++i) {
        cudaEventDestroy(evt_shadow[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
    delete [] output_torch_ready;
}


template<typename scalar_t>
void fmoe_cuda_fused_backward_impl(
        py::function backward_fn,
        py::function stash_fn,
        py::function pop_fn,
        py::function collect_fn,
        py::function set_grad_fn,
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
    smgr->syncTorch();

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker

    computePtrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);
    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_torch_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
        cudaEventCreate(output_torch_ready + i);
    }

    // S_0 ... S_n
    for (long step = 0; step < n_groups; ++step) {
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                exchangeWith(grad_out + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_grad_out + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(num_expert), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(num_expert));
    }

    // Shadowed experts backward and reduce
    cudaEvent_t *evt_reduce = new cudaEvent_t[num_expert];
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            stash_fn(si, 0);
            long offset = local_ptr[i];
            long micro_batch_size = local_expert_count[i];
            computeFn(backward_fn, device,
                    grad_out, grad_in,
                    0, n_groups * num_expert + si, offset, micro_batch_size, d_model, smgr);
            collect_fn(si, i / num_expert, 0);
            if (i / num_expert == rank) {
                cudaEventCreate(evt_reduce + i % num_expert);
                cudaEventRecord(evt_reduce[i % num_expert], smgr->stream(0));
            }
            ++si;
        }
    }
    pop_fn(0);

    // C_0 ... C_n
    for (long step = 0; step < n_groups; ++step) {
        FMOE_SWE(smgr->stream(0), input_ready[step]);
        FMOE_SWE(smgr->torchStream(), input_ready[step]);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size +
                (from_base + pipeline_gran)] - offset;

            computeFn(backward_fn, device,
                    global_grad_out, global_grad_in,
                    (long) ei, step * num_expert + ei, offset, micro_batch_size, d_model, smgr);
        }
        cudaEventRecord(output_ready[step], smgr->stream(0));
        cudaEventRecord(output_torch_ready[step], smgr->torchStream());
    }

    // Collect gradients for shadowed experts
    for (long i = 0, si = 0; i < world_size * num_expert; ++i) {
        if (stored_models[i]) {
            if (i / num_expert == rank) {
                FMOE_SWE(smgr->torchStream(), evt_reduce[i % num_expert]);
                set_grad_fn(si, i % num_expert);
            }
            ++si;
        }
    }

    // R_0 ... R_n
    for (long step = 0; step < n_groups; ++step) {
        FMOE_SWE(smgr->stream(num_expert), output_ready[step]);
        FMOE_SWE(smgr->stream(num_expert), output_torch_ready[step]);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                exchangeWith(global_grad_in + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        grad_in + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(num_expert), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    smgr->sync(num_expert + 1);
    checkCudaErrors(cudaGetLastError());

    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
        cudaEventDestroy(output_torch_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
    delete [] output_torch_ready;
    for (long i = 0; i < num_expert; ++i) {
        if (stored_models[i + rank * num_expert]) {
            cudaEventDestroy(evt_reduce[i]);
        }
    }
    delete [] evt_reduce;
}

#endif  // SMART_SCHEDULE_H
