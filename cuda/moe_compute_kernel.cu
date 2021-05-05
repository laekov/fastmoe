#include "moe_cuda_kernel.h"

#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAGuard.h>

#include "timer.hh"

#include "cublas_wrapper.h"
#include "cuda_stream_manager.h"

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)

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


/*
    This function is to be called with one block per each column
*/
template <typename scalar_t>
__global__ 
void column_reduce(const scalar_t * matrix, scalar_t * result, 
    int m /* lines */, int n /* columns*/) {
    
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    extern __shared__ unsigned char my_smem[];
    scalar_t *sdata = reinterpret_cast<scalar_t *>(my_smem);

    // normal tid
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    
    // transposed tid for shared memory
    int new_tid = threadIdx.y + threadIdx.x * blockDim.y;

    // true x value in the matrix
    int real_x = threadIdx.x + blockDim.x * blockIdx.x;
    
    int i = real_x + n * threadIdx.y;
    const int it = n*blockDim.y;
    int offset = it;
    float accumulator = 0;

    if (threadIdx.y < m && real_x < n) {
        // store all the values from this column in a warped way
        accumulator = matrix[i];
        while (i + offset < n*m) {
            accumulator += matrix[i + offset];
            offset += it;
        }
    }

    // save column reduction data in a transposed way
    sdata[new_tid] = accumulator;
    __syncthreads();

    for (size_t t= 16; t > 0; t>>=1) {
        if (tid < 32 * 32 - 16)
            sdata[tid] += sdata[tid + t];
        __syncthreads();
    }
    
    if (threadIdx.y == 0 && real_x < n) 
        result[real_x] = sdata[new_tid];
    
}


void moe_cuda_expert_count_impl(
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
void moe_cuda_local_scatter_impl(
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
void moe_cuda_local_gather_impl(
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

template <typename scalar_t>
void moe_cuda_forward_impl(
        const scalar_t* input_buf,
        const scalar_t* weight,
        const long* expert_count,
        scalar_t* output_buf,
        const bool has_bias,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert,
        CudaStreamManager* smgr) {
    scalar_t alpha = 1, beta = has_bias ? 1 : 0; 

    for (int i = 0, ptr = 0; i < num_expert; ++i) {
        if (expert_count[i] == 0) {
            continue;
        }
        // Use T(B) x T(A) = T(C) to produce row-major C
        checkCudaErrors(cublasXgemm(
                smgr->handle(i),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                out_feat, expert_count[i], in_feat,
                &alpha,
                weight + i * in_feat * out_feat, in_feat,
                input_buf + ptr * in_feat, in_feat,
                &beta,
                output_buf + out_feat * ptr, out_feat
                ));

        ptr += expert_count[i];
    }
    smgr->sync(num_expert);
}

template <typename scalar_t>
void moe_cuda_backward_impl(
        const scalar_t* grad_output_buf,
        const scalar_t* input_buf,
        const scalar_t* weight,
        const long* expert_count,
        scalar_t* grad_input_buf,
        scalar_t* grad_weight,
        scalar_t* grad_bias,
        const bool has_bias,
        const size_t batch_size,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert,
        CudaStreamManager* smgr) {
    scalar_t alpha = 1, beta = 0;

    // bias
    dim3 block_threads(32, 32);
    dim3 grid_threads(out_feat / 32 + (out_feat % 32 ? 1 : 0), 1);
    

    for (int i = 0, ptr = 0; i < num_expert; ++i) {
        if (expert_count[i] == 0) {
            cudaMemset(grad_weight + i * in_feat * out_feat, 0, 
                    sizeof(scalar_t) * in_feat * out_feat);
            cudaMemset(grad_bias + i * out_feat, 0, sizeof(scalar_t) * out_feat);
            continue;
        }
        // Use T(B) x T(A) = T(C) to produce row-major C

        // Backward input: g_i = w @ g_o
        checkCudaErrors(cublasXgemm(
                smgr->handle(i),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                in_feat, expert_count[i], out_feat,
                &alpha,
                weight + i * in_feat * out_feat, in_feat,
                grad_output_buf + ptr * out_feat, out_feat,
                &beta,
                grad_input_buf + in_feat * ptr, in_feat
                ));

        // Backward weight: g_w = i @ g_o
        checkCudaErrors(cublasXgemm(
                smgr->handle(i),
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                in_feat, out_feat, expert_count[i],
                &alpha,
                input_buf + in_feat * ptr, in_feat,
                grad_output_buf + ptr * out_feat, out_feat,
                &beta,
                grad_weight + i * in_feat * out_feat, in_feat
                ));
        
        if (has_bias) {
            column_reduce
            <<<grid_threads, block_threads, sizeof(scalar_t)*1024, smgr->stream(0)>>>
            (
                grad_output_buf + ptr * out_feat,
                grad_bias + i * out_feat,
                expert_count[i],
                out_feat
            );
        }

        ptr += expert_count[i];
    }
    smgr->sync(num_expert);
}


std::vector<torch::Tensor> moe_cuda_expert_count(
        torch::Tensor gate, 
        size_t num_expert) {
    const auto batch_size = gate.size(0);

    auto ec_options = torch::TensorOptions().dtype(torch::kInt32);
    auto expert_count = torch::empty(num_expert, ec_options);

    auto pos_options = torch::TensorOptions()
        .device(gate.device())
        .dtype(torch::kInt32);
    auto pos = torch::empty(batch_size, pos_options);
    moe_cuda_expert_count_impl(
            gate.data_ptr<int>(),
            expert_count.data_ptr<int>(),
            pos.data_ptr<int>(),
            num_expert,
            batch_size);

    return {expert_count, pos};
}

std::vector<torch::Tensor> moe_cuda_local_scatter(
    torch::Tensor input,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(input.device().index());
    const auto batch_size = pos.size(0);
    const auto in_feat = input.size(1);

    auto opt = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto input_buf = torch::empty({batch_size, in_feat}, opt);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "moe_local_scatter_cuda", 
            ([&] {
        moe_cuda_local_scatter_impl<scalar_t>(
            input.data_ptr<scalar_t>(),
            pos.data_ptr<long>(),
            input_buf.data_ptr<scalar_t>(),
            batch_size,
            in_feat,
            smgr);
    }));
    return {input_buf,};
}

std::vector<torch::Tensor> moe_cuda_local_gather(
    torch::Tensor output_buf,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(output_buf.device().index());
    const auto batch_size = pos.size(0);
    const auto out_feat = output_buf.size(1);

    auto opt = torch::TensorOptions()
        .dtype(output_buf.dtype())
        .device(output_buf.device());
    auto output = torch::empty({batch_size, out_feat}, opt);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(), "moe_local_gather_cuda", 
            ([&] {
        moe_cuda_local_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            pos.data_ptr<long>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            out_feat,
            smgr);
    }));
    return {output,};
}

std::vector<torch::Tensor> moe_cuda_forward(
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        ) {
    auto smgr = getCudaStreamManager(input_buf.device().index());
    const auto batch_size = input_buf.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);
            
#ifdef MOE_DEBUG
    printf("[forward] expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld\n", 
            num_expert, in_feat, out_feat);
#endif

    torch::Tensor output;
    
    if (bias.has_value()) {
        output = bias.value().repeat_interleave(expert_count.to(bias.value().device()), 0);
    } else{
        auto out_options = torch::TensorOptions()
            .device(input_buf.device())
            .dtype(input_buf.dtype());
        output = torch::empty({batch_size, out_feat}, out_options);
    }
        
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), "moe_forward_cuda", 
            ([&] {
        moe_cuda_forward_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            expert_count.data_ptr<long>(),
            output.data_ptr<scalar_t>(),
            bias.has_value(),
            in_feat,
            out_feat,
            num_expert,
            smgr
        );
    }));
    
    return {output, };           
}

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output_buf, 	// [batch_size x out_feat]
    torch::Tensor input_buf, 		// [batch_size x out_feat]
    torch::Tensor expert_count,
    torch::Tensor weight, 			// [num_expert x out_feat x in_feat]
    at::optional<torch::Tensor> bias
) {
    auto smgr = getCudaStreamManager(input_buf.device().index());
    const auto batch_size = input_buf.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);

#ifdef MOE_DEBUG
    printf("[backward] b=%ld, expert=%ld, in_feat (d_model)=%ld, "
            "out_feat (d_ffn)=%ld\n",
            batch_size, num_expert, in_feat, out_feat);
#endif

    auto grad_input_buf = grad_output_buf.new_empty({batch_size, in_feat}); 
    auto grad_weight = grad_output_buf.new_empty({num_expert, out_feat, in_feat});
    auto grad_bias = grad_output_buf.new_empty({num_expert, out_feat});

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), "moe_cuda_backward", ([&] {
        moe_cuda_backward_impl<scalar_t>(
            grad_output_buf.data_ptr<scalar_t>(),
            input_buf.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            expert_count.data_ptr<long>(),
            grad_input_buf.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            grad_bias.data_ptr<scalar_t>(),
            bias.has_value(),
            batch_size,
            in_feat,
            out_feat,
            num_expert,
            smgr
        );
    }));

    return {grad_input_buf, grad_weight, grad_bias};
}
