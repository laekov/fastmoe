#include "stream_manager.h"
#include "utils/cublas_wrapper.h"


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

template <typename scalar_t>
void fmoe_cuda_linear_forward_impl(
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
void fmoe_cuda_linear_backward_impl(
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
            <<<grid_threads, block_threads, sizeof(scalar_t)*1024, smgr->stream(i)>>>
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

