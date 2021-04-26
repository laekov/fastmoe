#include "stream_manager.h"
#include "utils/cublas_wrapper.h"


template <typename scalar_t>
void fmoe_cuda_forward_impl(
        const scalar_t* input_buf,
        const scalar_t* weight,
        const long* expert_count,
        scalar_t* output_buf,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert,
        CudaStreamManager* smgr) {
    scalar_t alpha = 1, beta = 0; 

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
void fmoe_cuda_backward_impl(
        const scalar_t* grad_output_buf,
        const scalar_t* input_buf,
        const scalar_t* weight,
        const long* expert_count,
        scalar_t* grad_input_buf,
        scalar_t* grad_weight,
        const size_t batch_size,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert,
        CudaStreamManager* smgr) {
    scalar_t alpha = 1, beta = 0;

    for (int i = 0, ptr = 0; i < num_expert; ++i) {
        if (expert_count[i] == 0) {
            cudaMemset(grad_weight + i * in_feat * out_feat, 0, 
                    sizeof(scalar_t) * in_feat * out_feat);
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

        ptr += expert_count[i];
    }
    smgr->sync(num_expert);
}
