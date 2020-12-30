#include <torch/extension.h>
#include <torch/torch.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>                                                                                          
#include <helper_cuda.h> 

// #include "timer.hh"

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)


class Helper {
public:
    Helper(const size_t num_expert_) : num_expert(num_expert_) {
        streams = new cudaStream_t[num_expert];
        checkCudaErrors(cublasCreate(&handle));
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamCreate(streams+i));
        }
    }
    ~Helper() {
        for (size_t i=0; i<num_expert; ++i) {
            checkCudaErrors(cudaStreamDestroy(*(streams+i)));
        }
        checkCudaErrors(cublasDestroy(handle));
    }
    const size_t num_expert;
    cublasHandle_t handle;
    cudaStream_t* streams;
}; 

Helper* helper = NULL;
Helper* getHelper(const size_t num_expert) { 
    if (!helper) {
        helper = new Helper(num_expert);        
    }
    assert(helper->num_expert == num_expert);
    return helper;
}


template <typename scalar_t>
__global__
void generate_ptr_offset_kernel(size_t n, const scalar_t* base, size_t stride, const int* offset, const scalar_t** ptrs) {
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		ptrs[idx] = base + stride * offset[idx];
	}
}


inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float           *alpha,
                                  const float           *Aarray[], int lda,
                                  const float           *Barray[], int ldb,
                                  const float           *beta,
                                  float           *Carray[], int ldc,
                                  int batchCount) {
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double           *alpha,
                                  const double           *Aarray[], int lda,
                                  const double           *Barray[], int ldb,
                                  const double           *beta,
                                  double           *Carray[], int ldc,
                                  int batchCount) {
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half           *alpha,
                                  const __half           *Aarray[], int lda,
                                  const __half           *Barray[], int ldb,
                                  const __half           *beta,
                                  __half           *Carray[], int ldc,
                                  int batchCount) {
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}


inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const float           *alpha,
                                const float           *A, int lda,
                                const float           *B, int ldb,
                                const float           *beta,
                                float           *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const double          *alpha,
                                const double          *A, int lda,
                                const double          *B, int ldb,
                                const double          *beta,
                                double          *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                cublasOperation_t transa, cublasOperation_t transb,
                                int m, int n, int k,
                                const __half *alpha,
                                const __half *A, int lda,
                                const __half *B, int ldb,
                                const __half *beta,
                                __half *C, int ldc) {
    return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename scalar_t>
void moe_cuda_forward_impl(
        const scalar_t* input,
        const int* gate,
        const scalar_t* weight,
        scalar_t* output,
        const size_t batch_size,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert,
        cublasOperation_t transb) {

    Helper* h = getHelper(num_expert);

    checkCudaErrors(cublasSetStream(h->handle, *(h->streams)));

    // setup Aarray, Barray and Carray
	std::vector<const scalar_t*> aptrs;
    std::vector<scalar_t*> cptrs;
	
    const scalar_t **Aarray;
    const scalar_t **Barray;
    scalar_t **Carray;
	checkCudaErrors(cudaMalloc(&Aarray, batch_size * sizeof(const scalar_t*)));
    checkCudaErrors(cudaMalloc(&Barray, batch_size * sizeof(const scalar_t*)));
    checkCudaErrors(cudaMalloc(&Carray, batch_size * sizeof(scalar_t*)));

	for (size_t i=0; i<batch_size; ++i) {
        aptrs.push_back(input + in_feat * i);
        cptrs.push_back(output + out_feat * i);
	}
	checkCudaErrors(cudaMemcpy(Aarray, aptrs.data(), batch_size * sizeof(const scalar_t*), cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(ptrs + batch_size * top_k, bptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Carray, cptrs.data(), batch_size * sizeof(scalar_t*), cudaMemcpyHostToDevice));

	dim3 griddim(CEIL(batch_size, 256));
	dim3 blockdim(256);
    generate_ptr_offset_kernel<<<griddim, blockdim, 0, *(h->streams)>>>(batch_size, weight, out_feat * in_feat, gate, Barray);

    scalar_t alpha = 1, beta = 0;
	checkCudaErrors(cublasXgemmBatched(h->handle, 
			CUBLAS_OP_N,
			transb,
			1, out_feat, in_feat,
			&alpha,
			Aarray, 1,
			Barray, (transb == CUBLAS_OP_T) ? out_feat : in_feat,
			&beta,
			Carray, 1,
			batch_size));

	checkCudaErrors(cudaStreamSynchronize(*(h->streams)));
}

template <typename scalar_t>
void moe_cuda_grad_weight(
        const scalar_t* input,
        const int* gate,
        const scalar_t* grad_output,
        scalar_t* grad_weight, // [num_expert x out_feat x in_feat]
        const size_t batch_size,
        const size_t in_feat,
        const size_t out_feat,
        const size_t num_expert) {

    Helper* h = getHelper(num_expert);
    
    int* gate_host = new int[batch_size];
    scalar_t alpha = 1, beta = 1;
    checkCudaErrors(cudaMemcpy(gate_host, gate, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t i=0; i<batch_size; ++i) {
        checkCudaErrors(cublasSetStream(h->handle, *(h->streams + gate_host[i])));
        checkCudaErrors(cublasXgemm(h->handle,
            CUBLAS_OP_N, 
            CUBLAS_OP_T,
            out_feat, 
            in_feat, 
            1,
            &alpha,
            grad_output + i * out_feat,
            out_feat,
            input + i * in_feat,
            in_feat,
            &beta,
            grad_weight + gate_host[i] * out_feat * in_feat,
            out_feat));
    }
    for (size_t i=0; i<num_expert; ++i) {
        checkCudaErrors(cudaStreamSynchronize(*(h->streams + i)));
    }
    delete[] gate_host;
}

std::vector<torch::Tensor> moe_cuda_forward(
        torch::Tensor input,
        torch::Tensor gate,
        torch::Tensor weight) {
    const auto batch_size = input.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);
            
    printf("[forward] b=%ld, expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld\n", batch_size, num_expert, in_feat, out_feat);
    auto output = input.new_zeros({batch_size, out_feat});
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_forward_cuda", ([&] {
                moe_cuda_forward_impl<scalar_t>(
                    input.data_ptr<scalar_t>(),
                    gate.data_ptr<int>(),
                    weight.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    in_feat,
                    out_feat,
                    num_expert,
                    CUBLAS_OP_T
                );
    }));
    
    return {output, };           
}

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output, // [batch_size x out_feat]
    torch::Tensor input, // [batch_size x out_feat]
    torch::Tensor gate,  // [batch_size]
    torch::Tensor weight // [num_expert x out_feat x in_feat]
) {
    const auto batch_size = input.size(0);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);
    printf("[backward] b=%ld, expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld\n", batch_size, num_expert, in_feat, out_feat);

    auto grad_input = grad_output.new_zeros({batch_size, in_feat});  // batch_size x in_feat
    auto grad_weight = grad_output.new_zeros({num_expert, out_feat, in_feat}); // num_expert x out_feat x in_feat

    // grad_input is easy to compute, exactly the same as forward
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_cuda_backward", ([&] {
        moe_cuda_forward_impl<scalar_t>(
            grad_output.data_ptr<scalar_t>(),
            gate.data_ptr<int>(),
            weight.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            batch_size,
            out_feat,
            in_feat,
            num_expert,
            CUBLAS_OP_N
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_cuda_backward", ([&] {
        moe_cuda_grad_weight<scalar_t>(
            input.data_ptr<scalar_t>(),
            gate.data_ptr<int>(),
            grad_output.data_ptr<scalar_t>(),
            grad_weight.data_ptr<scalar_t>(),
            batch_size,
            in_feat,
            out_feat,
            num_expert
        );
    }));

    return {grad_input, grad_weight};
}


/*
int main() {
    typedef float data_t;
    size_t batch_size = 4096;
    size_t top_k = 2;
    size_t num_expert = 128;
    size_t in_feat = 1024;
    size_t out_feat = 4096;
	data_t *input, *weight;
	data_t *output;
	size_t *gate;

	checkCudaErrors(cudaMalloc(&input, batch_size * in_feat * sizeof(data_t)));
	checkCudaErrors(cudaMalloc(&weight, num_expert * in_feat * out_feat * sizeof(data_t)));	
	checkCudaErrors(cudaMalloc(&output, batch_size * top_k * out_feat * sizeof(data_t)));
    checkCudaErrors(cudaMalloc(&gate, batch_size * top_k * sizeof(size_t)));
    
    size_t nt = 16;
    double tsum = 0, tmax = 0;

    size_t *gate_host = new size_t[batch_size * top_k];
    for (size_t i=0; i<batch_size * top_k; ++i) {
        gate_host[i] = rand() % num_expert;
    } 
    checkCudaErrors(cudaMemcpy(gate, gate_host, batch_size * top_k * sizeof(size_t), cudaMemcpyHostToDevice));

    moe_first_linear_cuda_forward<data_t>(input, gate, weight, output, batch_size, top_k, in_feat, out_feat);
    
    for (size_t i=0; i<nt; ++i) {
        timestamp(start);
		moe_first_linear_cuda_forward<data_t>(input, gate, weight, output, batch_size, top_k, in_feat, out_feat);
		timestamp(end);
		auto t = getDuration(start, end);
		tsum += t;
		if (t > tmax) tmax = t;
    }
    printf("Mean %.3lf us, max %.3lf us\n", tsum / nt * 1e6, tmax * 1e6);
	double tflops = (double)batch_size * top_k * in_feat * out_feat * nt * 2e-12 / tsum;
	printf("%.3lf TFLOPs\n", tflops);
}
*/