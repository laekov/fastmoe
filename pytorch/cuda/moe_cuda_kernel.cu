#include <cstdio>
#include <iostream>
#include <vector>

// CUDA runtime
#include <cuda.h>                                                                                             
#include <cuda_runtime.h>                                                                                                 
#include <cublas_v2.h>                                                                                                    
                                                                                                                            
// CUDA and CUBLAS functions                                                                                              
//#include <helper_functions.h>                                                                                             
#include <helper_cuda.h> 


typedef float data_t;
size_t batch_size = 4096;
size_t top_k = 2;
size_t num_expert = 128;
size_t in_feat = 512;
size_t out_feat = 2048;

#define CEIL(_x_,_y_) (((_x_)-1)/(_y_)+1)

template <typename scalar_t>
__global__
void generate_ptr_offset_kernel(size_t n, const scalar_t* base, size_t stride, const size_t* offset, const scalar_t** ptrs) {
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
                                  int batchCount)
{
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
                                  int batchCount)
{
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
                                  int batchCount)
{
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

template <typename scalar_t>
void moe_cuda_forward_impl(
        const scalar_t* input,
        const size_t* gate,
        const scalar_t* weight,
        scalar_t* output,
        const size_t batch_size,
        const size_t top_k,
        const size_t in_feat,
        const size_t out_feat) {
    

    cublasHandle_t handle;
	cudaStream_t st;
	cudaStreamCreate(&st);
    checkCudaErrors(cublasCreate(&handle));
    checkCudaErrors(cublasSetStream(handle, st));

    // setup Aarray, Barray and Carray
	std::vector<const scalar_t*> aptrs;
    std::vector<scalar_t*> cptrs;
	
    const scalar_t **Aarray;
    const scalar_t **Barray;
    scalar_t **Carray;
	checkCudaErrors(cudaMalloc(&Aarray, batch_size * sizeof(const scalar_t*) * top_k));
    checkCudaErrors(cudaMalloc(&Barray, batch_size * sizeof(const scalar_t*) * top_k));
    checkCudaErrors(cudaMalloc(&Carray, batch_size * sizeof(scalar_t*) * top_k));

	for (size_t i=0; i<batch_size; ++i) {
        for (size_t k=0; k<top_k; ++k) {
            aptrs.push_back(input + in_feat * i);
            // bptrs.push_back(weight + out_feat * in_feat * gate[i * top_k + k]);
            cptrs.push_back(output + out_feat * (i * top_k + k));
        }
	}
	checkCudaErrors(cudaMemcpy(Aarray, aptrs.data(), batch_size * sizeof(const scalar_t*) * top_k, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(ptrs + batch_size * top_k, bptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Carray, cptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));

	dim3 griddim(CEIL(batch_size * top_k, 256));
	dim3 blockdim(256);
    generate_ptr_offset_kernel<<<griddim, blockdim, 0, st>>>(batch_size * top_k, weight, out_feat * in_feat, gate, Barray);

    scalar_t alpha = 1, beta = 0;
	checkCudaErrors(cublasXgemmBatched(handle, 
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			1, out_feat, in_feat,
			&alpha,
			Aarray, 1,
			Barray, out_feat,
			&beta,
			Carray, 1,
			batch_size));

	checkCudaErrors(cudaStreamSynchronize(st));
    checkCudaErrors(cudaStreamDestroy(st));
    checkCudaErrors(cublasDestroy(handle));
}


int main() {
	data_t *input, *weight;
	data_t *output;
	size_t *gate;

	checkCudaErrors(cudaMalloc(&input, batch_size * in_feat * sizeof(data_t)));
	checkCudaErrors(cudaMalloc(&weight, num_expert * in_feat * out_feat * sizeof(data_t)));	
	checkCudaErrors(cudaMalloc(&output, batch_size * top_k * out_feat * sizeof(data_t)));
	checkCudaErrors(cudaMalloc(&gate, batch_size * top_k * sizeof(size_t)));

	moe_cuda_forward_impl<data_t>(input, gate, weight, output, batch_size, top_k, in_feat, out_feat);
}