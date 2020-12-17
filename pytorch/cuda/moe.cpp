#include <torch/extension.h>
#include <torch/torch.h>

#include <cstdio>
#include <iostream>
#include <vector>

// CUDA runtime                                                                                                           
#include <cuda_runtime.h>                                                                                                 
#include <cublas_v2.h>                                                                                                    
                                                                                                                            
// CUDA and CUBLAS functions                                                                                              
//#include <helper_functions.h>                                                                                             
#include <helper_cuda.h> 


const int num_stream=512;


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
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
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
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
}

inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half           *alpha,
                                  const __half           *Aarray[], int lda,
                                  const __half           *Barray[], int ldb,
                                  const __half           *beta,
                                  _half           *Carray[], int ldc,
                                  int batchCount)
{
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
}

template <typename scalar_t>
void moe_cuda_forward_impl(
        const scalar_t* input,
        const size_t* gate,
        const scalar_t* weight,
        scalar_t* output,
        size_t batch_size,
        size_t top_k,
        size_t in_feat,
        size_t out_feat) {
    

    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    // setup Aarray, Barray and Carray
	std::vector<scalar_t*> aptrs, bptrs, cptrs;
	scalar_t **ptrs;
	checkCudaErrors(cudaMalloc(&ptrs, batch_size * sizeof(scalar_t*) * top_k * 3));
	for (size_t i=0; i<batch_size; ++i) {
        for (size_t k=0; k<top_k; ++k) {
            aptrs.push_back(input + in_feat * i);
            bptrs.push_back(weight + out_feat * in_feat * gate[i * top_k + k]);
            cptrs.push_back(output + out_feat * (i * top_k + k));
        }
	}
	checkCudaErrors(cudaMemcpy(ptrs, aptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ptrs + batch_size * top_k, bptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ptrs + batch_size * top_k *  2, cptrs.data(), batch_size * sizeof(scalar_t*) * top_k, cudaMemcpyHostToDevice));

    scalar_t alpha = 1, beta = 0;
	checkCudaErrors(cublasXgemmBatched(handle, 
			CUBLAS_OP_N,
			CUBLAS_OP_T,
			1, out_feat, in_feat,
			&alpha,
			ptrs, 1,
			ptrs + batch_size * top_k, out_feat,
			&beta,
			ptrs + batch_size * top_k * 2, 1,
			batch_size));
	cudaStreamSynchronize(st);
}


void moe_cuda_forward(
        torch::Tensor input, // [B x D_model]
        torch::Tensor gate,  // [B x K]
        torch::Tensor weight, // [N x D_ffn x D_model]
        ) {
    /*
        The bias term should have been merged into weight. Note the following fact that 
        Wx+b = [W b] [x]
                     [1]  
    */
    const auto batch_size = input.size(0);
    const auto top_k = gate.size(1);
    const auto num_expert = weight.size(0);
    const auto out_feat = weight.size(1);
    const auto in_feat = weight.size(2);
    
    printf("b=%d, expert=%d, in_feat (d_model)=%d, out_feat (d_ffn)=%d, topk=%d\n", batch_size, num_expert, d_model, d_ffn, top_k);
    auto output = input.new_zeros({batch_size, top_k, out_feat});

    AT_DISPATCH_FLOATING_TYPES(input.type(), "moe_cuda_forward", ([&] {
        moe_cuda_forward_impl<scalar_t>(
            input.data_ptr<scalar_t>(),
            gate.data_ptr<size_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            top_k,
            in_feat,
            out_feat
        );
    }));

    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    
    cudaStream_t stream[num_stream];
    for (size_t i=0; i<num_stream; ++i) {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    
    size_t s;
    for (size_t i=0; i<batch_size; ++i) {
        for (size_t j=0; j<num_expert; ++j) {
            s = (i * num_expert + j) % num_stream;
            // printf("i=%d j=%d goes to stream %d\n", i, j, s);
            checkCudaErrors(cublasSetStream(handle, stream[s]));
            if (input.scalar_type() == torch::ScalarType::Float) {
                float alpha = 1.0;
                float beta = 0.0;
                checkCudaErrors(cublasSgemm(handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N,
                    1, // m
                    d_ffn, // n
                    d_model, // k
                    &alpha,
                    input[i].data_ptr<float>(),
                    1,
                    weight.index(gate[i][j]).data_ptr<float>(),
                    d_model,
                    &beta,
                    output[i][j].data_ptr<float>(),
                    1));
            } else {
                printf("only support float!!!\n");
            }
        }
    }
    // checkCudaErrors(cudaDeviceSynchronize());
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / batch_size / num_expert;
    double flopsPerMatrixMul = 2.0 * (double)d_model * (double)d_ffn;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

    // std::cout << output << std::endl;
    
    for (size_t i=0; i<num_stream; ++i) {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
    checkCudaErrors(cublasDestroy(handle));
}


// std::vector<torch::Tensor> 
void moe_cuda_forward_v1(
        torch::Tensor input, // [B x D_model]
        torch::Tensor gate,  // [B x N]
        torch::Tensor weight, // [N x D_model x D_ffn]
        torch::Tensor bias // [N x D_ffn]
        ) {
    const auto batch_size = input.size(0);
    const auto num_expert = gate.size(1);
    const auto d_model = weight.size(1);
    const auto d_ffn = weight.size(2);
    printf("b=%d, expert=%d, d_model=%d, d_ffn=%d\n", batch_size, num_expert, d_model, d_ffn);
    auto output = input.new_zeros({batch_size, num_expert, d_ffn});
    

    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    
    cudaStream_t stream[num_stream];
    for (size_t i=0; i<num_stream; ++i) {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    
    size_t s;
    for (size_t i=0; i<batch_size; ++i) {
        for (size_t j=0; j<num_expert; ++j) {
            s = (i * num_expert + j) % num_stream;
            // printf("i=%d j=%d goes to stream %d\n", i, j, s);
            checkCudaErrors(cublasSetStream(handle, stream[s]));
            if (input.scalar_type() == torch::ScalarType::Float) {
                float alpha = 1.0;
                float beta = 0.0;
                checkCudaErrors(cublasSgemm(handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N,
                    1, // m
                    d_ffn, // n
                    d_model, // k
                    &alpha,
                    input[i].data_ptr<float>(),
                    1,
                    weight.index(gate[i][j]).data_ptr<float>(),
                    d_model,
                    &beta,
                    output[i][j].data_ptr<float>(),
                    1));
            } else {
                printf("only support float!!!\n");
            }
        }
    }
    // checkCudaErrors(cudaDeviceSynchronize());
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / batch_size / num_expert;
    double flopsPerMatrixMul = 2.0 * (double)d_model * (double)d_ffn;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
            flopsPerMatrixMul);

    // std::cout << output << std::endl;
    
    for (size_t i=0; i<num_stream; ++i) {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
    checkCudaErrors(cublasDestroy(handle));
}


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int main() {
    int device=2;
    torch::Tensor input = torch::randn({2048, 512}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    torch::Tensor gate = torch::zeros({2048, 2}, torch::dtype(torch::kInt64));
    torch::Tensor weight = torch::randn({2, 512, 2048}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    torch::Tensor bias = torch::randn({2, 2048}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    checkCudaErrors(cudaSetDevice(device));
    moe_cuda_forward_v1(input, gate, weight, bias);
}