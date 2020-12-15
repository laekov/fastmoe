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


const int num_stream=16;

// std::vector<torch::Tensor> 
void moe_cuda_forward(
        torch::Tensor input, // [B x D_model]
        torch::Tensor gate,  // [B x N]
        torch::Tensor weight, // [N x D_model x D_ffn]
        torch::Tensor bias // [N x D_ffn]
        ) {
    const auto batch_size = input.size(0);
    const auto num_expert = gate.size(1);
    const auto d_model = weight.size(1);
    const auto d_ffn = weight.size(2);
    auto output = input.new_zeros({batch_size, num_expert, d_ffn});
    std::cout << output << std::endl;
    

    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));
    
    cudaStream_t stream[num_stream];
    for (size_t i=0; i<num_stream; ++i) {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    
    size_t s;
    for (size_t i=0; i<batch_size; ++i) {
        for (size_t j=0; j<num_expert; ++j) {
            s = (i * num_expert + j) % num_stream;
            printf("i=%d j=%d goes to stream %d\n", i, j, s);
            cublasSetStream(handle, stream[s]);
            if (input.scalar_type() == torch::ScalarType::Float) {
                float alpha = 1.0;
                float beta = 0.0;
                std::cout << input[i] << std::endl;
                std::cout << weight.index(gate[i][j]) << std::endl;
                std::cout << output[i][j] << std::endl;
                cublasSgemm(handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N,
                    1, // m
                    d_ffn, // n
                    d_model, // k
                    &alpha,
                    input.data_ptr<float>() + i * d_model,
                    // input[i].data_ptr<float>(),
                    1,
                    weight.index(gate[i][j]).data_ptr<float>(),
                    d_model,
                    &beta,
                    output.data_ptr<float>() + i * num_expert * d_ffn + j * d_ffn,
                    1);
            } else {
                printf("only support float!!!\n");
            }
        }
    }
    cudaDeviceSynchronize();
    printf("synchronized\n");
    

    
    for (size_t i=0; i<num_stream; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    std::cout << output << std::endl;
    
    cublasDestroy(handle);
}


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int main() {
    int device=2;
    torch::Tensor input = torch::randn({2, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    torch::Tensor gate = torch::zeros({2, 1}, torch::dtype(torch::kInt64).device(torch::kCUDA, device));
    torch::Tensor weight = torch::randn({2, 4, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    torch::Tensor bias = torch::randn({2, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    std::cout << input << std::endl;
    moe_cuda_forward(input, gate, weight, bias);
}