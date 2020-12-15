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
//#include <helper_cuda.h> 


const int num_stream=1024;

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

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream[num_stream];
    for (size_t i=0; i<num_stream; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    
    size_t s;
    for (size_t i=0; i<batch_size; ++i) {
        for (size_t j=0; j<num_expert; ++j) {
            s = (i * num_expert + j) % num_stream;
            printf("i=%d j=%d goes to stream %d\n", i, j, s);
            cublasSetStream(handle, stream[s]);
            if (input.scalar_type() == torch::ScalarType::Double) {
                double alpha = 1.0;
                double beta = 0.0;
                cublasDgemm(handle, 
                    CUBLAS_OP_N, 
                    CUBLAS_OP_N,
                    1,
                    d_ffn,
                    d_model,
                    &alpha,
                    input[i].data_ptr<double>(),
                    1,
                    weight.index(gate[i][j]).data_ptr<double>(),
                    d_model,
                    &beta,
                    output[i][j].data_ptr<double>(),
                    1);
            } else {
                printf("only support double!!!\n");
            }
            
        }
    }

    for (size_t i=0; i<num_stream; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    cublasDestroy(handle);
}


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


int main() {
    torch::Tensor input = torch::randn({2, 4}, torch::dtype(torch::kFloat64).device(torch::kCUDA, 3));
    torch::Tensor gate = torch::ones({2, 1}, torch::dtype(torch::kInt64).device(torch::kCUDA, 3));
    torch::Tensor weight = torch::randn({2, 4, 4}, torch::dtype(torch::kFloat64).device(torch::kCUDA, 3));
    torch::Tensor bias = torch::randn({2, 4}, torch::dtype(torch::kFloat64).device(torch::kCUDA, 3));
    std::cout << input << std::endl;
    moe_cuda_forward(input, gate, weight, bias);
}