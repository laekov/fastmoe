#include <torch/extension.h>
#include <torch/torch.h>

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


std::vector<torch::Tensor> moe_cuda_forward(
        torch::Tensor input, // [B x D_model]
        torch::Tensor gate,  // [B x K]
        torch::Tensor weight // [N x D_ffn x D_model]
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
    
    printf("b=%d, expert=%d, in_feat (d_model)=%d, out_feat (d_ffn)=%d, topk=%d\n", batch_size, num_expert, in_feat, out_feat, top_k);
    auto output = input.new_zeros({batch_size, top_k, out_feat});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_cuda_forward", ([&] {
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
    return {output, };
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
    checkCudaErrors(cudaSetDevice(device));
    moe_cuda_forward(input, gate, weight);
}