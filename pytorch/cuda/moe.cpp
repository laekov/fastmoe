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

template <typename scalar_t>
void moe_first_linear_cuda_forward(
        const scalar_t* input,
        const size_t* gate,
        const scalar_t* weight,
        scalar_t* output,
        const size_t batch_size,
        const size_t top_k,
        const size_t in_feat,
        const size_t out_feat);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> moe_first_linear_forward(
        torch::Tensor input, // [B x D_model]
        torch::Tensor gate,  // [B x K]
        torch::Tensor weight // [N x D_ffn x D_model]
        ) {
    CHECK_INPUT(input);
    CHECK_INPUT(gate);
    CHECK_INPUT(weight);
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
    
    printf("b=%ld, expert=%ld, in_feat (d_model)=%ld, out_feat (d_ffn)=%ld, topk=%ld\n", batch_size, num_expert, in_feat, out_feat, top_k);
    auto output = input.new_zeros({batch_size, top_k, out_feat});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "moe_first_linear_forward", ([&] {
        moe_first_linear_cuda_forward<scalar_t>(
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


/*
int main() {
    int device=2;
    torch::Tensor input = torch::randn({2048, 512}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    torch::Tensor gate = torch::zeros({2048, 2}, torch::dtype(torch::kInt64));
    torch::Tensor weight = torch::randn({2, 512, 2048}, torch::dtype(torch::kFloat32).device(torch::kCUDA, device));
    checkCudaErrors(cudaSetDevice(device));
    moe_cuda_forward(input, gate, weight);
}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &moe_first_linear_forward, "MoE first linear forward (CUDA)");
  // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}