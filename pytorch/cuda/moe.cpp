#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> moe1_cuda_forward(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> moe1_forward(
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
    return moe1_cuda_forward(input, gate, weight);
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
  m.def("forward", &moe1_forward, "MoE first linear forward (CUDA)");
  // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}