#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> moe_cuda_forward(
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight);

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gate,
    torch::Tensor weight);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> moe_forward(
        torch::Tensor input, // [batch_size x in_feat]
        torch::Tensor gate,  // [batch_size]
        torch::Tensor weight // [num_expert x out_feat x in_feat]
        ) {
    CHECK_INPUT(input);
    CHECK_INPUT(gate);
    CHECK_INPUT(weight);
    /*
        The bias term should have been merged into weight. Note the following fact that 
        Wx+b = [W b] [x]
                     [1]  
    */
    return moe_cuda_forward(input, gate, weight);
}

std::vector<torch::Tensor> moe_backward(
        torch::Tensor grad_output, // [batch_size x out_feat]
        torch::Tensor input, // [batch_size x out_feat]
        torch::Tensor gate,  // [batch_size]
        torch::Tensor weight // [num_expert x out_feat x in_feat]
        ) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(gate);
    CHECK_INPUT(weight);
    /*
        The bias term should have been merged into weight. Note the following fact that 
        Wx+b = [W b] [x]
                     [1]  
    */
    return moe_cuda_backward(grad_output, input, gate, weight);
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
  m.def("forward", &moe_forward, "MoE forward (CUDA)");
  m.def("backward", &moe_backward, "MoE backward (CUDA)");
}