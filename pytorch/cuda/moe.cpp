#include <torch/extension.h>

#include <cstdio>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> moe_cuda_expert_count(
    torch::Tensor gate, size_t num_expert);

std::vector<torch::Tensor> moe_cuda_local_scatter(
    torch::Tensor input,
	torch::Tensor pos);

std::vector<torch::Tensor> moe_cuda_local_gather(
	torch::Tensor output_buf,
	torch::Tensor pos);

std::vector<torch::Tensor> moe_cuda_forward(
    torch::Tensor input_buf,
    torch::Tensor weight,
	torch::Tensor expert_count);

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output_buf,
    torch::Tensor input_buf,
    torch::Tensor weight,
	torch::Tensor expert_count);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> moe_expert_count(
		torch::Tensor gate, 
		size_t num_expert) {
	CHECK_INPUT(gate);
	return moe_cuda_expert_count(gate, num_expert);
}

std::vector<torch::Tensor> moe_local_scatter(
		torch::Tensor input,
		torch::Tensor pos) {
	CHECK_INPUT(input);
	return moe_cuda_local_scatter(input, pos);
}

std::vector<torch::Tensor> moe_local_gather(
		torch::Tensor output_buf,
		torch::Tensor pos) {
	CHECK_INPUT(output_buf);
	return moe_cuda_local_gather(output_buf, pos);
}


std::vector<torch::Tensor> moe_forward(
        torch::Tensor input_buf,     // [batch_size x in_feat]
        torch::Tensor weight,        // [num_expert x out_feat x in_feat]
        torch::Tensor expert_count   // [batch_size]
        ) {
    CHECK_INPUT(input_buf);
    CHECK_INPUT(weight);
    /*
        The bias term should have been merged into weight. Note the following fact that 
        Wx+b = [W b] [x]
                     [1]  
    */
    return moe_cuda_forward(input_buf, weight, expert_count);
}

std::vector<torch::Tensor> moe_backward(
        torch::Tensor grad_output_buf, // [batch_size x out_feat]
        torch::Tensor input_buf,       // [batch_size x out_feat]
        torch::Tensor weight,          // [num_expert x out_feat x in_feat]
        torch::Tensor expert_count
        ) {
    CHECK_INPUT(grad_output_buf);
    CHECK_INPUT(input_buf);
    CHECK_INPUT(weight);
    /*
        The bias term should have been merged into weight. Note the following fact that 
        Wx+b = [W b] [x]
                     [1]  
    */
    return moe_cuda_backward(grad_output_buf, input_buf, weight, expert_count);
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
  m.def("expert_count", &moe_expert_count, "MoE expert count (CUDA)");
  m.def("local_scatter", &moe_local_scatter, "MoE local scatter (CUDA)");
  m.def("local_gather", &moe_local_gather, "MoE local gather (CUDA)");
  m.def("forward", &moe_forward, "MoE forward (CUDA)");
  m.def("backward", &moe_backward, "MoE backward (CUDA)");
}
