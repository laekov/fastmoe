#ifndef MOE_CUDA_KERNEL_H
#define MOE_CUDA_KERNEL_H

#include <vector>
#include <torch/extension.h>
#include <torch/torch.h>

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

#ifdef MOE_USE_NCCL

std::vector<torch::Tensor> moe_cuda_global_scatter(
    torch::Tensor input_buf,
	torch::Tensor local_expert_count,
	torch::Tensor global_expert_count,
	long batch_size, long n_workers);

std::vector<torch::Tensor> moe_cuda_global_gather(
	torch::Tensor output_buf,
	torch::Tensor local_expert_count,
	torch::Tensor global_expert_count,
	long batch_size, long n_workers);

#endif 

#endif  // MOE_CUDA_KERNEL_H
