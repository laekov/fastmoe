#ifndef MOE_CUDA_KERNEL_H
#define MOE_CUDA_KERNEL_H

#include <vector>
#include <torch/extension.h>
#include <torch/torch.h>
#include "helper_cuda.h"

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
	torch::Tensor expert_count,
    torch::Tensor weight,
	at::optional<torch::Tensor> bias);

std::vector<torch::Tensor> moe_cuda_backward(
    torch::Tensor grad_output_buf,
    torch::Tensor input_buf,
	torch::Tensor expert_count,
    torch::Tensor weight,
	at::optional<torch::Tensor> bias);

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

std::vector<torch::Tensor> moe_cuda_expert_exchange(
	torch::Tensor local_expert_count,
	long num_expert, long n_workers);

std::vector<torch::Tensor> moe_cuda_global_fused_forward(
		torch::Tensor input_buf,
        torch::Tensor weight,
		torch::Tensor local_expert_count,
		torch::Tensor global_expert_count,
		long global_batch_size, long local_batch_size, long n_workers);

#endif 

#endif  // MOE_CUDA_KERNEL_H
