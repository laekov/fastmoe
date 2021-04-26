#include <iostream>
#include <vector>
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
std::vector<torch::Tensor> _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers);
std::vector<torch::Tensor> _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);
std::vector<torch::Tensor> _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);
void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t);
#endif  // FMOE_USE_NCCL

std::vector<torch::Tensor> _expert_count(
        torch::Tensor gate, 
        size_t num_expert);
std::vector<torch::Tensor> _local_scatter(
    torch::Tensor input,
    torch::Tensor pos);
std::vector<torch::Tensor> _local_gather(
    torch::Tensor output_buf,
    torch::Tensor pos);

std::vector<torch::Tensor> _linear_forward(
        torch::Tensor input_buf,
        torch::Tensor weight,
        torch::Tensor expert_count);
std::vector<torch::Tensor> _linear_backward(
    torch::Tensor grad_output_buf, // [batch_size x out_feat]
    torch::Tensor input_buf, // [batch_size x out_feat]
    torch::Tensor weight, // [num_expert x out_feat x in_feat]
    torch::Tensor expert_count);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef FMOE_USE_NCCL
    m.def("expert_exchange", &_expert_exchange, "FastMoE expert exchange (CUDA)");
    m.def("global_scatter", &_global_scatter, "FastMoE global scatter (CUDA)");
    m.def("global_gather", &_global_gather, "FastMoE global gather (CUDA)");
    m.def("ensure_nccl", &_ensure_nccl, "FastMoE ensure torch nccl comm");
#endif

    m.def("expert_count", &_expert_count, "FastMoE expert count (CUDA)");
    m.def("local_scatter", &_local_scatter, "FastMoE local scatter (CUDA)");
    m.def("local_gather", &_local_gather, "FastMoE local gather (CUDA)");

    m.def("linear_forward", &_linear_forward, "FastMoE forward (CUDA)");
    m.def("linear_backward", &_linear_backward, "FastMoE backward (CUDA)");
}
