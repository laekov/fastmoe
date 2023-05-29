#include <iostream>
#include <vector>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

// global_exchange
#ifdef FMOE_USE_NCCL

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13))
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#else
#include <c10d/ProcessGroupNCCL.hpp>
#endif

torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers);
torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);
torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers);
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
void _ensure_nccl(c10d::ProcessGroup& p, torch::Tensor t);
#else
void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t);
#endif  // TORCH_VERSION

#endif  // FMOE_USE_NCCL

// local_exchange
void _assign_pos(
        torch::Tensor cum_count,
        torch::Tensor gate,
        torch::Tensor pos);
void _expert_count(
        torch::Tensor gate_idx,
        torch::Tensor expert_count);

// parallel_linear
torch::Tensor _linear_forward(
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        );
std::vector<torch::Tensor> _linear_backward(
        torch::Tensor grad_output_buf,
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        );

// balancing
torch::Tensor _limit_by_capacity(
        torch::Tensor expert_count, torch::Tensor capacity,
        long n_expert, long n_experts);
torch::Tensor _prune_gate_by_capacity(
        torch::Tensor gate_idx, torch::Tensor expert_count,
        long n_expert, long n_worker);
std::vector<torch::Tensor> _swipe_once(
        torch::Tensor gate_idx, torch::Tensor capacity_tensor,
        long n_expert, long n_worker, long bias);

// smart scheduling
std::vector<torch::Tensor> _smart_sch_forward(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long global_batch_size,
        long expert_size,
        long n_workers,
        py::function forward_fn,
        py::function get_param_fn,
        py::function stash_fn,
        py::function pop_fn);
torch::Tensor _smart_sch_backward(
        torch::Tensor grad_out,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long buf_batch_size,
        long global_batch_size,
        long n_workers,
        py::function backward_fn,
        py::function stash_fn,
        py::function pop_fn,
        py::function collect_fn,
        py::function set_grad_fn);
void _reduce_grad(
        torch::Tensor t,
        long root,
        long expert_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef FMOE_USE_NCCL
    m.def("expert_exchange", &_expert_exchange, "FastMoE expert exchange (CUDA)");
    m.def("global_scatter", &_global_scatter, "FastMoE global scatter (CUDA)");
    m.def("global_gather", &_global_gather, "FastMoE global gather (CUDA)");
    m.def("ensure_nccl", &_ensure_nccl, "FastMoE ensure torch nccl comm");
    m.def("swipe_once", &_swipe_once, "SWIPE balance strategy(CUDA)");

    m.def("smart_sch_forward", &_smart_sch_forward, "E2E MoE layer forward with smart scheduling");
    m.def("smart_sch_backward", &_smart_sch_backward, "E2E MoE layer backward with smart scheduling");
    m.def("reduce_grad", &_reduce_grad, "Reduce gradients over FastMoE's communication stream");
#endif

    m.def("expert_count", &_expert_count, "FastMoE count gate indices (CUDA)");
    m.def("assign_pos", &_assign_pos, "FastMoE assign pos by gate (CUDA)");

    m.def("linear_forward", &_linear_forward, "FastMoE forward (CUDA)");
    m.def("linear_backward", &_linear_backward, "FastMoE backward (CUDA)");

    m.def("limit_by_capacity", &_limit_by_capacity, "FastMoE limit experts by capacity(CUDA)");
    m.def("prune_gate_by_capacity", &_prune_gate_by_capacity, "FastMoE prune gate by capacity(CUDA)");
}
