#include "balancing.cuh"
#include <torch/extension.h>

/* 
 * note that due to limit of cuda atomic operator, capacity should be int32
 */
torch::Tensor _limit_by_capacity(
        torch::Tensor expert_count, torch::Tensor capacity,
        long n_expert, long n_worker) {
    CHECK_INPUT(expert_count);
    CHECK_INPUT(capacity);
    auto expert_count_ack = torch::empty_like(expert_count);
    auto smgr = getCudaStreamManager(expert_count.device().index());
    fmoe_cuda_limit_by_capacity_impl(
            expert_count.data_ptr<long>(),
            capacity.data_ptr<int>(),
            expert_count_ack.data_ptr<long>(),
            n_expert, n_worker, smgr);
    return expert_count_ack;
}

torch::Tensor _prune_gate_by_capacity(
        torch::Tensor gate_idx, torch::Tensor expert_count,
        long n_expert, long n_worker) {
    auto smgr = getCudaStreamManager(expert_count.device().index());
    auto batch_size = gate_idx.numel();
    auto opt = torch::TensorOptions()
        .dtype(gate_idx.dtype())
        .device(gate_idx.device());
    auto new_gate_idx = torch::empty(gate_idx.sizes(), opt);
    fmoe_cuda_prune_gate_by_capacity_impl(
            gate_idx.data_ptr<long>(),
            new_gate_idx.data_ptr<long>(),
            expert_count.data_ptr<int>(),
            batch_size, n_expert, n_worker, smgr);
    return new_gate_idx;
}
