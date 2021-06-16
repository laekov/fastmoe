#include "local_exchange.cuh"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

void _assign_pos(
    torch::Tensor cum_count,
    torch::Tensor gate,
    torch::Tensor pos) {
    auto smgr = getCudaStreamManager(cum_count.device().index());
    auto gate_shp = gate.sizes();
    size_t batch_size = gate_shp[0], topk = 1;
    if (gate_shp.size() == 2) {
        topk = gate_shp[1];
    }
    fmoe_cuda_assign_pos_impl(
            cum_count.data_ptr<int>(),
            gate.data_ptr<long>(),
            pos.data_ptr<long>(),
            batch_size, topk, smgr);
}

void _expert_count(
        torch::Tensor gate_idx,
        torch::Tensor expert_count) {
    auto smgr = getCudaStreamManager(gate_idx.device().index());
    auto batch_size = gate_idx.numel();
    auto n_expert = expert_count.numel();
    fmoe_cuda_expert_count_impl(
            gate_idx.data_ptr<long>(),
            expert_count.data_ptr<int>(),
            batch_size, n_expert, smgr);
}
